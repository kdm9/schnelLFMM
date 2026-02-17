# Out-of-Core LFMM2 in Rust — Design Doc

**Reference**: Caye et al. (2019) "LFMM 2: Fast and Accurate Inference of Gene-Environment
Associations in Genome-Wide Studies." *Molecular Biology and Evolution*, 36(4), 852–860.
https://doi.org/10.1093/molbev/msz008

## Problem

Implement LFMM2 for GWAS-scale data: **n ≈ 10k samples, p ≈ 1e9 SNPs, d ≈ 1–100 covariates, K ≈ 5–50 latent factors**.

Y (n × p) is ~2.5 TB on disk in PLINK .bed format (2-bit packed) and cannot fit in RAM as
f64 (~80 TB). Everything involving only n fits comfortably in RAM. B (p × d) and test
statistics (p × d) are large outputs written to disk.

## Model

> **Paper Eq. 1**: Y = X B^T + W + E, where W = U V^T is rank K.

- Y: n × p response matrix (genotypes, methylation, etc.)
- X: n × d matrix of primary/nuisance variables
- B: p × d fixed effect sizes
- U: n × K latent factors (confounders)
- V: p × K loadings
- E: n × p residual error

## Objective

> **Paper Eq. 2**: L_ridge(U,V,B) = ||Y − UV^T − XB^T||_F^2 + λ||B||_2^2, λ > 0

The regularization parameter λ > 0 is required for identifiability (at λ=0 the solution
is not unique; the λ=0 case reduces to PCA-then-regress).

## Practical Workflow Note

> **Paper, Materials & Methods / GEA Study**: "For all methods the latent factors were
> estimated from the pruned genotypes, and association tests were performed for all
> 5,397,214 loci."

In practice, **latent factor estimation** (Steps 1–3 below) is performed on an LD-pruned
subset of SNPs (Y_pruned, n × p_pruned), and then **testing** (Step 4) is run on all p SNPs.
The design should support specifying separate Y matrices for estimation vs. testing.

---

## Algorithm

### Inputs
- `Y_est`: n × p_est genotype matrix for factor estimation (LD-pruned subset, PLINK .bed format on disk)
- `Y_full`: n × p genotype matrix for testing (all SNPs, PLINK .bed format on disk)
- `X`: n × d covariate matrix (in RAM)
- `λ`: ridge penalty
- `K`: number of latent factors
- `q`: number of power iterations (default: 2)

All .bed inputs are 2-bit packed (SNP-major). Each streaming pass decodes chunks on-the-fly
to centered f64, imputing missing genotypes to the per-SNP mean (see Storage Format below).

---

### Step 0: Precomputation (all in RAM)

> **Paper, Ridge Estimates**: "The minimization algorithm starts with an SVD of the
> explanatory matrix, X = Q Σ R^T, where Q is an n×n unitary matrix, R is a d×d unitary
> matrix and Σ is an n×d matrix containing the singular values of X."

```
SVD of X:  X = Q Σ R^T              # Q: n×n, Σ: n×d, R: d×d
```

> **Paper, Ridge Estimates**: "D_λ is the n×n diagonal matrix with coefficients defined as
> d_λ = (λ/(λ+σ₁²), ..., λ/(λ+σ_d²), 1, ..., 1)"

```
D_λ = diag([λ/(λ + σ_j²) for j in 0..d] ++ [1.0; n-d])   # n × n
D_λ_inv = diag(1 / d_λ)                                      # n × n
M = D_λ @ Q^T                                                 # n × n
ridge_inv = inv(X^T X + λ I_d)                                # d × d
```

All of these are n×n or smaller — fully in RAM at n=10k (~800 MB).

---

### Steps 1–2: Compute Ŵ via randomized SVD

> **Paper Eq. 3**: Ŵ = Q D_λ^{-1} svd_K(D_λ Q^T Y)

We need the rank-K SVD of `A = D_λ Q^T Y = M @ Y_est`, which is n × p_est (fat matrix,
n << p_est). Following Halko et al. (2011) — the reference cited in the paper for O(np log K)
complexity — we use randomized SVD with power iterations, sketching from the left since the
matrix is fat.

**The core identity**: since `A = M @ Y_est`, we never form A explicitly. Instead we
stream Y_est and apply M (n×n, in RAM) on the fly.

#### Step 1a: Initial sketch — Z = A^T Ω = Y_est^T (M^T Ω)

```
Ω = randn(n, K + oversampling)      # n × ~60, in RAM
MtΩ = M^T @ Ω                        # n × 60, precomputed in RAM
```

**Pass 1** — stream Y_est by column-chunks:
```
for bed_chunk in stream(Y_est.bed, by_columns):
    Y_chunk = decode_bed_chunk(bed_chunk)            # 2-bit → f64, impute missing → mean, center
    Z[chunk_cols, :] = Y_chunk^T @ MtΩ               # chunk_size × 60, write to disk
```
Z is p_est × 60 (~5 GB for p_est=1e8). Compute QR: `Q_z, R_z = QR(Z)`.

#### Step 1b: Power iterations (q passes, each = 2 streaming passes)

> **Paper, Ridge Estimates**: references Halko et al. (2011) for the randomized SVD.
> Power iteration improves accuracy for matrices with slow singular value decay.

Each power iteration refines the column space estimate by multiplying by A A^T
(= M Y Y^T M^T) and re-orthogonalizing. Since A is never formed, each iteration requires
one pass to compute `Y^T @ (...)` and one pass to compute `Y @ (...)`:

```
for i in 0..q:
    # Pass 2i+2: Compute A @ Q_z = M @ Y_est @ Q_z  (result: n × 60, in RAM)
    AQz = zeros(n, K + oversampling)
    for bed_chunk in stream(Y_est.bed, by_columns):
        Y_chunk = decode_bed_chunk(bed_chunk)        # 2-bit → f64, impute, center
        AQz += M @ Y_chunk @ Q_z[chunk_cols, :]

    QR orthogonalize AQz → Q_aqz   # n × 60, in RAM

    # Pass 2i+3: Compute A^T @ Q_aqz = Y_est^T @ (M^T @ Q_aqz)  (result: p_est × 60)
    MtQ = M^T @ Q_aqz               # n × 60, precomputed
    for bed_chunk in stream(Y_est.bed, by_columns):
        Y_chunk = decode_bed_chunk(bed_chunk)        # 2-bit → f64, impute, center
        Z[chunk_cols, :] = Y_chunk^T @ MtQ

    QR of Z → Q_z, R_z              # update Q_z
```

With q=2 power iterations, this adds 4 streaming passes.

#### Step 2: Project and recover SVD

**Pass (2q+2)**: Compute `B_svd = A @ Q_z = M @ Y_est @ Q_z` (same structure as the
power iteration forward pass):
```
B_svd = zeros(n, K + oversampling)
for bed_chunk in stream(Y_est.bed, by_columns):
    Y_chunk = decode_bed_chunk(bed_chunk)            # 2-bit → f64, impute, center
    B_svd += M @ Y_chunk @ Q_z[chunk_cols, :]        # accumulate n × 60
```

Small SVD of B_svd (n × 60, trivial):
```
U_small, s, Vt_small = SVD(B_svd)
```

Recover the LFMM latent factors:

> **Paper Eq. 3**: Ŵ = Q D_λ^{-1} svd_K(D_λ Q^T Y)
> The U from svd_K lives in the column space of D_λ Q^T Y. Lifting back:
> Û = Q D_λ^{-1} U_small[:, :K]

```
U_hat = Q @ D_λ_inv @ U_small[:, :K]     # n × K, in RAM
```

V is p_est × K — recover from the randomized SVD:
```
V_hat[chunk_cols, :] = Q_z[chunk_cols, :] @ Vt_small[:K, :].T @ diag(1/s[:K])
```
V_hat is large (p_est × K), stream to disk if needed. However, V_hat from the estimation
subset is only needed transiently — the important outputs are U_hat (for testing) and B
(from Step 3).

---

### Step 3: Compute B̂ (effect sizes) via ridge regression

> **Paper Eq. 4**: B̂^T = (X^T X + λ I_d)^{-1} X^T (Y − Ŵ)

Since Ŵ = U_hat V_hat^T, and we need B for **all p SNPs** (not just the estimation subset),
this step streams over Y_full:

```
XtR = ridge_inv @ X^T                    # d × n, precomputed in RAM
```

**Pass over Y_full** — stream by column-chunks:
```
for bed_chunk in stream(Y_full.bed, by_columns):
    Y_chunk = decode_bed_chunk(bed_chunk)            # 2-bit → f64, impute, center
    # We need V_hat for these columns — but for SNPs outside Y_est, V_hat is not available.
    # Instead, compute the residual directly: Y - Ŵ = Y - U_hat V_hat^T
    # For the full-SNP case, we re-derive V from the regression:
    #   v_ℓ^T is re-estimated in Step 4, so here we use:
    #   Ŵ_ℓ = U_hat @ (U_hat^T U_hat)^{-1} @ U_hat^T @ Y_ℓ  (projection)
    # But more directly: B^T = ridge_inv @ X^T @ (Y - U_hat V_hat^T)
    # where V_hat for the full set can be computed as V_ℓ = Y_ℓ^T @ U_hat @ inv(U_hat^T U_hat)
    
    # Simplification: define P_U = U_hat @ inv(U_hat^T U_hat) @ U_hat^T  (n×n, in RAM)
    # Then Y - Ŵ ≈ (I - P_U) @ Y for the projection interpretation
    # And B_chunk = (XtR @ (I - P_U) @ Y_chunk)^T
    
    residual = Y_chunk - P_U @ Y_chunk
    B[chunk_cols, :] = (XtR @ residual).T              # chunk_size × d, write to disk
```

**Note**: P_U = U_hat @ inv(U_hat^T U_hat) @ U_hat^T is n×n, precomputed once in RAM.
Since U_hat = Q D_λ_inv U_small[:,:K], its columns are generally NOT orthonormal — the
Gram matrix correction inv(U_hat^T U_hat) is required.

---

### Step 4: Statistical testing (per-locus)

> **Paper Eq. 5**: "To test association between the primary variables and each response
> variable, Y_ℓ, we use the latent score estimates obtained from the LFMM model as
> covariates in multivariate regression models":
>
>   y_ℓ = x β_ℓ + Û v_ℓ^T + e_ℓ,  ℓ = 1, ..., p
>
> "To test the null hypothesis H₀: β_ℓ = 0, we use a Student distribution with n−K−1
> degrees of freedom."

This step runs a **separate regression per locus** on all p SNPs (Y_full), with Û_hat as
fixed covariates. This is distinct from Step 3 — here we re-estimate v_ℓ jointly with β_ℓ.

Design matrix per locus: `C = [X | U_hat]`, shape n × (d + K). This is the same for all
loci and fits in RAM.

Precompute (once, in RAM):
```
C = hstack(X, U_hat)                     # n × (d+K)
CtC_inv = inv(C^T @ C)                   # (d+K) × (d+K)
H = CtC_inv @ C^T                        # (d+K) × n  — the "hat" coefficients
```

**Pass over Y_full** — stream by column-chunks:
```
for bed_chunk in stream(Y_full.bed, by_columns):
    Y_chunk = decode_bed_chunk(bed_chunk)            # 2-bit → f64, impute, center
    # Coefficients: [β; v]_chunk = H @ Y_chunk    # (d+K) × chunk_size
    coefs = H @ Y_chunk
    
    # Residuals
    residuals = Y_chunk - C @ coefs                # n × chunk_size
    
    # Residual variance per locus
    rss = col_sum_of_squares(residuals)            # chunk_size
    sigma2 = rss / (n - d - K)                     # df = n - K - d (paper says n-K-1 for d=1)
    
    # Standard errors for β (first d rows of coefs)
    # Var(β_ℓ) = σ²_ℓ * (C^T C)^{-1}[0:d, 0:d]
    # For each primary variable j in 0..d:
    se_beta = sqrt(sigma2 * CtC_inv[j, j])         # chunk_size
    
    # t-statistics and p-values
    t_stat = coefs[j, :] / se_beta                 # chunk_size
    p_values = 2 * t_cdf(-abs(t_stat), df=n-d-K)  # two-sided
    
    # Write to disk: t_stat, p_values for chunk_cols
```

**Optimization**: Steps 3 and 4 can be **fused into a single pass** over Y_full since both
require the same column-chunks. Compute B and test statistics simultaneously per chunk.

#### Empirical null calibration (genomic control)

> **Paper, Statistical Tests**: "To improve test calibration and false discovery rate
> estimation, we eventually apply an empirical-null testing approach to the test statistics
> (Efron 2004)."

After computing all p z-scores (or t-statistics), fit the empirical null:
```
# Estimate the empirical null N(μ₀, σ₀²) from the central mass of z-scores
# (e.g., using Efron's method or simple genomic inflation factor)
GIF = median(z²) / 0.456        # genomic inflation factor (χ² with df=1)
z_calibrated = z / sqrt(GIF)
p_calibrated = 2 * Φ(-|z_calibrated|)
```

This calibration step operates on the full vector of p-values/z-scores and can be done
in a streaming fashion (compute median via streaming quantile) or after collecting all
statistics.

---

## Streaming Pass Summary

| Pass | Over | Purpose | RAM accumulator |
|------|------|---------|-----------------|
| 1 | Y_est | Z = Y^T (M^T Ω) — initial sketch | write Z (p_est×60) to disk |
| 2..2q+1 | Y_est | Power iterations (q iterations × 2 passes each) | n×60 + write Z to disk |
| 2q+2 | Y_est | B_svd = M Y Q_z — final projection | n × 60 |
| 2q+3 | Y_full | B + per-locus t-tests (Steps 3 & 4 fused) | write B + stats to disk |

**With q=2**: 1 (sketch) + 4 (power iter) + 1 (project) + 1 (B + test fused) = **7 passes**.

Note: if Y_est is an LD-pruned subset (e.g., 1% of SNPs), the first 6 passes read only
~1% of Y_full. Only the final fused pass reads all of Y_full.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Y on disk: PLINK .bed (2-bit packed, SNP-major)      │
└──────┬───────┬───────┬──────────┬───────────────────┘
       │       │       │          │
  ┌────▼───────▼───────▼──────────▼────────────────┐
  │  decode_bed_chunk(): 2-bit → f64               │
  │  impute missing (0b01) → per-SNP mean          │
  │  center: subtract per-SNP mean                 │
  └────┬───────┬───────┬──────────┬────────────────┘
       │ Pass 1│ 2..2q+1│ Pass 2q+2│ Pass 2q+3
       │sketch │power it│ project  │ B + testing (fused)
       ▼       ▼        ▼          ▼
  Z=Yᵀ(MᵀΩ) refine Q_z  B_svd   [B, β, t, p-val per locus]
       │       │        │          │
  ┌────▼───────▼────────▼──────────▼────────┐
  │  Small dense ops (ndarray-linalg/MKL)    │
  │  QR, SVD, inv — all ≤ n×n               │
  └──────────────────────────────────────────┘
```

Each pass is embarrassingly parallel over column-chunks via rayon.

---

## Memory Budget

| Object | Shape | Size (f64, n=10k) | Location |
|--------|-------|-------------------|----------|
| X, Q, M, D_λ, D_λ_inv | n × n | ~800 MB | RAM |
| P_U | n × n | ~800 MB | RAM |
| Ω, MtΩ | n × 60 | ~5 MB | RAM |
| ridge_inv | d × d | < 1 MB | RAM |
| C, CtC_inv, H | n×(d+K), (d+K)², (d+K)×n | < 100 MB | RAM |
| Y chunk (packed on disk) | ceil(n/4) × chunk_size | 250 MB at 100k SNPs | disk read |
| Y chunk (decoded f64) | n × chunk_size | ~8 GB at 100k SNPs | RAM (transient) |
| Z, Q_z | p_est × 60 | ~5 GB at p_est=1e8 | RAM or disk |
| B_svd | n × 60 | ~5 MB | RAM |
| U_hat | n × K | ~4 MB | RAM |
| B | p × d | up to 800 GB | disk (streamed out) |
| t-stats, p-values | p × d | up to 800 GB | disk (streamed out) |

**Peak RAM: ~2–10 GB** depending on whether Z/Q_z are held in RAM or streamed from disk.

---

## Crate Stack

| Crate | Purpose |
|-------|---------|
| `ndarray` | Array types, chunk-level ops |
| `ndarray-linalg` (MKL backend) | SVD, QR, inv on small dense matrices |
| `rayon` | Parallel iteration over chunks |
| `memmap2` | Memory-mapped access to .bed files on disk |
| `rand` / `rand_distr` | Random sketch matrix Ω |
| `statrs` | Student's t distribution for p-values |
| `bed-reader` (optional) | PLINK .bed/.bim/.fam parsing (or implement custom 2-bit decoder) |

---

## Storage Format for Y

### Primary format: PLINK .bed (2-bit packed genotypes)

Genotype data is inherently low-entropy (values 0, 1, 2, or missing). PLINK's .bed format
stores each genotype in **2 bits**, achieving 4 genotypes per byte:

| Format | Bits/genotype | 10k × 1e9 size |
|--------|--------------|----------------|
| f64 | 64 | 80 TB |
| f32 / f16 | 32 / 16 | 40 / 20 TB |
| u8 | 8 | 10 TB |
| **2-bit packed (.bed)** | **2** | **2.5 TB** |
| 2-bit + zstd | ~0.5–1 | ~0.6–1.2 TB |

PLINK .bed encoding (per genotype, 2 bits):
```
0b00 = homozygous ref  → 0.0
0b01 = missing          → impute to SNP mean (see below)
0b10 = heterozygous     → 1.0
0b11 = homozygous alt   → 2.0
```

Note: PLINK .bed is **SNP-major** (column-major for our purposes): each SNP's genotypes
for all n samples are stored contiguously in ceil(n/4) bytes. This is ideal for our
column-chunked streaming — a chunk of `chunk_size` SNPs is a contiguous byte range.

### On-the-fly unpack, impute, and center

Each streaming pass reads raw 2-bit packed bytes and produces a centered f64 chunk in RAM.
The decode pipeline per column-chunk:

```rust
/// Decode a chunk of SNPs from .bed format into a centered f64 matrix.
/// Missing values (0b01) are imputed to the per-SNP mean genotype.
fn decode_bed_chunk(
    packed: &[u8],        // raw .bed bytes for this chunk
    n_samples: usize,
    chunk_size: usize,
) -> Array2<f64> {
    let bytes_per_snp = (n_samples + 3) / 4;  // ceil(n/4)
    let mut out = Array2::<f64>::zeros((n_samples, chunk_size));

    for snp in 0..chunk_size {
        let snp_bytes = &packed[snp * bytes_per_snp..(snp + 1) * bytes_per_snp];
        let col = out.column_mut(snp);

        // Pass 1: decode and compute mean (excluding missing)
        let mut sum = 0.0f64;
        let mut n_valid = 0u32;
        for sample in 0..n_samples {
            let byte = snp_bytes[sample / 4];
            let code = (byte >> (2 * (sample % 4))) & 0x03;
            let val = match code {
                0b00 => 0.0,
                0b10 => 1.0,
                0b11 => 2.0,
                0b01 => f64::NAN,   // mark missing, impute below
                _ => unreachable!(),
            };
            col[sample] = val;
            if !val.is_nan() {
                sum += val;
                n_valid += 1;
            }
        }
        let mean = if n_valid > 0 { sum / n_valid as f64 } else { 0.0 };

        // Pass 2: impute missing → mean, then center all values
        for sample in 0..n_samples {
            if col[sample].is_nan() {
                col[sample] = 0.0;   // missing → mean, and mean - mean = 0
            } else {
                col[sample] -= mean;  // center
            }
        }
    }
    out
}
```

**Performance notes on decoding**:
- The inner loop is trivially vectorizable with SIMD (lookup table on 8 genotypes at a time).
- Decode cost is negligible vs. the BLAS matmul that follows (~0.1% of chunk time).
- Missing rate in typical GWAS data is <1%, so branch misprediction is minimal.
- Optional: apply unit-variance scaling by `1/sqrt(2*p*(1-p))` where p = mean/2 (allele freq),
  as done in EIGENSTRAT-style analyses. The paper uses different scaling for GEA vs. EWAS.

### File layout options


**Single PLINK .bed file** (standard): SNP-major layout, read with seek offsets
per chunk. Chunk offset = `3 + chunk_start * bytes_per_snp` (3-byte .bed magic
header). Simple mmap via `memmap2` works well. Ideally: have one thread solely
responsible for IO, seeking to maximise streaming IO. Assume the underlying FS
(which may be raid HDD array, NFS, etc, not just fast local nvme) has best
performance in streaming mode.

In addition to an alternative BED for estimation, support supplying a simple
random scale factor to do simple thinning of data for estimation subset, or a
list of SNPs to include by bed file (as in 3+ column chrom\tstart\tend BED, not
plink BED, they are two completely different formats).

---

## Performance Notes

> **Paper, Ridge Estimates**: "the complexity of the estimation algorithm is of order
> O(n²p + np log K)"

With 2-bit packed .bed format:
```
Y_full = 2.5 TB  (2-bit packed, 10k × 1e9)
Y_est  = 25 GB   (1% LD-pruned subset)

6 estimation passes over Y_est:  6 × 25 GB  =  150 GB
1 testing pass over Y_full:      1 × 2.5 TB = 2500 GB
Total I/O: ~2.7 TB

At 3 GB/s NVMe: ~15 minutes
```

- Decode overhead (2-bit → f64 + impute + center) is <1% of total time per chunk.
- BLAS matmul on the decoded f64 chunk dominates compute.
- Column-chunk size is tunable: larger = better BLAS throughput, smaller = less RAM.
  At chunk_size=1M SNPs: f64 chunk = n × 1M × 8 bytes = 80 GB (too large).
  At chunk_size=100k SNPs: f64 chunk = n × 100k × 8 = 8 GB (reasonable).
  Packed on disk: 100k SNPs = n/4 × 100k = 250 MB per chunk read.
- The final fused pass (B + testing) is the I/O bottleneck and is embarrassingly parallel.
- Optional zstd compression on .bed chunks can further reduce I/O by ~2–4× at the cost
  of CPU decompression (typically still I/O-bound on spinning disks, break-even on NVMe).

---

## API Sketch

```rust
pub struct Lfmm2Config {
    pub k: usize,              // latent factors (K)
    pub lambda: f64,           // ridge penalty (λ)
    pub chunk_size: usize,     // SNPs per chunk (default: 100_000)
    pub oversampling: usize,   // randomized SVD oversampling (default: 10)
    pub n_power_iter: usize,   // power iterations q (default: 2)
}

/// On-disk genotype source (PLINK .bed + .bim + .fam triplet).
pub struct BedFile {
    pub bed_path: PathBuf,     // .bed file (2-bit packed, SNP-major)
    pub n_samples: usize,      // from .fam
    pub n_snps: usize,         // from .bim
    pub subset_rate: Option<f64>,  // subset this proportion of SNPs
    pub snp_subset: Option<PathBuf>,  // subset specifically these SNPs (must have two columns, chrom and pos, can have more)
}

pub struct Lfmm2Result {
    pub u_hat: Array2<f64>,         // n × K latent factors, in RAM
    pub b_path: PathBuf,            // p × d effect sizes, on disk
    pub scores_path: PathBuf,       // p × d t-statistics, on disk
    pub pvalues_path: PathBuf,      // p × d p-values (calibrated), on disk
    pub gif: f64,                   // genomic inflation factor
}

/// Estimate latent factors from (possibly LD-pruned) Y_est.
pub fn estimate_factors(
    y_est: &BedFile,
    x: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<Array2<f64>>;   // returns U_hat (n × K)

/// Run association tests on all SNPs using pre-estimated U_hat.
/// Computes B and per-locus t-tests in a single fused pass.
pub fn test_associations(
    y_full: &BedFile,
    x: &Array2<f64>,
    u_hat: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<Lfmm2Result>;

/// Full pipeline: estimate + test.
pub fn fit_lfmm2(
    y_est: &BedFile,           // LD-pruned subset for factor estimation
    y_full: &BedFile,          // all SNPs for testing
    x: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<Lfmm2Result>;
```

---

## Cross-reference to Paper

| Design step | Paper reference |
|-------------|----------------|
| Model Y = XB^T + UV^T + E | Eq. 1 |
| Ridge loss function | Eq. 2 |
| SVD of X, D_λ construction | Ridge Estimates section, below Eq. 2 |
| Ŵ = Q D_λ^{-1} svd_K(D_λ Q^T Y) | Eq. 3 |
| Randomized SVD with power iter | Halko et al. 2011 (cited in paper) |
| B̂^T = (X^TX + λI)^{-1} X^T(Y − Ŵ) | Eq. 4 |
| Per-locus regression with Û as covariates | Eq. 5 |
| Student t-test, df = n−K−d | Statistical Tests section |
| Empirical null / genomic inflation | Statistical Tests: "empirical-null testing approach (Efron 2004)" |
| LD-pruned estimation, full testing | Materials & Methods, GEA Study section |
| Complexity O(n²p + np log K) | Ridge Estimates section, final paragraph |
