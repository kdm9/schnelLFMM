# Mathematical Notes: LFMM2 with NMF Imputation

## 1. Genotype Encoding and Preprocessing

### 1.1 PLINK BED Format

Genotypes are stored in PLINK 1 BED format with the standard 2-bit encoding:

| PLINK Code | Genotype       | Decoded Value |
|------------|---------------|---------------|
| `00`       | Homozygous reference | 0 |
| `01`       | Missing              | `NaN` |
| `10`       | Heterozygous         | 1 |
| `11`       | Homozygous alternate | 2 |

A BED file consists of a 3-byte magic header (`0x6C 0x1B 0x01`), followed by SNP-major packed data. Each SNP occupies $\lceil n_{\text{phys}}/4 \rceil$ bytes where $n_{\text{phys}}$ is the number of physical samples on disk. Optionally, the software subsets to $n$ output samples via `sample_keep`.

### 1.2 Genotype Normalization

Each SNP column $y_j \in \mathbb{R}^n$ (after raw decode to $\{0, 1, 2\}$) is centered and optionally scaled. Let $\bar{y}_j$ be the column mean computed over non-missing entries.

**Mean-based centering:**
$$
\tilde{y}_{ij} = y_{ij} - \bar{y}_j
$$

**Eigenstrat normalization:** additionally divide by $\sqrt{2 p_j (1-p_j)}$ where $p_j = \bar{y}_j / 2$:

$$
\tilde{y}_{ij}^{\text{eigen}} = \frac{y_{ij} - 2p_j}{\sqrt{2p_j(1-p_j)}}
$$

This yields approximately unit-variance genotypes under Hardy-Weinberg equilibrium. Monomorphic SNPs ($2p(1-p) < 10^{-10}$) are set to zero to avoid division by zero.

### 1.3 Missing Data Strategies

The software supports three imputation strategies for handling missing genotypes (NaN entries in the raw decode):

1. **Mean imputation** (`ImputeConfig::Mean`): Missing values are filled with the column mean before centering — after centering, these become exactly 0.

2. **NMF in-RAM** (`ImputeConfig::NmfInRam`): Missing values are filled with the NMF reconstruction $W H_{\text{chunk}}$ before centering, where $W$ and $H$ were pre-computed during the estimation phase.

3. **NMF on-the-fly** (`ImputeConfig::NmfOnTheFly`): $H_{\text{chunk}}$ is estimated per chunk as $\max(0, W^{\dagger} Y_{\text{filled}})$, then missing values are filled with $W H_{\text{chunk}}$ before centering (see §8).

---

## 2. Covariate Preprocessing

Given a covariate matrix $X \in \mathbb{R}^{n \times d}$:

$$
X \leftarrow \text{center}(X)
$$

Optionally, if `--scale-cov` is set, each column is further divided by its standard deviation:

$$
x_{ij} \leftarrow \frac{x_{ij} - \bar{x}_j}{\text{sd}(x_j)}
$$

---

## 3. Step 0: Precomputation of X-Related Quantities

Let $X \in \mathbb{R}^{n \times d}$ be the centered (and optionally scaled) covariate matrix. Perform:

### 3.1 SVD of X

Compute the full SVD:

$$
X = Q \Sigma R^T
$$

where $Q \in \mathbb{R}^{n \times n}$ is orthogonal, $\Sigma$ contains singular values $\sigma_1, \ldots, \sigma_d$ (for $n > d$, only $\min(n, d)$ are non-zero), and $R \in \mathbb{R}^{d \times d}$ is orthogonal.

### 3.2 Ridge-Regularized Weight Matrix

For ridge penalty $\lambda > 0$, define the diagonal matrix $D_\lambda \in \mathbb{R}^{n \times n}$ with entries:

$$
d_\lambda[j] = \begin{cases}
\sqrt{\frac{\lambda}{\lambda + \sigma_j^2}}, & j < \min(n, d) \\
1, & j \geq d
\end{cases}
$$

and its inverse $D_\lambda^{-1}$ with entries $d_\lambda^{-1}[j] = 1 / d_\lambda[j]$.

Define the operator $M \in \mathbb{R}^{n \times n}$:

$$
M = D_\lambda Q^T
$$

which is the key matrix passed to the randomized SVD step.

### 3.3 Ridge Inverse

Precompute the ridge-regularized inverse:

$$
(X^T X + \lambda I_d)^{-1} \in \mathbb{R}^{d \times d}
$$

This is used for effect size estimation in Step 3 (Equation 16).

### 3.4 Fallback for Singular Matrices

If any matrix inversion fails (near-singularity from collinear covariates or over-specified $K$), Tikhonov regularization is added: $\varepsilon I$ where $\varepsilon = 10^{-8} \cdot \max(|\text{diag}(A)|)$.

---

## 4. NMF Imputation of Missing Genotypes

When `--nmf-impute` is active, before latent factor estimation, a Non-negative Matrix Factorisation is performed on the estimation SNP subset to learn a low-rank representation for imputing missing genotypes.

### 4.1 Model

Let $Y^{\text{est}} \in \mathbb{R}^{n \times p_{\text{est}}}$ denote the genotype matrix for the estimation subset. The NMF approximates:

$$
Y^{\text{est}} \approx W H
$$

where $W \in \mathbb{R}^{n \times K}$ (sample loadings, non-negative) and $H \in \mathbb{R}^{K \times p_{\text{est}}}$ (SNP loadings, non-negative). The genotype values are centered during decoding, so the NMF operates on centered data (with missing entries filled by the reconstruction before centering).

### 4.2 Initialization (Random Probe)

Initial values $W^{(0)}$ and $H^{(0)}$ are obtained via a randomized probing procedure (cf. Nyström-like):

1. Generate a random Gaussian sketch matrix $\Omega \in \mathbb{R}^{n \times (K+10)}$ with entries $\omega_{ij} \sim \mathcal{N}(0, 1)$, seeded by $\text{seed} + 10^6$.

2. Compute the sketch $Z = Y_{\text{est}}^T \Omega \in \mathbb{R}^{p_{\text{est}} \times (K+10)}$ via a streaming pass (mean-imputed genotypes).

3. Compute the thin SVD of $Z$ (only $V_t$ needed): $Z = U_Z \Sigma_Z V_Z^T$.

4. Take $W^{(0)}_{\text{probe}} = \Omega V_Z[:, 1:K]^T \in \mathbb{R}^{n \times K}$, then set $W^{(0)} = |W^{(0)}_{\text{probe}}|$ (entrywise absolute value, enforcing non-negativity).

5. Initialize $H^{(0)}$ via one backward pass over the estimation subset:

   $$H^{(0)}_{\text{chunk}} = \max(0, (W^{(0)T} W^{(0)})^{-1} W^{(0)T} Y_{\text{chunk}})$$

   using mean-imputed genotypes.

Both $W^{(0)}$ and $H^{(0)}$ are clamped to $[\varepsilon, \infty)$ with $\varepsilon = 10^{-16}$, then $W$ is L1-normalized column-wise with $H$ rescaled accordingly to preserve the product $WH$.

### 4.3 Multiplicative Updates

For each iteration $t = 0, \ldots, n_{\text{iter}}-1$, the NMF alternates between a forward pass (updating $W$) and a backward pass (updating $H$). Missing genotypes are filled with the current reconstruction $W^{(t)} H^{(t)}_{\text{chunk}}$ before each pass (EM-like approach).

#### Forward Pass (Update W)

For each chunk $c$, decode genotypes with NMF fill, yielding $Y_c \in \mathbb{R}^{n \times p_c}$. Compute:

$$
N_W \leftarrow N_W + Y_c H_c^T, \qquad HH \leftarrow HH + H_c H_c^T
$$

After streaming all chunks, update $W$ element-wise:

$$
W_{ik}^{(t+1)} \leftarrow W_{ik}^{(t)} \cdot \frac{(N_W)_{ik}}{(W^{(t)} HH + \varepsilon)_{ik}}
$$

Clamp all entries to $[\varepsilon, \infty)$. Then L1-normalize columns of $W$ and rescale rows of $H$:

For each $k$: $\alpha_k = \sum_i W_{ik}$, then $W_{:,k} \leftarrow W_{:,k} / \alpha_k$, $H_{k,:} \leftarrow H_{k,:} \cdot \alpha_k$.

#### Backward Pass (Update H)

First compute $W^T W \in \mathbb{R}^{K \times K}$. For each chunk, recompute $Y_c$ (with updated $W$ used for NMF fill) and update column $c$ of $H$ element-wise:

$$
(H_c)_{k} \leftarrow (H_c)_{k} \cdot \frac{(W^T Y_c)_{k}}{(W^T W H_c + \varepsilon)_{k}}
$$

Clamp all entries to $[\varepsilon, \infty)$, then renormalize $W$ as in the forward pass.

### 4.4 Cross-Validation (NMF Phase)

After each NMF iteration $t$, a cross-validation error is computed to track convergence. A random mask is generated (seed = $\text{seed} + t$) holding out a fraction $\text{cv\_rate}$ (default $5 \times 10^{-4}$) of *observed* (non-missing) genotype positions.

For each chunk:

- Decode centered genotypes **without** NMF fill (mean-impute for missing), yielding $Y^{\text{centered}} \in \mathbb{R}^{n \times p_c}$.
- Compute the NMF reconstruction: $\hat{Y} = W^{(t)} H_c$ where $H_c$ is a slice of the full $H^{(t)}$.
- For each masked position $(i,j)$ where $Y^{\text{centered}}_{ij}$ is observed (non-NaN):

  $$
  e^{\text{nmf}}_{ij} = |Y^{\text{centered}}_{ij} - \hat{Y}_{ij}|
  $$

  $$
  e^{\text{mean}}_{ij} = |Y^{\text{centered}}_{ij} - 0| = |Y^{\text{centered}}_{ij}|
  $$

  (after centering, mean imputation yields 0).

- Report $\text{MAE}_{\text{nmf}} = \sum e^{\text{nmf}}_{ij} / N_{\text{masked}}$ and $\text{MAE}_{\text{mean}}$ aggregated across chunks.

**Important caveat:** This CV uses the global $H^{(t)}$ which was jointly optimized with $W^{(t)}$ on these same SNPs. This measures the NMF convergence but does *not* evaluate the actual imputation pipeline used during GWAS testing (which uses the per-chunk NLS estimate, see §9).

---

## 5. Steps 1-2: Randomized SVD for Latent Factor Estimation

The LFMM2 model assumes the genotype matrix decomposes as:

$$
Y = \mathbf{1} \mu^T + X B + U V^T + E
$$

where $U \in \mathbb{R}^{n \times K}$ are unobserved latent factors, $V \in \mathbb{R}^{p \times K}$ their loadings, and $E$ is noise. To estimate $U$ without iteratively fitting $V$, the LFMM2 approach notes that under the ridge-regularized least-squares framework, the latent factors can be recovered from the top $K$ singular vectors of $M Y_{\text{est}}$ where $M$ is from §3.2.

### 5.1 The M Operator

The matrix $M = D_\lambda Q^T$ acts as a generalized projection onto the subspace where the covariates $X$ contribute most. Recovering latent factors from $M Y_{\text{est}}$ rather than raw $Y_{\text{est}}$ removes X-confounded signal from the first singular vectors.

### 5.2 Randomized SVD Algorithm

Given $M \in \mathbb{R}^{n \times n}$ and $Y_{\text{est}} \in \mathbb{R}^{n \times p_{\text{est}}}$, we seek the top $K$ left singular vectors of $A = M Y_{\text{est}}$.

**Sketch dimension:** $l = K + l_{\text{oversample}}$ (default $l_{\text{oversample}} = 10$).

**Step 1a: Initial Sketch**

- Generate random Gaussian matrix $\Omega \in \mathbb{R}^{n \times l}$ with $\omega_{ij} \sim \mathcal{N}(0, 1)$.
- Precompute $M^T \Omega \in \mathbb{R}^{n \times l}$ via BLAS.
- Streaming sketch: $Z = Y_{\text{est}}^T (M^T \Omega)$. For each chunk $Y_c$:

  $$Z_c = Y_c^T (M^T \Omega)$$

  accumulate into $Z \in \mathbb{R}^{p_{\text{est}} \times l}$.
- Compute the thin QR decomposition $Z = Q_z R_z$, obtaining an orthonormal basis $Q_z$ for the approximate column space of $A^T$.

**Step 1b: Power Iterations** (repeated $n_{\text{power}}$ times, default 2)

For each iteration:

1. **Forward pass** ($A Q_z$): For each chunk $Y_c$ with corresponding slice $Q_{z,c}$:

   $$A Q_{z,c} = M \cdot (Y_c Q_{z,c}) \in \mathbb{R}^{n \times l}$$

   Accumulated across workers using per-worker buffers, summed after.

2. QR decompose $A Q_z$ to get $Q_{aqz}$.
3. Compute $M^T Q_{aqz}$ via BLAS.
4. **Backward pass** ($A^T Q_{aqz}$): For each chunk:

   $$Z_c = Y_c^T (M^T Q_{aqz}) \in \mathbb{R}^{p_c \times l}$$

5. QR decompose the accumulated $Z$ to update $Q_z$.

This alternation sharpens the approximation of the left singular subspace, improving by a factor of $(\sigma_{K+1}/\sigma_K)^{2 n_{\text{power}}}$ per iteration.

**Step 2: Project and Recover**

- Compute $B_{\text{svd}} = A Q_z = M Y_{\text{est}} Q_z \in \mathbb{R}^{n \times l}$ via one final streaming pass.
- Thin SVD of $B_{\text{svd}}$ (only $U$ needed):

  $$B_{\text{svd}} = U_{\text{small}} \Sigma_{\text{small}} V_{\text{small}}^T$$

  with $U_{\text{small}} \in \mathbb{R}^{n \times l}$.

- Recover latent factors via:

  $$U_{\text{hat}} = Q \cdot \left(D_\lambda^{-1} \odot U_{\text{small}}[:, 1:K]\right)$$

  where $\odot$ denotes row-wise scaling: $(D_\lambda^{-1} U_{[1:K]})_{ik} = d_\lambda^{-1}[i] \cdot U_{\text{small}}[i, k]$.

The output is $U_{\text{hat}} \in \mathbb{R}^{n \times K}$, the estimated latent factor matrix.

---

## 6. Step 3: Covariate Effect Size Estimation

Given $U_{\text{hat}}$ from §5, the goal is to estimate the covariate effects $B \in \mathbb{R}^{d \times p}$ while correcting for latent structure. The approach projects out the latent subspace before estimating $B$.

### 6.1 Orthogonal Projector Onto Latent Subspace

Define the orthogonal projector onto the column space of $U_{\text{hat}}$:

$$
P_U = U_{\text{hat}} (U_{\text{hat}}^T U_{\text{hat}})^{-1} U_{\text{hat}}^T \in \mathbb{R}^{n \times n}
$$

Then $I - P_U$ projects onto the orthogonal complement, removing latent-factor signal from the genotypes.

### 6.2 Ridge-Regularized Least Squares

The covariate effects $\hat{B}$ are estimated column-by-column via ridge regression on the residualised genotypes:

For each SNP column $y_j \in \mathbb{R}^n$:

$$
r_j = (I - P_U) y_j
$$

$$
\hat{\beta}_j = (X^T X + \lambda I_d)^{-1} X^T r_j
$$

In matrix form, for all SNPs simultaneously:

$$
\hat{B}^T = (X^T X + \lambda I_d)^{-1} X^T (I - P_U) Y_{\text{full}}
$$

This is computed chunk-by-chunk in the fused pass. Define $X_t R = (X^T X + \lambda I_d)^{-1} X^T \in \mathbb{R}^{d \times n}$, precomputed in §3.3. For each chunk $Y_c$:

$$
\hat{B}_c^T = X_t R \cdot (I - P_U) \cdot Y_c
$$

The betas are written directly to disk as chunk fragments; no $p$-dimensional array is held in RAM.

---

## 7. Step 4: Per-SNP Association Testing

For each SNP $j$, we test the null hypothesis that the covariate effects equal zero ($\beta_j = 0$), while adjusting for both observed covariates and estimated latent factors.

### 7.1 Full Model Specification

For SNP $j$, the full linear model is:

$$
y_j = \mathbf{1} \alpha_j + X \beta_j + U_{\text{hat}} \gamma_j + \varepsilon_j
$$

where $\alpha_j$ is the intercept, $\beta_j \in \mathbb{R}^d$ are covariate effects of interest, $\gamma_j \in \mathbb{R}^K$ are latent factor effects, and $\varepsilon_j \sim \mathcal{N}(0, \sigma_j^2 I_n)$.

Define the design matrix $C = [\mathbf{1} \;|\; X \;|\; U_{\text{hat}}] \in \mathbb{R}^{n \times (1+d+K)}$.

### 7.2 Ordinary Least Squares

For each column $y_j$, the OLS coefficient vector is:

$$
\hat{\theta}_j = (C^T C)^{-1} C^T y_j = H y_j
$$

where $H = (C^T C)^{-1} C^T \in \mathbb{R}^{(1+d+K) \times n}$ is precomputed once. For each chunk $Y_c$, this is applied as a single matrix multiplication:

$$
\hat{\Theta}_c = H Y_c
$$

yielding coefficients $\hat{\Theta}_c \in \mathbb{R}^{(1+d+K) \times p_c}$.

### 7.3 Residual Variance

The residual sum of squares for SNP $j$:

$$
\text{RSS}_j = \|y_j - C \hat{\theta}_j\|^2 = \varepsilon_j^T \varepsilon_j
$$

The residual variance estimate:

$$
\hat{\sigma}_j^2 = \frac{\text{RSS}_j}{n - 1 - d - K}
$$

with degrees of freedom $\text{df} = n - 1 - d - K$.

### 7.4 t-Tests for Covariate Effects

The covariate coefficients occupy indices $1, \ldots, d$ in $\hat{\theta}_j$ (index 0 is the intercept). For covariate $m$:

$$
\text{SE}(\hat{\beta}_{j,m}) = \sqrt{\hat{\sigma}_j^2 \cdot [(C^T C)^{-1}]_{m+1, m+1}}
$$

$$
t_{j,m} = \frac{\hat{\beta}_{j,m}}{\text{SE}(\hat{\beta}_{j,m})}
$$

The two-sided p-value is computed from Student's $t$ distribution with df degrees of freedom:

$$
p_{j,m} = 2 \cdot F_{t_{\text{df}}}(-|t_{j,m}|)
$$

where $F_{t_{\text{df}}}$ is the CDF. Zero-variance (monomorphic) SNPs yield $t = 0, p = 1$.

### 7.5 Variance Decomposition

The total sum of squares for SNP $j$ is partitioned into three components:

$$
\text{TSS}_j = \|y_j\|^2 = \text{SS}_{\text{cov}}(j) + \text{SS}_{\text{latent}}(j) + \text{RSS}_j
$$

where

$$\text{SS}_{\text{cov}}(j) = \|X \hat{\beta}_j\|^2, \quad \text{SS}_{\text{latent}}(j) = \|C \hat{\theta}_j - \mathbf{1}\hat{\alpha}_j - X\hat{\beta}_j\|^2$$

and the $R^2$ fractions are $r^2_{\text{cov}} = \text{SS}_{\text{cov}} / \text{TSS}$, etc. These are reported per-SNP in the output.

---

## 8. NMF On-The-Fly Imputation (GWAS Phase)

During the GWAS/testing phase, when NMF imputation is used, the genotype matrix $Y_{\text{full}} \in \mathbb{R}^{n \times p}$ (all SNPs, not just the estimation subset) must be imputed. Since $H$ was only estimated for the estimation subset ($p_{\text{est}}$ SNPs), the SNP loadings $H$ for the full set ($p$ SNPs) are not available. Instead, $H$ is estimated per chunk on-the-fly.

### 8.1 Per-Chunk NLS Estimation

For each chunk $c$ of $Y_{\text{full}}$, with $W \in \mathbb{R}^{n \times K}$ fixed from the NMF training phase:

- **Pass 1:** Raw-decode genotypes to $\{0, 1, 2, \text{NaN}\}$. Fill missing (NaN) entries with the per-column mean computed from observed entries, yielding $Y^{\text{filled}}_c \in \mathbb{R}^{n \times p_c}$.

- **Estimate $H_c$** via non-negative constrained least squares:

  $$H_c^{\text{raw}} = W^{\dagger} Y^{\text{filled}}_c = (W^T W)^{-1} W^T Y^{\text{filled}}_c$$

  $$H_c = \max(0, H_c^{\text{raw}}) \quad \text{(element-wise clamping)}$$

  where $W^{\dagger} = (W^T W)^{-1} W^T \in \mathbb{R}^{K \times n}$ is the Moore-Penrose pseudoinverse (since $W$ has full column rank), precomputed after NMF training.

- **Compute fill values:** $\hat{Y}_c = W H_c \in \mathbb{R}^{n \times p_c}$.

- **Pass 2:** Re-decode from raw bytes. For each SNP column, NaN entries are filled with the corresponding $\hat{Y}_{ij}$ values **before** centering and normalization. Then standard centering (± Eigenstrat scaling) is applied.

This means genotypes imputed via NMF participate in the centering mean computation (the column mean is computed over all entries including NMF-filled ones). The fill values are on the raw genotype scale $\{0, 1, 2\}$.

---

## 9. GWAS-Phase Cross-Validation

Since the NMF-phase CV (§4.4) evaluates $W H^{(t)}$ using the jointly-optimized $H^{(t)}$, it does not reflect the actual on-the-fly imputation pipeline used during GWAS. Therefore, a separate per-block cross-validation is performed during the GWAS pass.

### 9.1 Per-Block NCV Procedure

For each chunk $c$ during the GWAS streaming pass:

1. **Raw decode** genotypes from `block.raw` into $Y^{\text{raw}}_c \in \mathbb{R}^{n \times p_c}$ with values in $\{0, 1, 2, \text{NaN}\}$.

2. **Generate mask:** For each entry $(i, j)$ where $Y^{\text{raw}}_{ij}$ is observed (non-NaN), with probability $\text{cv\_rate}$ (default $5 \times 10^{-4}$), the entry is selected for cross-validation. Seed: $\text{seed} + \text{chunk\_seq}$.

3. **Create leave-out matrix** $Y^{\text{masked}}_c$ as a copy of $Y^{\text{raw}}_c$ with masked positions set to NaN.

4. **Mean-impute** $Y^{\text{masked}}_c$: for each column, replace all NaN entries (truly missing + CV-masked) with the column mean computed from the remaining observed entries (LOO basis — the masked positions are NaN and thus excluded from the mean).

5. **Estimate $H$** using the same on-the-fly method as §8.1 but on the masked matrix:

   $$H^{\text{cv}}_c = \max(0, W^{\dagger} Y^{\text{masked}}_c)$$

6. **Predict:** $\hat{Y}^{\text{cv}}_c = W H^{\text{cv}}_c$.

7. **Compute errors on masked positions** (on the raw genotype scale $\{0, 1, 2\}$):

   $$
   e^{\text{nmf}}_{\text{gwas}} = \sum_{(i,j) \in \text{mask}} |Y^{\text{raw}}_{ij} - \hat{Y}^{\text{cv}}_{ij}|
   $$

   $$
   e^{\text{mean}}_{\text{gwas}} = \sum_{(i,j) \in \text{mask}} |Y^{\text{raw}}_{ij} - \bar{Y}^{\text{col}(j)}|
   $$

   where $\bar{Y}^{\text{col}(j)}$ is the LOO column mean (mean-imputation baseline prediction).

8. Accumulate across all chunks. After the GWAS pass completes:

   $$ \text{MAE}^{\text{nmf}}_{\text{gwas}} = \frac{\sum e^{\text{nmf}}_{\text{gwas}}}{N_{\text{masked}}}, \quad \text{MAE}^{\text{mean}}_{\text{gwas}} = \frac{\sum e^{\text{mean}}_{\text{gwas}}}{N_{\text{masked}}} $$

These values are reported in the summary output alongside the NMF-phase CV metrics.

---

## 10. Genomic Inflation Factor (GIF) Calibration

Population stratification and cryptic relatedness can inflate test statistics. The genomic inflation factor (GIF) is estimated from the empirical distribution of $t^2$ statistics across all SNPs and covariates.

### 10.1 Streaming Histogram

A streaming histogram over $t^2 \in [0, 100]$ with bin width $0.001$ (100,000 bins) is maintained per covariate trait across all chunks. Let $h_m[k]$ be the count of $t^2$ values falling in bin $k$ for trait $m$. After processing all $p$ SNPs:

- Total tested SNPs for trait $m$: $N_m = \sum_k h_m[k]$.
- Median $t^2$: the value at position $N_m/2$ (or average of positions $N_m/2 - 1$ and $N_m/2$ for even $N_m$), found by walking cumulative bin counts.

### 10.2 GIF Computation

Under the null hypothesis and proper calibration, each $t^2$ follows approximately a scaled $\chi^2_1$ distribution. The median of a $\chi^2_1$ distribution is approximately $0.4549$. The GIF for trait $m$ is:

$$
\text{GIF}_m = \frac{\text{median}(t^2_m)}{0.4549}
$$

with $\text{GIF}_m$ clamped to a minimum of 1.0. The average GIF across all $d$ traits is $\overline{\text{GIF}} = \frac{1}{d} \sum_{m=1}^d \text{GIF}_m$.

### 10.3 GIF Calibration of p-values

In the final output, each SNP's t-statistic is calibrated and converted to a two-sided p-value using the standard normal distribution:

$$
z_{j,m} = \frac{t_{j,m}}{\sqrt{\text{GIF}_m}}
$$

$$
p^{\text{cal}}_{j,m} = 2 \cdot \Phi(-|z_{j,m}|)
$$

where $\Phi$ is the standard normal CDF. This is equivalent to dividing the test statistic by the GIF estimate to correct for inflation.

---

## 11. Complete Pipeline Summary

The full LFMM2 pipeline proceeds as follows:

1. **Load data:** Read BED, FAM, BIM, and covariate files. Align and subset samples.

2. **(Optional) NMF imputation on estimation subset:**
   - Initialize $W$ and $H$ via random probe (§4.2).
   - Stream multiplicative updates for $n_{\text{iter}}$ iterations (§4.3).
   - After each iteration, compute $\text{MAE}_{\text{nmf}}$ and $\text{MAE}_{\text{mean}}$ on masked genotypes (§4.4).
   - Precompute $W^{\dagger} = (W^T W)^{-1} W^T$ for later on-the-fly imputation.

3. **Step 0:** Precompute SVD of $X$, build $M = D_\lambda Q^T$, $\text{ridge\_inv}$, and $D_\lambda^{-1}$ (§3).

4. **Steps 1-2 (Latent factor estimation via RSVD):**
   - Generate random sketch matrix $\Omega$.
   - Compute initial sketch $Z = Y_{\text{est}}^T (M^T \Omega)$.
   - QR decompose to get $Q_z$.
   - Run $n_{\text{power}}$ power iterations (forward $M Y_{\text{est}} Q_z$, QR, backward $Y_{\text{est}}^T (M^T Q_{\text{aqz}})$, QR).
   - Final projection $B = M Y_{\text{est}} Q_z$, SVD, recover $U_{\text{hat}}$ (§5).

5. **Fused Steps 3-4 (GWAS/testing):**
   - Precompute OLS hat matrices: $I - P_U$, $X_t R$, $H = (C^T C)^{-1} C^T$.
   - For each chunk $c$:
     - **(If NMF)** Impute via $H_c = \max(0, W^{\dagger} Y_c^{\text{filled}})$ then $W H_c$ fills NaNs before centering (§8).
     - Compute $\hat{B}_c$ (effect sizes) via ridge regression on residualised genotypes (§6).
     - Compute $\hat{\Theta}_c$ (full OLS coefficients) and $t_{j,m}$ statistics (§7).
     - Compute $r^2$ variance decomposition.
     - **(If NMF CV)** Compute GWAS-level cross-validation errors using the same on-the-fly estimator on held-out positions (§9).
     - Feed $t^2$ values into streaming histogram for GIF estimation (§10).
     - Write per-chunk results to temporary binary files.

6. **Post-processing:**
   - Compute GIF from histogram medians (§10.2).
   - Read back chunk files, calibrate p-values with GIF (§10.3), and write final TSV output.
   - Write summary file with CV metrics, GIF, and diagnostics.

### Computational Complexity

| Step | Passes over Y | Memory (RAM) |
|------|--------------|--------------|
| SVD of X | 0 (in-core) | $O(nd + d^2)$ |
| NMF init | 2 passes | $O(np_{\text{est}})$ (full H) |
| NMF updates | $2 n_{\text{iter}}$ passes | $O(np_{\text{est}})$ (full H) |
| RSVD | $3 + 2 n_{\text{power}}$ passes | $O(nK + p_{\text{est}} K)$ |
| GWAS fused | 1 pass | $O(n(1+d+K) + p_c d)$ per chunk |
| GWAS CV | 0 extra passes | $O(nK + p_c K)$ in process_fn |

All passes over genotype data are fully streaming: at any point, only one chunk of $\text{chunk\_size}$ SNPs (default 10,000) is held in RAM.

### Imputation Strategies Summary

| Phase | Strategy | Method |
|-------|----------|--------|
| NMF training (estimation subset) | `NmfInRam` | $W H_c$ with precomputed $H$ |
| RSVD (estimation subset) | `NmfInRam` or `Mean` | Same; NMF fill if enabled |
| GWAS testing (all SNPs) | `NmfOnTheFly` or `Mean` | $H_c = \max(0, W^{\dagger} Y_c)$, then $W H_c$ fill |
| NMF CV (estimation subset) | $W H_c$ direct | Compares centered $W H$ vs centered truth |
| GWAS CV (all SNPs) | `NmfOnTheFly` with LOO mask | Compares $W H_c^{\text{cv}}$ vs raw truth on held-out positions |
