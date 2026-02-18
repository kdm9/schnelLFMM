use anyhow::{Context, Result};
use ndarray::Array2;
use ndarray_linalg::InverseInto;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::bed::{BedFile, BimRecord, SubsetSpec};
use crate::parallel::{parallel_stream, DisjointRowWriter};
use crate::precompute::Precomputed;
use crate::progress::make_progress_bar;
use crate::Lfmm2Config;

/// Configuration for streaming results output.
///
/// When provided to `test_associations_fused`, each chunk writes a TSV fragment
/// during the streaming pass. After GIF calibration, fragments are coalesced
/// into a single output file with calibrated p-values.
pub struct OutputConfig<'a> {
    pub path: &'a Path,
    pub bim: &'a [BimRecord],
    pub cov_names: &'a [String],
}

/// Results from the LFMM2 association testing pass.
pub struct TestResults {
    /// Estimated latent factors: n × K
    pub u_hat: Array2<f64>,
    /// Effect sizes: p × d (ridge regression B from Step 3)
    pub effect_sizes: Array2<f64>,
    /// t-statistics: p × d (from per-locus regression, Step 4)
    pub t_stats: Array2<f64>,
    /// Calibrated p-values: p × d (after GIF correction)
    pub p_values: Array2<f64>,
    /// Genomic inflation factor
    pub gif: f64,
}

/// Perform Steps 3-4 fused in a single pass over Y_full.
///
/// Step 3: B_hat^T = (X^T X + λI)^{-1} X^T (Y - P_U Y)
///   where P_U = U_hat (U_hat^T U_hat)^{-1} U_hat^T is the orthogonal projector onto col(U_hat).
///
/// Step 4: Per-locus OLS with C = [X | U_hat], t-tests, GIF calibration.
///   For each SNP j: y_j = C γ_j + ε_j, then t_j = γ̂[1..d] / se(γ̂[1..d]).
///   Standard errors come from se²(γ̂_j) = σ̂² · diag((C^T C)^{-1}), where σ̂² = RSS / df.
///   Degrees of freedom: df = n - d - K (residual df after fitting d covariates + K latent factors).
///
/// When `output` is `Some`, each chunk's effect sizes and t-statistics are written
/// to a temporary TSV fragment. After GIF calibration, fragments are coalesced into
/// the final output file with calibrated p-values inserted.
pub fn test_associations_fused(
    y_full: &BedFile,
    x: &Array2<f64>,
    u_hat: &Array2<f64>,
    pre: &Precomputed,
    config: &Lfmm2Config,
    output: Option<&OutputConfig>,
) -> Result<TestResults> {
    let n = y_full.n_samples;
    let p = y_full.n_snps;
    let d = x.ncols();
    let k = config.k;
    let chunk_size = config.chunk_size;

    // Validate degrees of freedom: df = n - d - K must be positive for a valid t-test.
    // With df ≤ 0 the Student-t distribution is undefined, and the usize subtraction
    // (n - d - k) would silently wrap around to a huge value.
    if n <= d + k {
        anyhow::bail!(
            "Insufficient degrees of freedom: n={} samples but d+K={}+{}={}. \
             Need n > d + K for valid t-tests. Reduce K or add more samples.",
            n, d, k, d + k,
        );
    }
    let df = (n - d - k) as f64;

    // Precompute P_U = U_hat (U^T U)^{-1} U^T (n × n).
    let utu = u_hat.t().dot(u_hat);
    let utu_inv = safe_inv(&utu, "U_hat^T U_hat")?;
    let p_u = u_hat.dot(&utu_inv).dot(&u_hat.t());

    // XtR = (X^T X + λI)^{-1} X^T (d × n) — precomputed ridge projection
    let xtr = pre.ridge_inv.dot(&x.t());

    // I - P_U for Step 3 residual: projects Y onto the space orthogonal to U_hat
    let mut i_minus_pu = Array2::<f64>::eye(n);
    i_minus_pu -= &p_u;

    // Step 4 precomputes:
    // C = [X | U_hat] (n × (d+K)) — combined covariate + latent factor design matrix.
    let mut c = Array2::<f64>::zeros((n, d + k));
    c.slice_mut(ndarray::s![.., ..d]).assign(x);
    c.slice_mut(ndarray::s![.., d..]).assign(u_hat);

    // (C^T C)^{-1}: needed for standard errors.
    let ctc = c.t().dot(&c);
    let ctc_inv = safe_inv(&ctc, "C^T C  where C = [X | U_hat]")?;

    // H = (C^T C)^{-1} C^T — the OLS hat matrix for coefficient estimation
    let h = ctc_inv.dot(&c.t());

    // Diagonal of (C^T C)^{-1} for standard error computation
    let ctc_inv_diag: Vec<f64> = (0..d).map(|j| ctc_inv[(j, j)]).collect();

    // Allocate output arrays
    let mut effect_sizes = Array2::<f64>::zeros((p, d));
    let mut t_stats = Array2::<f64>::zeros((p, d));

    // Create temp dir for chunk files if writing output
    let chunk_dir = if output.is_some() {
        let parent = output.unwrap().path.parent().unwrap_or(Path::new("."));
        Some(
            tempfile::Builder::new()
                .prefix(".lfmm2_chunks_")
                .tempdir_in(parent)
                .context("Failed to create temp directory for chunk files")?,
        )
    } else {
        None
    };

    // Single fused pass over Y_full
    let subset = SubsetSpec::All;
    let n_chunks = p.div_ceil(chunk_size);
    let pb = make_progress_bar(n_chunks as u64, "Association tests", config.progress);

    {
        let wr_effects = DisjointRowWriter::new(&mut effect_sizes);
        let wr_tstats = DisjointRowWriter::new(&mut t_stats);
        parallel_stream(y_full, &subset, chunk_size, config.n_workers, |_worker_id, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let chunk_cols = block.n_cols;
            let start = block.seq * chunk_size;

            // Step 3: B = (XtR @ (I - P_U) @ chunk)^T
            let residual = i_minus_pu.dot(&chunk);
            let b_chunk = xtr.dot(&residual); // d × chunk_cols
            let b_chunk_t = b_chunk.t().to_owned();

            // Step 4: OLS with C = [X | U_hat]
            let coefs = h.dot(&chunk); // (d+K) × chunk_cols
            let fitted = c.dot(&coefs); // n × chunk_cols
            let residuals = &chunk - &fitted; // n × chunk_cols

            let mut local_tstats = Array2::<f64>::zeros((chunk_cols, d));

            for col_in_chunk in 0..chunk_cols {
                let res_col = residuals.column(col_in_chunk);
                let rss: f64 = res_col.dot(&res_col);
                let sigma2 = rss / df;

                for j in 0..d {
                    let (t, _) = t_test(coefs[(j, col_in_chunk)], sigma2, ctc_inv_diag[j], df);
                    local_tstats[(col_in_chunk, j)] = t;
                }
            }

            unsafe {
                wr_effects.write_rows(start, &b_chunk_t);
                wr_tstats.write_rows(start, &local_tstats);
            }

            // Write chunk TSV fragment if output configured
            if let Some(ref dir) = chunk_dir {
                let bim_slice = &output.unwrap().bim[start..start + chunk_cols];
                write_chunk_tsv(dir.path(), block.seq, bim_slice, &b_chunk_t, &local_tstats, d)
                    .expect("failed to write chunk file");
            }

            pb.inc(1);
        });
    }
    pb.finish_and_clear();

    // GIF calibration (genomic inflation factor).
    //
    // Under the null, t² ~ χ²(1). The GIF is:
    //   GIF = median(t²) / median(χ²(1)) = median(t²) / 0.4549
    // where 0.4549 ≈ qchisq(0.5, df=1). A well-calibrated test gives GIF ≈ 1.
    //
    // Calibrated z-scores: z_cal = t / sqrt(GIF), p_cal = 2Φ(-|z_cal|).
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut p_values = Array2::<f64>::zeros((p, d));
    let mut gif_per_trait = Vec::with_capacity(d);
    let mut total_gif = 0.0;

    for j in 0..d {
        let t_col = t_stats.column(j);
        let mut z_sq: Vec<f64> = t_col.iter().map(|&t| t * t).filter(|v| v.is_finite()).collect();
        z_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_z_sq = median_sorted(&z_sq);

        let gif = if median_z_sq < 1e-10 {
            1.0
        } else {
            median_z_sq / 0.4549
        };
        gif_per_trait.push(gif);
        total_gif += gif;

        let gif_sqrt = gif.sqrt();
        for i in 0..p {
            let z = t_stats[(i, j)];
            let z_cal = z / gif_sqrt;
            p_values[(i, j)] = 2.0 * normal.cdf(-z_cal.abs());
        }
    }

    let avg_gif = total_gif / d as f64;

    // Coalesce chunk files into final output
    if let Some(out) = output {
        coalesce_output(
            out.path,
            out.cov_names,
            chunk_dir.as_ref().unwrap().path(),
            n_chunks,
            &gif_per_trait,
        )?;
    }

    Ok(TestResults {
        u_hat: u_hat.to_owned(),
        effect_sizes,
        t_stats,
        p_values,
        gif: avg_gif,
    })
}

/// Write a chunk's effect sizes and t-statistics as a TSV fragment (no header).
///
/// Format: `chr\tpos\tsnp_id\tbeta_0\tt_0\tbeta_1\tt_1\t...`
fn write_chunk_tsv(
    dir: &Path,
    seq: usize,
    bim: &[BimRecord],
    betas: &Array2<f64>,
    tstats: &Array2<f64>,
    d: usize,
) -> Result<()> {
    let path = dir.join(format!("chunk_{:06}.tsv", seq));
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    let n_rows = betas.nrows();
    for i in 0..n_rows {
        write!(f, "{}\t{}\t{}", bim[i].chrom, bim[i].pos, bim[i].snp_id)?;
        for j in 0..d {
            write!(f, "\t{:.6e}\t{:.6e}", betas[(i, j)], tstats[(i, j)])?;
        }
        writeln!(f)?;
    }
    Ok(())
}

/// Coalesce chunk TSV fragments into a single output file with calibrated p-values.
///
/// Reads each chunk file in sequence order, parses t-statistics, applies GIF
/// calibration, and writes the final output with columns:
/// `chr\tpos\tsnp_id\tp_cov1\tbeta_cov1\tt_cov1\t...`
fn coalesce_output(
    output_path: &Path,
    cov_names: &[String],
    chunk_dir: &Path,
    n_chunks: usize,
    gif_per_trait: &[f64],
) -> Result<()> {
    let d = cov_names.len();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let gif_sqrt: Vec<f64> = gif_per_trait.iter().map(|g| g.sqrt()).collect();

    let mut out = BufWriter::new(
        std::fs::File::create(output_path)
            .with_context(|| format!("Failed to create {}", output_path.display()))?,
    );

    // Header: chr, pos, snp_id, then per-covariate p/beta/t triples
    write!(out, "chr\tpos\tsnp_id")?;
    for name in cov_names {
        write!(out, "\tp_{}\tbeta_{}\tt_{}", name, name, name)?;
    }
    writeln!(out)?;

    // Concatenate chunk files in sequence order, inserting calibrated p-values
    for seq in 0..n_chunks {
        let chunk_path = chunk_dir.join(format!("chunk_{:06}.tsv", seq));
        let f = std::fs::File::open(&chunk_path)
            .with_context(|| format!("Failed to open chunk file: {}", chunk_path.display()))?;
        let reader = BufReader::new(f);
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            // Chunk format: chr, pos, snp_id, beta_0, t_0, beta_1, t_1, ...
            write!(out, "{}\t{}\t{}", fields[0], fields[1], fields[2])?;
            for j in 0..d {
                let beta_str = fields[3 + j * 2];
                let t_str = fields[4 + j * 2];
                let t: f64 = t_str.parse().unwrap_or(0.0);
                let z_cal = t / gif_sqrt[j];
                let p_cal = 2.0 * normal.cdf(-z_cal.abs());
                write!(out, "\t{:.6e}\t{}\t{}", p_cal, beta_str, t_str)?;
            }
            writeln!(out)?;
        }
    }

    Ok(())
}

/// Invert a square matrix, falling back to diagonal regularization if singular.
///
/// Real-data edge cases (collinear covariates, over-specified K) can make
/// U^T U or C^T C singular. Rather than crashing, we add ε·I where
/// ε = 1e-8 · max(diag(A)). This is small enough to not affect well-conditioned
/// results (relative perturbation ~1e-8) but prevents hard failures.
fn safe_inv(a: &Array2<f64>, name: &str) -> Result<Array2<f64>> {
    match a.clone().inv_into() {
        Ok(inv) => Ok(inv),
        Err(_) => {
            let n = a.nrows();
            let diag_max = (0..n)
                .map(|i| a[(i, i)].abs())
                .fold(0.0f64, f64::max)
                .max(1e-300);
            let eps = 1e-8 * diag_max;
            eprintln!(
                "Warning: {} is singular, adding ε={:.2e} diagonal regularization",
                name, eps,
            );
            let mut a_reg = a.clone();
            for i in 0..n {
                a_reg[(i, i)] += eps;
            }
            a_reg.inv_into().map_err(|e| {
                anyhow::anyhow!(
                    "{} inversion failed even with ε={:.2e} regularization: {}",
                    name,
                    eps,
                    e,
                )
            })
        }
    }
}

/// Compute t-statistic and two-sided p-value, guarding against zero-variance SNPs.
///
/// For a monomorphic SNP (all samples have the same genotype), the centered column
/// is all zeros. After OLS: residuals ≡ 0, RSS = 0, σ̂² = 0, se = 0.
/// Then t = β̂/se = finite/0 = ±Inf, which crashes statrs::StudentsT::cdf
/// (it passes t through the beta function which requires finite input).
///
/// We return (t=0, p=1): zero variance → no evidence of association.
#[inline]
fn t_test(coef: f64, sigma2: f64, ctc_inv_jj: f64, df: f64) -> (f64, f64) {
    let se = (sigma2 * ctc_inv_jj).sqrt();
    if se < 1e-300 || !se.is_finite() {
        return (0.0, 1.0);
    }
    let t = coef / se;
    if !t.is_finite() {
        return (0.0, 1.0);
    }
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_val = 2.0 * t_dist.cdf(-t.abs());
    (t, p_val)
}

/// Compute median of a sorted slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        assert!((median_sorted(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert!((median_sorted(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-10);
        assert!((median_sorted(&[5.0]) - 5.0).abs() < 1e-10);
    }
}
