use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::InverseInto;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

use crate::bed::{BedFile, SubsetSpec};
use crate::parallel::{parallel_stream, DisjointRowWriter};
use crate::precompute::Precomputed;
use crate::progress::make_progress_bar;
use crate::Lfmm2Config;

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
pub fn test_associations_fused(
    y_full: &BedFile,
    x: &Array2<f64>,
    u_hat: &Array2<f64>,
    pre: &Precomputed,
    config: &Lfmm2Config,
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
    //
    // P_U is the orthogonal projector onto col(U_hat). If U_hat has near-collinear
    // columns (e.g. K exceeds the effective rank of the data, or a latent factor
    // is nearly constant), U^T U becomes singular. We use a regularized inverse
    // to avoid a hard failure — this slightly shrinks the projection but preserves
    // the residual computation (I - P_U)Y needed for Step 3.
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
    //
    // If a covariate in X aligns with a latent factor in U_hat (common when an
    // environmental variable is confounded with population structure), C has
    // collinear columns and C^T C is singular. Regularized inverse prevents the
    // crash; affected standard errors will be slightly inflated (conservative).
    let ctc = c.t().dot(&c);
    let ctc_inv = safe_inv(&ctc, "C^T C  where C = [X | U_hat]")?;

    // H = (C^T C)^{-1} C^T — the OLS hat matrix for coefficient estimation
    let h = ctc_inv.dot(&c.t());

    // Diagonal of (C^T C)^{-1} for standard error computation:
    // se(γ̂_j) = sqrt(σ̂² · (C^T C)^{-1}_{jj})
    let ctc_inv_diag: Vec<f64> = (0..d).map(|j| ctc_inv[(j, j)]).collect();

    // Allocate output arrays
    let mut effect_sizes = Array2::<f64>::zeros((p, d));
    let mut t_stats = Array2::<f64>::zeros((p, d));
    let mut raw_p_values = Array2::<f64>::zeros((p, d));

    // Single fused pass over Y_full
    let subset = SubsetSpec::All;
    let n_chunks = ((p + chunk_size - 1) / chunk_size) as u64;
    let pb = make_progress_bar(n_chunks, "Association tests", config.progress);

    {
        let wr_effects = DisjointRowWriter::new(&mut effect_sizes);
        let wr_tstats = DisjointRowWriter::new(&mut t_stats);
        let wr_pvals = DisjointRowWriter::new(&mut raw_p_values);
        parallel_stream(y_full, &subset, chunk_size, config.n_workers, |block| {
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
            let mut local_pvals = Array2::<f64>::zeros((chunk_cols, d));

            for col_in_chunk in 0..chunk_cols {
                let res_col = residuals.column(col_in_chunk);
                let rss: f64 = res_col.dot(&res_col);
                let sigma2 = rss / df;

                for j in 0..d {
                    let (t, p_val) = t_test(coefs[(j, col_in_chunk)], sigma2, ctc_inv_diag[j], df);
                    local_tstats[(col_in_chunk, j)] = t;
                    local_pvals[(col_in_chunk, j)] = p_val;
                }
            }

            unsafe {
                wr_effects.write_rows(start, &b_chunk_t);
                wr_tstats.write_rows(start, &local_tstats);
                wr_pvals.write_rows(start, &local_pvals);
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

    let mut calibrated_p = raw_p_values.clone();
    let mut total_gif = 0.0;

    for j in 0..d {
        let t_col = t_stats.column(j);
        // Filter non-finite values (shouldn't occur after t_test guard, but be safe)
        let mut z_sq: Vec<f64> = t_col.iter().map(|&t| t * t).filter(|v| v.is_finite()).collect();
        z_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_z_sq = median_sorted(&z_sq);

        // Guard: if median(t²) ≈ 0, all test statistics are near zero (e.g. all SNPs
        // monomorphic or K is too large). GIF is undefined; use 1.0 (no calibration)
        // to avoid division by zero in z_cal = t / sqrt(GIF).
        let gif = if median_z_sq < 1e-10 {
            1.0
        } else {
            median_z_sq / 0.4549
        };
        total_gif += gif;

        let gif_sqrt = gif.sqrt();
        for i in 0..p {
            let z = t_stats[(i, j)];
            let z_cal = z / gif_sqrt;
            let p_cal = 2.0 * normal.cdf(-z_cal.abs());
            calibrated_p[(i, j)] = p_cal;
        }
    }

    let avg_gif = total_gif / d as f64;

    Ok(TestResults {
        u_hat: u_hat.to_owned(),
        effect_sizes,
        t_stats,
        p_values: calibrated_p,
        gif: avg_gif,
    })
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
    if n % 2 == 0 {
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
