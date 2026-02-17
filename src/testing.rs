use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::InverseInto;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

use crate::bed::{BedFile, SubsetSpec};
use crate::precompute::Precomputed;
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
/// Step 4: Per-locus OLS with C = [X | U_hat], t-tests, GIF calibration
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

    // Precompute P_U = U_hat @ inv(U_hat^T U_hat) @ U_hat^T (n × n)
    let utu = u_hat.t().dot(u_hat);
    let utu_inv = utu.inv_into()?;
    let p_u = u_hat.dot(&utu_inv).dot(&u_hat.t());

    // XtR = ridge_inv @ X^T (d × n)
    let xtr = pre.ridge_inv.dot(&x.t());

    // I - P_U for Step 3 residual
    let mut i_minus_pu = Array2::<f64>::eye(n);
    i_minus_pu -= &p_u;

    // Step 4 precomputes:
    // C = [X | U_hat] (n × (d+K))
    let mut c = Array2::<f64>::zeros((n, d + k));
    c.slice_mut(ndarray::s![.., ..d]).assign(x);
    c.slice_mut(ndarray::s![.., d..]).assign(u_hat);

    // CtC_inv = inv(C^T @ C) ((d+K) × (d+K))
    let ctc = c.t().dot(&c);
    let ctc_inv = ctc.inv_into()?;

    // H = CtC_inv @ C^T ((d+K) × n)
    let h = ctc_inv.dot(&c.t());

    // Degrees of freedom for t-test
    let df = (n - d - k) as f64;

    // Diagonal elements of CtC_inv for standard error computation
    // We need CtC_inv[j, j] for each covariate j in 0..d
    let ctc_inv_diag: Vec<f64> = (0..d).map(|j| ctc_inv[(j, j)]).collect();

    // Allocate output arrays
    let mut effect_sizes = Array2::<f64>::zeros((p, d));
    let mut t_stats = Array2::<f64>::zeros((p, d));
    let mut raw_p_values = Array2::<f64>::zeros((p, d));

    // Single fused pass over Y_full
    let subset = SubsetSpec::All;
    for (start, chunk) in y_full.stream_chunks(config.chunk_size, &subset) {
        let chunk_cols = chunk.ncols();

        // Step 3: B = (XtR @ (I - P_U) @ chunk)^T
        let residual = i_minus_pu.dot(&chunk);
        let b_chunk = xtr.dot(&residual); // d × chunk_cols
        effect_sizes
            .slice_mut(ndarray::s![start..start + chunk_cols, ..])
            .assign(&b_chunk.t());

        // Step 4: OLS with C = [X | U_hat]
        let coefs = h.dot(&chunk); // (d+K) × chunk_cols
        let fitted = c.dot(&coefs); // n × chunk_cols
        let residuals = &chunk - &fitted; // n × chunk_cols

        // Residual sum of squares per locus
        // sigma2[j] = sum(residuals[:,j]^2) / df
        for col_in_chunk in 0..chunk_cols {
            let res_col = residuals.column(col_in_chunk);
            let rss: f64 = res_col.dot(&res_col);
            let sigma2 = rss / df;

            let snp_idx = start + col_in_chunk;
            for j in 0..d {
                let se = (sigma2 * ctc_inv_diag[j]).sqrt();
                let t = coefs[(j, col_in_chunk)] / se;
                t_stats[(snp_idx, j)] = t;

                // Two-sided p-value from Student's t distribution
                let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
                let p_val = 2.0 * t_dist.cdf(-t.abs());
                raw_p_values[(snp_idx, j)] = p_val;
            }
        }
    }

    // GIF calibration (genomic inflation factor)
    // Convert t-stats to z-scores, compute GIF = median(z^2) / 0.456
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Calibrate each covariate column independently
    let mut calibrated_p = raw_p_values.clone();
    let mut total_gif = 0.0;

    for j in 0..d {
        let t_col = t_stats.column(j);
        // z-scores: use t-stats directly as approximate z-scores for large df
        let mut z_sq: Vec<f64> = t_col.iter().map(|&t| t * t).collect();
        z_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_z_sq = median_sorted(&z_sq);
        let gif = median_z_sq / 0.456;
        total_gif += gif;

        // Calibrate: z_cal = z / sqrt(GIF), p_cal = 2 * Phi(-|z_cal|)
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
