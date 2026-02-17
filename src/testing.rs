use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::InverseInto;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

use crate::bed::{BedFile, SubsetSpec};
use crate::parallel::{parallel_stream, DisjointRowWriter};
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
    let chunk_size = config.chunk_size;

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
    let ctc_inv_diag: Vec<f64> = (0..d).map(|j| ctc_inv[(j, j)]).collect();

    // Allocate output arrays
    let mut effect_sizes = Array2::<f64>::zeros((p, d));
    let mut t_stats = Array2::<f64>::zeros((p, d));
    let mut raw_p_values = Array2::<f64>::zeros((p, d));

    // Single fused pass over Y_full
    let subset = SubsetSpec::All;
    if config.n_workers > 0 {
        // Pattern A: scatter — each chunk writes to disjoint rows of 3 output arrays
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
        });
    } else {
        for (start, chunk) in y_full.stream_chunks(chunk_size, &subset) {
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

            for col_in_chunk in 0..chunk_cols {
                let res_col = residuals.column(col_in_chunk);
                let rss: f64 = res_col.dot(&res_col);
                let sigma2 = rss / df;

                let snp_idx = start + col_in_chunk;
                for j in 0..d {
                    let (t, p_val) = t_test(coefs[(j, col_in_chunk)], sigma2, ctc_inv_diag[j], df);
                    t_stats[(snp_idx, j)] = t;
                    raw_p_values[(snp_idx, j)] = p_val;
                }
            }
        }
    }

    // GIF calibration (genomic inflation factor)
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut calibrated_p = raw_p_values.clone();
    let mut total_gif = 0.0;

    for j in 0..d {
        let t_col = t_stats.column(j);
        let mut z_sq: Vec<f64> = t_col.iter().map(|&t| t * t).filter(|v| v.is_finite()).collect();
        z_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_z_sq = median_sorted(&z_sq);
        let gif = median_z_sq / 0.456;
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

/// Compute t-statistic and two-sided p-value, guarding against zero-variance SNPs.
///
/// When se ≈ 0 (monomorphic SNP), t would be Inf/NaN which crashes statrs::StudentsT::cdf.
/// In that case we return t=0, p=1 (no evidence of association).
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
