use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::SVD;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use crate::bed::{BedFile, SubsetSpec};
use crate::precompute::Precomputed;
use crate::Lfmm2Config;

/// Estimate latent factors U_hat via randomized SVD of M @ Y_est.
///
/// This implements Steps 1-2 of the LFMM2 algorithm:
/// - Step 1a: Initial sketch Z = Y_est^T @ (M^T @ Omega)
/// - Step 1b: Power iterations to refine column space
/// - Step 2: Project, small SVD, recover U_hat = Q @ D_lambda_inv @ U_small[:, :K]
///
/// The algorithm streams over Y_est multiple times (2q+2 passes where q = n_power_iter).
pub fn estimate_factors_streaming(
    y_est: &BedFile,
    subset: &SubsetSpec,
    pre: &Precomputed,
    config: &Lfmm2Config,
) -> Result<Array2<f64>> {
    let n = y_est.n_samples;
    let k = config.k;
    let oversample = config.oversampling;
    let l = k + oversample; // sketch dimension
    let p_est = y_est.subset_snp_count(subset);

    // Generate random sketch matrix Omega (n × l)
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Array2::from_shape_fn((n, l), |_| normal.sample(&mut rng));

    // Precompute Mt_omega = M^T @ Omega (n × l)
    let mt_omega = pre.m.t().dot(&omega);

    // Step 1a: Initial sketch
    // Z = Y_est^T @ Mt_omega (p_est × l)
    let mut z = Array2::<f64>::zeros((p_est, l));
    {
        let mut row_offset = 0;
        for (_start, chunk) in y_est.stream_chunks(config.chunk_size, subset) {
            let chunk_cols = chunk.ncols();
            // Z[row_offset..row_offset+chunk_cols, :] = chunk^T @ Mt_omega
            let z_block = chunk.t().dot(&mt_omega);
            z.slice_mut(ndarray::s![row_offset..row_offset + chunk_cols, ..])
                .assign(&z_block);
            row_offset += chunk_cols;
        }
    }

    // QR of Z -> Q_z
    let mut q_z = qr_q(&z);

    // Step 1b: Power iterations
    for _iter in 0..config.n_power_iter {
        // Forward pass: A @ Q_z = M @ Y_est @ Q_z (n × l, accumulated)
        let mut a_qz = Array2::<f64>::zeros((n, l));
        {
            let mut row_offset = 0;
            for (_start, chunk) in y_est.stream_chunks(config.chunk_size, subset) {
                let chunk_cols = chunk.ncols();
                // chunk is n × chunk_cols
                // Q_z_block is chunk_cols × l
                let q_z_block =
                    q_z.slice(ndarray::s![row_offset..row_offset + chunk_cols, ..]);
                // M @ chunk @ Q_z_block
                let y_qz = chunk.dot(&q_z_block);
                a_qz = a_qz + pre.m.dot(&y_qz);
                row_offset += chunk_cols;
            }
        }

        // QR orthogonalize
        let q_aqz = qr_q(&a_qz);

        // Backward pass: A^T @ Q_aqz = Y_est^T @ (M^T @ Q_aqz) (p_est × l)
        let mt_q = pre.m.t().dot(&q_aqz);
        z = Array2::<f64>::zeros((p_est, l));
        {
            let mut row_offset = 0;
            for (_start, chunk) in y_est.stream_chunks(config.chunk_size, subset) {
                let chunk_cols = chunk.ncols();
                let z_block = chunk.t().dot(&mt_q);
                z.slice_mut(ndarray::s![row_offset..row_offset + chunk_cols, ..])
                    .assign(&z_block);
                row_offset += chunk_cols;
            }
        }

        q_z = qr_q(&z);
    }

    // Step 2: Project and recover SVD
    // B_svd = A @ Q_z = M @ Y_est @ Q_z (n × l)
    let mut b_svd = Array2::<f64>::zeros((n, l));
    {
        let mut row_offset = 0;
        for (_start, chunk) in y_est.stream_chunks(config.chunk_size, subset) {
            let chunk_cols = chunk.ncols();
            let q_z_block =
                q_z.slice(ndarray::s![row_offset..row_offset + chunk_cols, ..]);
            let y_qz = chunk.dot(&q_z_block);
            b_svd = b_svd + pre.m.dot(&y_qz);
            row_offset += chunk_cols;
        }
    }

    // Small SVD of B_svd (n × l)
    let (u_opt, _s, _vt_opt) = b_svd.svd(true, false)?;
    let u_small = u_opt.expect("SVD should return U");

    // Recover: U_hat = Q @ D_lambda_inv @ U_small[:, :K]
    // U_small is n × l, take first K columns
    let u_small_k = u_small.slice(ndarray::s![.., ..k]).to_owned();

    // Apply D_lambda_inv as diagonal: scale each row i by d_lambda_inv[i]
    let mut dlam_inv_u = u_small_k.clone();
    for i in 0..n {
        dlam_inv_u
            .row_mut(i)
            .mapv_inplace(|v| v * pre.d_lambda_inv[i]);
    }

    // U_hat = Q @ dlam_inv_u
    let u_hat = pre.q_full.dot(&dlam_inv_u);

    Ok(u_hat)
}

/// Compute the thin QR factorization and return Q.
///
/// For a matrix A (m × n), returns Q (m × min(m, n)).
fn qr_q(a: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);

    // Use Gram-Schmidt orthogonalization (modified for numerical stability)
    // For our use case, the matrices are either p×l (tall, l small) or n×l (small)
    let mut q = a.clone();

    for j in 0..k {
        // Orthogonalize column j against previous columns
        for i in 0..j {
            let dot: f64 = q.column(i).dot(&q.column(j));
            let qi = q.column(i).to_owned();
            q.column_mut(j)
                .zip_mut_with(&qi, |v, &qi_v| *v -= dot * qi_v);
        }

        // Normalize
        let norm: f64 = q.column(j).dot(&q.column(j)).sqrt();
        if norm > 1e-14 {
            q.column_mut(j).mapv_inplace(|v| v / norm);
        }
    }

    // Re-orthogonalize (second pass for stability)
    for j in 0..k {
        for i in 0..j {
            let dot: f64 = q.column(i).dot(&q.column(j));
            let qi = q.column(i).to_owned();
            q.column_mut(j)
                .zip_mut_with(&qi, |v, &qi_v| *v -= dot * qi_v);
        }
        let norm: f64 = q.column(j).dot(&q.column(j)).sqrt();
        if norm > 1e-14 {
            q.column_mut(j).mapv_inplace(|v| v / norm);
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_qr_orthogonality() {
        let a = Array2::from_shape_fn((10, 3), |(_i, _j)| rand::random::<f64>());
        let q = qr_q(&a);

        // Q^T Q should be approximately identity
        let qtq = q.t().dot(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[(i, j)] - expected).abs() < 1e-10,
                    "Q^T Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_column_space() {
        // QR should preserve the column space
        let a = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
        let q = qr_q(&a);

        // Each original column should be in the span of Q's columns
        for j in 0..2 {
            let col = a.column(j);
            // Project onto Q's columns
            let mut proj = Array1::<f64>::zeros(5);
            for k in 0..2 {
                let coef = q.column(k).dot(&col);
                proj = proj + &(q.column(k).to_owned() * coef);
            }
            // Residual should be near zero
            let residual: f64 = (&col.to_owned() - &proj)
                .mapv(|v| v * v)
                .sum()
                .sqrt();
            assert!(
                residual < 1e-10,
                "Column {} has residual {} after projection",
                j,
                residual
            );
        }
    }
}
