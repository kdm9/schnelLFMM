use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::{QR, SVD};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::sync::Mutex;

use crate::bed::{BedFile, SubsetSpec};
use crate::parallel::{parallel_stream, PerWorkerAccumulator};
use crate::precompute::Precomputed;
use crate::progress::make_progress_bar;
use crate::Lfmm2Config;
use crate::timer::Timer;



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
    let chunk_size = config.chunk_size;
    let n_chunks = p_est.div_ceil(chunk_size) as u64;
    let show = config.progress;

    // Generate random sketch matrix Omega (n × l)
    let t = Timer::new("generate omega");
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Array2::from_shape_fn((n, l), |_| normal.sample(&mut rng));
    t.finish();

    // Precompute Mt_omega = M^T @ Omega (n × l)
    let t = Timer::new("Mt @ omega");
    let mt_omega = crate::with_multithreaded_blas(config.n_workers, || pre.m.t().dot(&omega));
    t.finish();

    // Step 1a: Initial sketch
    // Z = Y_est^T @ Mt_omega (p_est × l)
    let mut z = Array2::<f64>::zeros((p_est, l));
    {
        let t = Timer::new("initial sketch (parallel)");
        let pb = make_progress_bar(n_chunks, "RSVD sketch", show);
        let z_mutex = Mutex::new(&mut z);
        parallel_stream(y_est, subset, chunk_size, config.n_workers, config.norm, |_worker_id, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let z_block = chunk.t().dot(&mt_omega);
            let start = block.seq * chunk_size;
            z_mutex.lock().unwrap()
                .slice_mut(ndarray::s![start..start + z_block.nrows(), ..])
                .assign(&z_block);
            pb.inc(1);
        });
        pb.finish_and_clear();
        t.finish();
    }

    let t = Timer::new("QR of Z (initial)");
    let (mut q_z, _) = crate::with_multithreaded_blas(config.n_workers, || z.qr().expect("QR failed"));
    t.finish();

    // Step 1b: Power iterations
    for iter in 0..config.n_power_iter {
        // Forward pass: A @ Q_z = M @ Y_est @ Q_z (n × l, accumulated)
        let a_qz;
        {
            let t = Timer::new("power iter fwd (parallel)");
            let label = format!("Power iter {}/{} (fwd)", iter + 1, config.n_power_iter);
            let pb = make_progress_bar(n_chunks, &label, show);
            let n_w = config.n_workers.max(1);
            let acc = PerWorkerAccumulator::new(n_w, (n, l));
            parallel_stream(y_est, subset, chunk_size, config.n_workers, config.norm, |worker_id, block| {
                let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
                let offset = block.seq * chunk_size;
                let q_z_block =
                    q_z.slice(ndarray::s![offset..offset + block.n_cols, ..]);
                let y_qz = chunk.dot(&q_z_block);
                let partial = pre.m.dot(&y_qz);
                *acc.get_mut(worker_id) += &partial;
                pb.inc(1);
            });
            a_qz = acc.sum();
            pb.finish_and_clear();
            t.finish();
        }

        let t = Timer::new("power iter QR(a_qz) + Mt@Q (BLAS)");
        let mt_q = crate::with_multithreaded_blas(config.n_workers, || {
            let (q_aqz, _) = a_qz.qr().expect("QR failed");
            pre.m.t().dot(&q_aqz)
        });
        t.finish();

        // Backward pass: A^T @ Q_aqz = Y_est^T @ (M^T @ Q_aqz) (p_est × l)
        z = Array2::<f64>::zeros((p_est, l));
        {
            let t = Timer::new("power iter bwd (parallel)");
            let label = format!("Power iter {}/{} (bwd)", iter + 1, config.n_power_iter);
            let pb = make_progress_bar(n_chunks, &label, show);
            let z_mutex = Mutex::new(&mut z);
            parallel_stream(y_est, subset, chunk_size, config.n_workers, config.norm, |_worker_id, block| {
                let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
                let z_block = chunk.t().dot(&mt_q);
                let start = block.seq * chunk_size;
                z_mutex.lock().unwrap()
                    .slice_mut(ndarray::s![start..start + z_block.nrows(), ..])
                    .assign(&z_block);
                pb.inc(1);
            });
            pb.finish_and_clear();
            t.finish();
        }

        let t = Timer::new("QR of Z (power iter)");
        let (q, _) = crate::with_multithreaded_blas(config.n_workers, || z.qr().expect("QR failed"));
        q_z = q;
        t.finish();
    }

    // Step 2: Project and recover SVD
    // B_svd = A @ Q_z = M @ Y_est @ Q_z (n × l)
    let b_svd;
    {
        let t = Timer::new("final projection (parallel)");
        let pb = make_progress_bar(n_chunks, "RSVD project", show);
        let n_w = config.n_workers.max(1);
        let acc = PerWorkerAccumulator::new(n_w, (n, l));
        parallel_stream(y_est, subset, chunk_size, config.n_workers, config.norm, |worker_id, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let offset = block.seq * chunk_size;
            let q_z_block =
                q_z.slice(ndarray::s![offset..offset + block.n_cols, ..]);
            let y_qz = chunk.dot(&q_z_block);
            let partial = pre.m.dot(&y_qz);
            *acc.get_mut(worker_id) += &partial;
            pb.inc(1);
        });
        b_svd = acc.sum();
        pb.finish_and_clear();
        t.finish();
    }

    let t = Timer::new("final SVD + recover U_hat (BLAS)");
    let u_hat = crate::with_multithreaded_blas(config.n_workers, || -> Result<Array2<f64>> {
        // Small SVD of B_svd (n × l)
        let (u_opt, _s, _vt_opt) = b_svd.svd(true, false)?;
        let u_small = u_opt.expect("SVD should return U");

        // Recover: U_hat = Q @ D_lambda_inv @ U_small[:, :K]
        // U_small is n × l, take first K columns
        // Apply D_lambda_inv as diagonal: scale each row i by d_lambda_inv[i]
        let d_col = pre.d_lambda_inv.view().insert_axis(ndarray::Axis(1)); // (n,) → (n, 1)
        let dlam_inv_u = &d_col * &u_small.slice(ndarray::s![.., ..k]);

        // U_hat = Q @ dlam_inv_u
        Ok(pre.q_full.dot(&dlam_inv_u))
    })?;
    t.finish();

    Ok(u_hat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_qr_orthogonality() {
        let a = Array2::from_shape_fn((10, 3), |(_i, _j)| rand::random::<f64>());
        let (q, _) = a.qr().unwrap();

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
        let (q, _) = a.qr().unwrap();

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
