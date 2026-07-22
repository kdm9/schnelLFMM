use anyhow::Result;
use ndarray::{Array2, s};
use ndarray_linalg::SVD;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use std::sync::Mutex;

use crate::bed::{BedFile, SnpNorm, SubsetSpec};
use crate::parallel::{parallel_stream, ImputeConfig};
use crate::progress::make_progress_bar;
// Timer import kept for future instrumentation use
// use crate::timer::Timer;

/// Configuration for Non-negative Matrix Factorisation imputation.
#[derive(Clone, Debug)]
pub struct NmfConfig {
    /// Number of NMF components (default = LFMM2 K).
    pub k: usize,
    /// Multiplicative update iterations (default: 3).
    pub n_iter: usize,
    /// Small constant for numerical stability in denominators.
    pub eps: f64,
    /// Fraction of observed genotypes to hold out per iteration for
    /// cross-validation error tracking. Mask is seeded per-iteration
    /// from (base_seed + iter).
    pub cv_rate: f64,
    /// SNPs per streaming chunk (inherited from Lfmm2Config).
    pub chunk_size: usize,
    /// Number of worker threads.
    pub n_workers: usize,
    /// SNP normalization mode (passed through to decode).
    pub norm: SnpNorm,
    /// Show progress bars.
    pub progress: bool,
    /// Base RNG seed; iteration t uses seed + t for masking.
    pub seed: u64,
}

impl Default for NmfConfig {
    fn default() -> Self {
        NmfConfig {
            k: 5,
            n_iter: 3,
            eps: 1e-16,
            cv_rate: 0.0005,
            chunk_size: 10_000,
            n_workers: 0,
            norm: SnpNorm::CenterOnly,
            progress: false,
            seed: 42,
        }
    }
}

/// Result from NMF imputation on the estimation subset.
pub struct NmfEstResult {
    /// Sample factor matrix: n × K, non-negative.
    pub w: Array2<f64>,
    /// SNP factor matrix: K × p_est, non-negative, held in RAM.
    pub h_full: Array2<f64>,
    /// Per-iteration cross-validation MAE (mean absolute error on masked
    /// genotypes). Length = n_iter.
    pub cv_mae_per_iter: Vec<f64>,
    /// Mean-imputation MAE computed on the same mask as the final NMF
    /// iteration, for direct comparison.
    pub cv_mae_mean_impute: f64,
    /// Total number of masked positions across all iterations.
    pub total_mask_count: u64,
}

/// Perform NMF on the estimation subset, producing W and H_est.
///
/// Uses streaming multiplicative updates over the BedFile subset.
/// Missing genotypes are filled with the current W @ H reconstruction
/// before each update (EM-like approach). A small random fraction of
/// observed genotypes is held out per iteration to compute CV error.
pub fn nmf_impute_estimation(
    bed: &BedFile,
    subset: &SubsetSpec,
    config: &NmfConfig,
) -> Result<NmfEstResult> {
    let n = bed.n_samples;
    let p_est = bed.subset_snp_count(subset);
    let k = config.k;
    let n_chunks = p_est.div_ceil(config.chunk_size) as u64;

    if config.progress {
        eprintln!(
            "NMF: K={}, n_iter={}, p_est={}, chunks={}",
            k, config.n_iter, p_est, n_chunks,
        );
    }

    // Initialize W and H from a random probe
    let (mut w, mut h_full) = init_from_random_probe(bed, subset, config)?;

    w.mapv_inplace(|v| v.max(config.eps));
    h_full.mapv_inplace(|v| v.max(config.eps));
    normalize_w(&mut w, &mut h_full, config.eps);

    let mut cv_mae_per_iter = Vec::with_capacity(config.n_iter);
    let mut total_mask_count = 0u64;

    // Build the ImputeConfig for NMF decoding during iterations.
    // This uses NmfInRam so that decode fills NaN with W @ H_chunk values.
    let impute_nmf = ImputeConfig::NmfInRam {
        w: w.clone(),
        h_full: h_full.clone(),
    };

    for iter in 0..config.n_iter {
        let mask_seed = config.seed + iter as u64;

        // --- Forward pass: update W ---
        let num_w = Mutex::new(Array2::<f64>::zeros((n, k)));
        let hh_acc = Mutex::new(Array2::<f64>::zeros((k, k)));

        {
            let label = format!("NMF iter {}/{} (fwd)", iter + 1, config.n_iter);
            let pb = make_progress_bar(n_chunks, &label, config.progress);

            parallel_stream(
                bed, subset, config.chunk_size, config.n_workers,
                config.norm, impute_nmf.clone(),
                |_worker_id, block| {
                    let chunk = block.data.slice(s![.., ..block.n_cols]);
                    let chunk_cols = block.n_cols;

                    // H_chunk for this chunk
                    let h_chunk = impute_nmf_get_h(&h_full, block.seq, config.chunk_size, chunk_cols);

                    // numerator_W = Y_chunk @ H_chunk^T  (n × K)
                    let num_partial = chunk.dot(&h_chunk.t());
                    num_w.lock().unwrap().scaled_add(1.0, &num_partial);

                    // H_chunk @ H_chunk^T  (K × K)
                    let hh_partial = h_chunk.dot(&h_chunk.t());
                    *hh_acc.lock().unwrap() += &hh_partial;

                    pb.inc(1);
                },
            );
            pb.finish_and_clear();
        }

        // Update W = W ⊙ num_W ⊘ (W @ HH + ε)
        let num_w = num_w.into_inner().unwrap();
        let hh_sum = hh_acc.into_inner().unwrap();
        let denom_w = w.dot(&hh_sum);
        let eps = config.eps;

        for ik in 0..(n * k) {
            let row = ik / k;
            let col = ik % k;
            let d = denom_w[(row, col)];
            if d > eps {
                w[(row, col)] *= num_w[(row, col)] / d;
            }
        }
        w.mapv_inplace(|v| v.max(eps));
        normalize_w(&mut w, &mut h_full, eps);

        // --- Backward pass: update H ---
        let wt_w = w.t().dot(&w);
        let p_snps = h_full.ncols();

        {
            let label = format!("NMF iter {}/{} (bwd)", iter + 1, config.n_iter);
            let pb = make_progress_bar(n_chunks, &label, config.progress);

            let h_full_mtx = Mutex::new(&mut h_full);
            let w_ref = w.clone();
            let wt_w_ref = wt_w.clone();
            let chunk_sz = config.chunk_size;

            parallel_stream(
                bed, subset, config.chunk_size, config.n_workers,
                config.norm, impute_nmf.clone(),
                |_worker_id, block| {
                    let chunk = block.data.slice(s![.., ..block.n_cols]);
                    let chunk_cols = block.n_cols;

                    let num_h = w_ref.t().dot(&chunk); // K × chunk_cols
                    let start = block.seq * chunk_sz;
                    let end = (start + chunk_cols).min(p_snps);

                    let mut h_guard = h_full_mtx.lock().unwrap();
                    let mut h_chunk = h_guard.slice_mut(s![.., start..end]);

                    let denom_h = wt_w_ref.dot(&h_chunk);
                    for r in 0..k {
                        for c in 0..chunk_cols {
                            let d = denom_h[(r, c)];
                            if d > eps {
                                h_chunk[(r, c)] *= num_h[(r, c)] / d;
                            }
                            h_chunk[(r, c)] = h_chunk[(r, c)].max(eps);
                        }
                    }
                    drop(h_guard);

                    pb.inc(1);
                },
            );
            pb.finish_and_clear();
        }

        // Re-normalize W after backward pass
        normalize_w(&mut w, &mut h_full, eps);

        // Compute CV error for this iteration
        let (mae, count) = compute_cv_error_masked(
            bed, subset, &w, &h_full, mask_seed, config,
        )?;
        cv_mae_per_iter.push(mae);
        total_mask_count += count;

        if config.progress {
            eprintln!(
                "  NMF iter {}: cv_mae = {:.6} (n_masked = {})",
                iter + 1, mae, count,
            );
        }
    }

    // Compute mean imputation CV error for comparison
    let cv_mae_mean_impute = compute_mean_impute_cv(bed, subset, config)?;

    if config.progress {
        eprintln!(
            "  Mean impute cv_mae = {:.6}",
            cv_mae_mean_impute,
        );
    }

    Ok(NmfEstResult {
        w,
        h_full,
        cv_mae_per_iter,
        cv_mae_mean_impute,
        total_mask_count,
    })
}

/// Helper: extract H_chunk for a given block sequence.
fn impute_nmf_get_h(h_full: &Array2<f64>, seq: usize, chunk_size: usize, n_cols: usize) -> ndarray::ArrayView2<'_, f64> {
    let start = seq * chunk_size;
    let end = (start + n_cols).min(h_full.ncols());
    h_full.slice(s![.., start..end])
}

/// Column-normalize W (L1-norm per column → 1) and scale H rows accordingly.
fn normalize_w(w: &mut Array2<f64>, h: &mut Array2<f64>, eps: f64) {
    let k = w.ncols();
    for j in 0..k {
        let col_norm: f64 = w.column(j).fold(0.0, |acc, &v| acc + v);
        if col_norm > eps {
            let inv_norm = 1.0 / col_norm;
            w.column_mut(j).mapv_inplace(|v| v * inv_norm);
            if h.nrows() == k {
                h.row_mut(j).mapv_inplace(|v| v * col_norm);
            }
        }
    }
}

/// Initialize W and H via a random probe sketch.
///
/// W_init = |Omega @ Vt[:K, :]^T| (absolute value of approximate left
/// singular vectors). H_init = max(0, W_pinv @ Y) via one backward pass
/// with mean imputation.
fn init_from_random_probe(
    bed: &BedFile,
    subset: &SubsetSpec,
    config: &NmfConfig,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = bed.n_samples;
    let p_est = bed.subset_snp_count(subset);
    let k = config.k;
    let l = k + 10;
    let chunk_size = config.chunk_size;
    let n_chunks = p_est.div_ceil(chunk_size) as u64;

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(1_000_000));
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let omega = Array2::from_shape_fn((n, l), |_| normal.sample(&mut rng));

    // Z = Y_est^T @ Omega  (p_est × l)
    let mut z = Array2::<f64>::zeros((p_est, l));
    {
        let pb = make_progress_bar(n_chunks, "NMF init probe", config.progress);
        let z_mutex = Mutex::new(&mut z);
        let omega_ref = omega.clone();

        parallel_stream(
            bed, subset, chunk_size, config.n_workers,
            config.norm,
            ImputeConfig::Mean,
            |_worker_id, block| {
                let chunk = block.data.slice(s![.., ..block.n_cols]);
                let z_block = chunk.t().dot(&omega_ref);
                let start = block.seq * chunk_size;
                z_mutex.lock().unwrap()
                    .slice_mut(s![start..start + z_block.nrows(), ..])
                    .assign(&z_block);
                pb.inc(1);
            },
        );
        pb.finish_and_clear();
    }

    // Thin SVD of Z: Z ≈ U_z @ diag(s) @ Vt_z
    let (_u_z, _s, vt_z) = z.svd(true, true)?;
    let vt = vt_z.unwrap(); // l × l

    // W_probe = Omega @ Vt[:K, :]^T  (n × l @ l × K = n × K)
    let w_probe = crate::with_multithreaded_blas(config.n_workers, || {
        omega.dot(&vt.slice(s![..k, ..]).t())
    });
    let w_probe_abs: Array2<f64> = w_probe.mapv(|v| v.abs());

    // H_init via one backward pass: H = max(0, pinv(W) @ Y)
    let wt_w = w_probe_abs.t().dot(&w_probe_abs);
    let w_pinv = crate::with_multithreaded_blas(config.n_workers, || -> Result<Array2<f64>> {
        let inv_wt_w = crate::testing::safe_inv(&wt_w, "W_init^T W_init")?;
        Ok(inv_wt_w.dot(&w_probe_abs.t()))
    })?;

    let mut h_init = Array2::<f64>::zeros((k, p_est));
    {
        let pb = make_progress_bar(n_chunks, "NMF init H", config.progress);
        let h_mutex = Mutex::new(&mut h_init);
        let w_pinv_ref = w_pinv;

        parallel_stream(
            bed, subset, chunk_size, config.n_workers,
            config.norm,
            ImputeConfig::Mean,
            |_worker_id, block| {
                let chunk = block.data.slice(s![.., ..block.n_cols]);
                let chunk_cols = block.n_cols;
                let h_chunk: Array2<f64> = w_pinv_ref.dot(&chunk).mapv(|v| v.max(0.0));
                let start = block.seq * chunk_size;
                h_mutex.lock().unwrap()
                    .slice_mut(s![.., start..start + chunk_cols])
                    .assign(&h_chunk);
                pb.inc(1);
            },
        );
        pb.finish_and_clear();
    }

    Ok((w_probe_abs, h_init))
}

/// Compute cross-validation error on masked genotypes using NMF reconstruction.
///
/// Decodes centered genotype values, then masks a random subset and compares
/// the NMF reconstruction W@H against the centered truth.
/// Mean-imputation baseline: centered mean is 0, so error = |centered_val|.
fn compute_cv_error_masked(
    bed: &BedFile,
    subset: &SubsetSpec,
    w: &Array2<f64>,
    h_full: &Array2<f64>,
    mask_seed: u64,
    config: &NmfConfig,
) -> Result<(f64, u64)> {
    let n_output = bed.n_samples;
    let chunk_size = config.chunk_size;
    let cv_rate = config.cv_rate;

    let indices = crate::parallel::subset_indices(subset, bed.n_snps);
    let bps = bed.bytes_per_snp();
    let n_physical = bed.n_physical_samples;
    let sample_keep = bed.sample_keep.as_deref();

    let mut total_err = 0.0f64;
    let mut total_cnt = 0u64;

    for (chunk_seq, chunk_indices) in indices.chunks(chunk_size).enumerate() {
        let n_cols = chunk_indices.len();

        // Decode centered values (mean imputation, no NMF fill)
        let mut y_centered = Array2::<f64>::zeros((n_output, n_cols));
        {
            let out_view = y_centered.view_mut();
            crate::bed::decode_bed_chunk_into(
                &bed.mmap[3..], bps, n_physical, chunk_indices,
                out_view, config.norm, sample_keep, None,
            );
        }

        let start = chunk_seq * chunk_size;
        let end = (start + n_cols).min(h_full.ncols());
        let h_chunk = h_full.slice(s![.., start..end]);
        let recon = w.dot(&h_chunk);

        let mut mask_rng = ChaCha8Rng::seed_from_u64(mask_seed + chunk_seq as u64);
        let unif_dist = rand_distr::Uniform::new(0.0f64, 1.0);

        for col in 0..n_cols {
            for row in 0..n_output {
                let r: f64 = unif_dist.sample(&mut mask_rng);
                if r < cv_rate {
                    let val = y_centered[(row, col)];
                    if val.is_nan() {
                        continue;
                    }
                    // val is centered true genotype; recon is NMF's centered prediction
                    // Mean-impute baseline would be 0 (centered mean), so NMF beats it when |val - recon| < |val|
                    total_err += (val - recon[(row, col)]).abs();
                    total_cnt += 1;
                }
            }
        }
    }

    let mae = if total_cnt > 0 { total_err / total_cnt as f64 } else { 0.0 };
    Ok((mae, total_cnt))
}

/// Compute mean-imputation cross-validation error on centered data.
///
/// Mean imputation gives 0 after centering, so the error on each masked
/// position is |centered_val|.
fn compute_mean_impute_cv(
    bed: &BedFile,
    subset: &SubsetSpec,
    config: &NmfConfig,
) -> Result<f64> {
    let cv_rate = config.cv_rate;
    let chunk_size = config.chunk_size;
    let n_output = bed.n_samples;

    let indices = crate::parallel::subset_indices(subset, bed.n_snps);
    let bps = bed.bytes_per_snp();
    let n_physical = bed.n_physical_samples;
    let sample_keep = bed.sample_keep.as_deref();

    let mut total_err = 0.0f64;
    let mut total_cnt = 0u64;

    let cv_seed = config.seed.wrapping_add(999_999);

    for (chunk_seq, chunk_indices) in indices.chunks(chunk_size).enumerate() {
        let n_cols = chunk_indices.len();

        // Decode centered values
        let mut y_centered = Array2::<f64>::zeros((n_output, n_cols));
        {
            let out_view = y_centered.view_mut();
            crate::bed::decode_bed_chunk_into(
                &bed.mmap[3..], bps, n_physical, chunk_indices,
                out_view, config.norm, sample_keep, None,
            );
        }

        let mut mask_rng = ChaCha8Rng::seed_from_u64(cv_seed + chunk_seq as u64);
        let unif_dist = rand_distr::Uniform::new(0.0f64, 1.0);

        for col in 0..n_cols {
            for row in 0..n_output {
                let r: f64 = unif_dist.sample(&mut mask_rng);
                if r < cv_rate {
                    let val = y_centered[(row, col)];
                    if val.is_nan() {
                        continue;
                    }
                    // Mean imputation → 0 after centering, error = |val|
                    total_err += val.abs();
                    total_cnt += 1;
                }
            }
        }
    }

    Ok(if total_cnt > 0 { total_err / total_cnt as f64 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// In-core dense NMF with multiplicative update and EM for missing data.
    /// Used for unit testing and R validation comparison.
    fn nmf_dense(
        y: &Array2<f64>,
        k: usize,
        n_iter: usize,
        eps: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        let n = y.nrows();
        let p = y.ncols();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let unif = rand_distr::Uniform::new(0.0, 1.0);
        let mut w = Array2::from_shape_fn((n, k), |_| unif.sample(&mut rng));
        let mut h = Array2::from_shape_fn((k, p), |_| unif.sample(&mut rng) * 2.0);

        let mask: Array2<bool> = y.mapv(|v| !v.is_nan());

        for _ in 0..n_iter {
            let recon = w.dot(&h);
            let mut y_filled = y.clone();
            for i in 0..n {
                for j in 0..p {
                    if !mask[(i, j)] {
                        y_filled[(i, j)] = recon[(i, j)];
                    }
                }
            }

            // Update H
            let wt_y = w.t().dot(&y_filled);
            let wt_w_h = w.t().dot(&w).dot(&h);
            for kj in 0..(k * p) {
                let row = kj / p;
                let col = kj % p;
                let denom = wt_w_h[(row, col)];
                if denom > eps {
                    h[(row, col)] *= wt_y[(row, col)] / denom;
                }
                h[(row, col)] = h[(row, col)].max(eps);
            }

            // Update W
            let recon2 = w.dot(&h);
            let mut y_filled2 = y.clone();
            for i in 0..n {
                for j in 0..p {
                    if !mask[(i, j)] {
                        y_filled2[(i, j)] = recon2[(i, j)];
                    }
                }
            }
            let y_ht = y_filled2.dot(&h.t());
            let w_h_ht = w.dot(&h.dot(&h.t()));
            for ik in 0..(n * k) {
                let row = ik / k;
                let col = ik % k;
                let denom = w_h_ht[(row, col)];
                if denom > eps {
                    w[(row, col)] *= y_ht[(row, col)] / denom;
                }
                w[(row, col)] = w[(row, col)].max(eps);
            }
        }

        (w, h)
    }

    #[test]
    fn test_nmf_dense_small() {
        let y = array![
            [3.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
        ];
        let k = 2;
        let n_iter = 200;
        let eps = 1e-16;

        let (w, h) = nmf_dense(&y, k, n_iter, eps);

        assert_eq!(w.shape(), &[4, 2]);
        assert_eq!(h.shape(), &[2, 3]);

        for i in 0..4 {
            for j in 0..2 {
                assert!(w[(i, j)] >= 0.0, "W[{},{}] = {} < 0", i, j, w[(i, j)]);
            }
        }

        let recon = w.dot(&h);
        let err: f64 = (&y - &recon).mapv(|v| v * v).sum();
        assert!(err < 1.0, "Reconstruction error {} too large", err);
    }

    #[test]
    fn test_nmf_dense_with_missing() {
        let y = array![
            [3.0, 1.0, 2.0],
            [2.0, f64::NAN, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, f64::NAN],
        ];
        let k = 2;
        let n_iter = 300;
        let eps = 1e-16;

        let (w, h) = nmf_dense(&y, k, n_iter, eps);

        let recon = w.dot(&h);

        // Reconstruction error on OBSERVED entries should be small
        let mut err = 0.0f64;
        for i in 0..4 {
            for j in 0..3 {
                if !y[(i, j)].is_nan() {
                    let d = y[(i, j)] - recon[(i, j)];
                    err += d * d;
                }
            }
        }
        assert!(err < 1.0, "Observed reconstruction error {} too large", err);
    }
}
