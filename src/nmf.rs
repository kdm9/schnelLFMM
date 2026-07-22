use anyhow::Result;
use ndarray::{Array2, s};
use ndarray_linalg::SVD;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use std::sync::Mutex;

use crate::bed::{BedFile, SnpNorm, SubsetSpec};
use crate::parallel::{parallel_stream, ImputeConfig, PerWorkerAccumulator};
use crate::progress::make_progress_bar;

/// Configuration for Non-negative Matrix Factorisation imputation.
#[derive(Clone, Debug)]
pub struct NmfConfig {
    /// Number of NMF components (default = LFMM2 K).
    pub k: usize,
    /// Multiplicative update iterations (default: 10). Multiplicative NMF
    /// converges slowly; watch the per-iteration CV MAE for convergence.
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
    /// Show progress bars.
    pub progress: bool,
    /// Base RNG seed; iteration t uses seed + t for masking.
    pub seed: u64,
}

impl Default for NmfConfig {
    fn default() -> Self {
        NmfConfig {
            k: 5,
            n_iter: 10,
            eps: 1e-16,
            cv_rate: 0.0005,
            chunk_size: 10_000,
            n_workers: 0,
            progress: false,
            seed: 42,
        }
    }
}

/// Result from NMF imputation on the estimation subset.
pub struct NmfEstResult {
    /// Sample factor matrix: n x K, non-negative.
    pub w: Array2<f64>,
    /// SNP factor matrix: K x p_est, non-negative, held in RAM.
    pub h_full: Array2<f64>,
    /// Per-iteration cross-validation MAE (mean absolute error on masked
    /// genotypes, raw dosage scale {0,1,2}). Length = n_iter.
    pub cv_mae_per_iter: Vec<f64>,
    /// Mean-imputation MAE computed on the same mask as the final NMF
    /// iteration (paired comparison), raw dosage scale.
    pub cv_mae_mean_impute: f64,
    /// Total number of masked positions across all iterations.
    pub total_mask_count: u64,
}

/// Perform NMF on the estimation subset, producing W and H_est.
///
/// The factorisation is trained on RAW dosages {0,1,2}: these are naturally
/// non-negative (required for the Lee-Seung multiplicative updates) and match
/// the scale on which W and H are later applied — on-the-fly imputation fills
/// are inserted as raw dosages before centering (see notes §8).
///
/// Uses streaming multiplicative updates over the BedFile subset. Missing
/// genotypes are filled with the CURRENT W @ H reconstruction, rebuilt before
/// every pass (EM-like E-step). A small random fraction of observed genotypes
/// is held out per iteration to compute a cross-validation error on the raw
/// scale, paired against the mean-imputation baseline on the same mask.
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
    let mut cv_mae_mean_impute = 0.0;
    let mut total_mask_count = 0u64;

    for iter in 0..config.n_iter {
        let mask_seed = config.seed + iter as u64;

        // EM-like E-step: missing genotypes are filled with the CURRENT
        // reconstruction W @ H, rebuilt before every pass. The fill config
        // owns cloned arrays, so it is reconstructed each pass.
        let impute_fwd = ImputeConfig::NmfRawInRam {
            w: w.clone(),
            h_full: h_full.clone(),
        };

        // --- Forward pass: update W ---
        // Per-worker accumulators: the floating-point reduction order is fixed
        // regardless of worker scheduling, so results are bitwise reproducible.
        let n_w = config.n_workers.max(1);
        let num_w_acc = PerWorkerAccumulator::new(n_w, (n, k));
        let hh_acc = PerWorkerAccumulator::new(n_w, (k, k));

        {
            let label = format!("NMF iter {}/{} (fwd)", iter + 1, config.n_iter);
            let pb = make_progress_bar(n_chunks, &label, config.progress);

            parallel_stream(
                bed, subset, config.chunk_size, config.n_workers,
                SnpNorm::CenterOnly, impute_fwd.clone(), // ignored by raw decode
                |worker_id, block| {
                    let chunk = block.data.slice(s![.., ..block.n_cols]);
                    let chunk_cols = block.n_cols;

                    // H_chunk for this chunk
                    let h_chunk = impute_nmf_get_h(&h_full, block.seq, config.chunk_size, chunk_cols);

                    // numerator_W = Y_chunk @ H_chunk^T  (n x K)
                    let num_partial = chunk.dot(&h_chunk.t());
                    *num_w_acc.get_mut(worker_id) += &num_partial;

                    // H_chunk @ H_chunk^T  (K x K)
                    let hh_partial = h_chunk.dot(&h_chunk.t());
                    *hh_acc.get_mut(worker_id) += &hh_partial;

                    pb.inc(1);
                },
            );
            pb.finish_and_clear();
        }

        // Update W = W ⊙ num_W ⊘ (W @ HH + ε)
        let num_w = num_w_acc.sum();
        let hh_sum = hh_acc.sum();
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

        // Rebuild the fill config with the updated W (and current H) so the
        // backward pass also imputes from the current reconstruction.
        let impute_bwd = ImputeConfig::NmfRawInRam {
            w: w.clone(),
            h_full: h_full.clone(),
        };

        {
            let label = format!("NMF iter {}/{} (bwd)", iter + 1, config.n_iter);
            let pb = make_progress_bar(n_chunks, &label, config.progress);

            let h_full_mtx = Mutex::new(&mut h_full);
            let w_ref = w.clone();
            let wt_w_ref = wt_w.clone();
            let chunk_sz = config.chunk_size;

            parallel_stream(
                bed, subset, config.chunk_size, config.n_workers,
                SnpNorm::CenterOnly, impute_bwd.clone(), // ignored by raw decode
                |_worker_id, block| {
                    let chunk = block.data.slice(s![.., ..block.n_cols]);
                    let chunk_cols = block.n_cols;

                    let num_h = w_ref.t().dot(&chunk); // K x chunk_cols
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

        // Compute CV error for this iteration (raw scale, paired mask)
        let (mae, mean_mae, count) = compute_cv_error_masked(
            bed, subset, &w, &h_full, mask_seed, config,
        )?;
        cv_mae_per_iter.push(mae);
        cv_mae_mean_impute = mean_mae;
        total_mask_count += count;

        if config.progress {
            eprintln!(
                "  NMF iter {}: cv_mae = {:.6} (mean-impute = {:.6}, n_masked = {})",
                iter + 1, mae, mean_mae, count,
            );
        }
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

/// Initialize W and H via a random probe sketch on raw dosages.
///
/// W_init = |Omega @ Vt[:K, :]^T| (absolute value of approximate left
/// singular vectors of the raw, mean-imputed genotype matrix).
/// H_init = max(0, W_pinv @ Y) via one backward pass with mean imputation,
/// also on the raw dosage scale.
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

    // Z = Y_est^T @ Omega  (p_est x l)
    let mut z = Array2::<f64>::zeros((p_est, l));
    {
        let pb = make_progress_bar(n_chunks, "NMF init probe", config.progress);
        let z_mutex = Mutex::new(&mut z);
        let omega_ref = omega.clone();

        parallel_stream(
            bed, subset, chunk_size, config.n_workers,
            SnpNorm::CenterOnly, // ignored by raw decode
            ImputeConfig::MeanRaw,
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

    // Thin SVD of Z: only need Vt (l x l), NOT full U (p_est x p_est)
    // Computing full U would be p_est^2 entries, too large for RAM
    let (_u_z, _s, vt_z) = z.svd(false, true)?;
    let vt = vt_z.unwrap(); // l x l

    // W_probe = Omega @ Vt[:K, :]^T  (n x l @ l x K = n x K)
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
            SnpNorm::CenterOnly, // ignored by raw decode
            ImputeConfig::MeanRaw,
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

/// Compute cross-validation error on masked genotypes, on the raw dosage
/// scale {0,1,2}.
///
/// Decodes raw genotypes (no fill, no centering), masks a random subset of
/// observed positions, and compares on the same masked positions:
///   - NMF reconstruction error: |g - (W H)_ij|
///   - Mean-imputation baseline: |g - col_mean_j| (mean over observed entries)
///
/// Returns (nmf_mae, mean_impute_mae, n_masked).
///
/// Note: H was trained on these same SNPs, so this is an in-sample
/// convergence diagnostic rather than an honest imputation-accuracy estimate
/// (the GWAS-phase CV measures the deployed on-the-fly pipeline). The paired
/// mask makes the NMF-vs-mean comparison unbiased, and the raw scale matches
/// the scale on which the factorisation is applied.
fn compute_cv_error_masked(
    bed: &BedFile,
    subset: &SubsetSpec,
    w: &Array2<f64>,
    h_full: &Array2<f64>,
    mask_seed: u64,
    config: &NmfConfig,
) -> Result<(f64, f64, u64)> {
    let n_output = bed.n_samples;
    let chunk_size = config.chunk_size;
    let cv_rate = config.cv_rate;

    let indices = crate::parallel::subset_indices(subset, bed.n_snps);
    let bps = bed.bytes_per_snp();
    let n_physical = bed.n_physical_samples;
    let sample_keep = bed.sample_keep.as_deref();

    let mut nmf_err = 0.0f64;
    let mut mean_err = 0.0f64;
    let mut total_cnt = 0u64;

    for (chunk_seq, chunk_indices) in indices.chunks(chunk_size).enumerate() {
        let n_cols = chunk_indices.len();

        // Decode raw genotypes {0,1,2,NaN}: no fill, no centering
        let mut y_raw = Array2::<f64>::zeros((n_output, n_cols));
        {
            let out_view = y_raw.view_mut();
            crate::bed::decode_raw_bed_chunk_into(
                &bed.mmap[3..], bps, n_physical, chunk_indices,
                out_view, sample_keep,
            );
        }

        // Column means over observed entries (mean-imputation baseline)
        let mut col_mean = vec![0.0f64; n_cols];
        for col in 0..n_cols {
            let mut sum = 0.0;
            let mut n_obs = 0u32;
            for row in 0..n_output {
                let v = y_raw[(row, col)];
                if !v.is_nan() {
                    sum += v;
                    n_obs += 1;
                }
            }
            col_mean[col] = if n_obs > 0 { sum / n_obs as f64 } else { 0.0 };
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
                    let val = y_raw[(row, col)];
                    if val.is_nan() {
                        continue;
                    }
                    nmf_err += (val - recon[(row, col)]).abs();
                    mean_err += (val - col_mean[col]).abs();
                    total_cnt += 1;
                }
            }
        }
    }

    let nmf_mae = if total_cnt > 0 { nmf_err / total_cnt as f64 } else { 0.0 };
    let mean_mae = if total_cnt > 0 { mean_err / total_cnt as f64 } else { 0.0 };
    Ok((nmf_mae, mean_mae, total_cnt))
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
