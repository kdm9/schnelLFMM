pub mod bed;
pub mod parallel;
pub mod precompute;
pub mod progress;
pub mod rsvd;
pub mod simulate;
pub mod testing;

use anyhow::Result;
use ndarray::{Array2, Axis};

use bed::{BedFile, SubsetSpec};
pub use bed::SnpNorm;
use precompute::precompute;
use rsvd::estimate_factors_streaming;
use testing::{test_associations_fused, TestResults};
pub use testing::OutputConfig;

extern "C" {
    fn openblas_set_num_threads(num_threads: std::ffi::c_int);
}

/// Temporarily enable multithreaded BLAS for a serial section,
/// then restore single-threaded mode for the next parallel sweep.
pub fn with_multithreaded_blas<T>(n_workers: usize, f: impl FnOnce() -> T) -> T {
    let n = (n_workers.max(1)) as std::ffi::c_int;
    unsafe { openblas_set_num_threads(n); }
    let result = f();
    unsafe { openblas_set_num_threads(1); }
    result
}

/// Configuration for the LFMM2 algorithm.
pub struct Lfmm2Config {
    /// Number of latent factors (K)
    pub k: usize,
    /// Ridge penalty (λ)
    pub lambda: f64,
    /// SNPs per streaming chunk (default: 10_000)
    pub chunk_size: usize,
    /// Randomized SVD oversampling (default: 10)
    pub oversampling: usize,
    /// Power iterations for randomized SVD (default: 2)
    pub n_power_iter: usize,
    /// RNG seed for reproducibility
    pub seed: u64,
    /// Number of worker threads for parallel chunk processing.
    ///
    /// 0 is treated as 1 — all paths use the parallel streaming infrastructure
    /// (1 decoder thread + n_workers worker threads via crossbeam channels).
    ///
    /// During parallel streaming sweeps, BLAS is single-threaded (each worker
    /// calls BLAS independently). Between sweeps, BLAS is temporarily set to
    /// n_workers threads via `with_multithreaded_blas` for serial linear algebra.
    pub n_workers: usize,
    /// Show progress bars on stderr for streaming passes.
    pub progress: bool,
    /// SNP normalization mode.
    pub norm: SnpNorm,
    /// Whether to scale (divide by std-dev) covariate columns after centering.
    pub scale_cov: bool,
}

impl Default for Lfmm2Config {
    fn default() -> Self {
        Lfmm2Config {
            k: 5,
            lambda: 1e-5,
            chunk_size: 10_000,
            oversampling: 10,
            n_power_iter: 2,
            seed: 42,
            n_workers: 0,
            progress: false,
            norm: SnpNorm::default(),
            scale_cov: false,
        }
    }
}

/// Center columns of X (subtract column means). If `scale` is true, also
/// divide each column by its standard deviation (like R's `scale()`).
pub fn center_covariates(x: &Array2<f64>, scale: bool) -> Array2<f64> {
    let n = x.nrows() as f64;
    let means = x.mean_axis(Axis(0)).unwrap();
    let mut xs = x - &means.insert_axis(Axis(0));
    if scale {
        for j in 0..xs.ncols() {
            let col = xs.column(j);
            let var = col.dot(&col) / (n - 1.0);
            let sd = var.sqrt();
            if sd > 1e-14 {
                xs.column_mut(j).mapv_inplace(|v| v / sd);
            }
        }
    }
    xs
}

/// Estimate latent factors U_hat from (possibly LD-pruned) Y_est.
///
/// Implements Steps 0-2 of the LFMM2 algorithm:
/// 1. Precompute SVD of X, D_λ, M, ridge_inv
/// 2. Randomized SVD of M @ Y_est via streaming
/// 3. Recover U_hat = Q @ D_λ_inv @ U_small[:, :K]
///
/// X is centered (and optionally scaled) before use.
pub fn estimate_factors(
    y_est: &BedFile,
    subset: &SubsetSpec,
    x: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<Array2<f64>> {
    let xs = center_covariates(x, config.scale_cov);
    let pre = with_multithreaded_blas(config.n_workers, || precompute(&xs, config.lambda))?;
    estimate_factors_streaming(y_est, subset, &pre, config)
}

/// Run association tests on all SNPs using pre-estimated U_hat.
///
/// Computes effect sizes (B) and per-locus t-tests in a single fused pass.
/// Returns TestResults with calibrated p-values (GIF correction).
///
/// X is centered (and optionally scaled) before use.
pub fn test_associations(
    y_full: &BedFile,
    x: &Array2<f64>,
    u_hat: &Array2<f64>,
    config: &Lfmm2Config,
    output: Option<&OutputConfig>,
) -> Result<TestResults> {
    let xs = center_covariates(x, config.scale_cov);
    let pre = with_multithreaded_blas(config.n_workers, || precompute(&xs, config.lambda))?;
    test_associations_fused(y_full, &xs, u_hat, &pre, config, output)
}

/// Full LFMM2 pipeline: estimate latent factors + test associations.
///
/// - y_est: BedFile for factor estimation (may be LD-pruned or the same as y_full)
/// - est_subset: which SNPs from y_est to use (All, Rate, or Indices)
/// - y_full: All SNPs for testing (Steps 3-4)
/// - x: Covariate matrix (n × d) — centered (and optionally scaled) internally
pub fn fit_lfmm2(
    y_est: &BedFile,
    est_subset: &SubsetSpec,
    y_full: &BedFile,
    x: &Array2<f64>,
    config: &Lfmm2Config,
    output: Option<&OutputConfig>,
) -> Result<TestResults> {
    let xs = center_covariates(x, config.scale_cov);
    if config.progress {
        eprintln!("Precomputing SVD of X...");
    }
    let pre = with_multithreaded_blas(config.n_workers, || precompute(&xs, config.lambda))?;
    if config.progress {
        let p_est = y_est.subset_snp_count(est_subset);
        eprintln!(
            "Estimating latent factors (RSVD, {} power iterations, {} estimation SNPs)...",
            config.n_power_iter, p_est,
        );
    }
    let u_hat = estimate_factors_streaming(y_est, est_subset, &pre, config)?;
    if config.progress {
        eprintln!("Testing associations...");
    }
    test_associations_fused(y_full, &xs, &u_hat, &pre, config, output)
}
