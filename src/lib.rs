pub mod bed;
pub mod precompute;
pub mod rsvd;
pub mod simulate;
pub mod testing;

use anyhow::Result;
use ndarray::Array2;

use bed::{BedFile, SubsetSpec};
use precompute::precompute;
use rsvd::estimate_factors_streaming;
use testing::{test_associations_fused, TestResults};

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
        }
    }
}

/// Estimate latent factors U_hat from (possibly LD-pruned) Y_est.
///
/// Implements Steps 0-2 of the LFMM2 algorithm:
/// 1. Precompute SVD of X, D_λ, M, ridge_inv
/// 2. Randomized SVD of M @ Y_est via streaming
/// 3. Recover U_hat = Q @ D_λ_inv @ U_small[:, :K]
pub fn estimate_factors(
    y_est: &BedFile,
    x: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<Array2<f64>> {
    let pre = precompute(x, config.lambda)?;
    let subset = SubsetSpec::All;
    estimate_factors_streaming(y_est, &subset, &pre, config)
}

/// Run association tests on all SNPs using pre-estimated U_hat.
///
/// Computes effect sizes (B) and per-locus t-tests in a single fused pass.
/// Returns TestResults with calibrated p-values (GIF correction).
pub fn test_associations(
    y_full: &BedFile,
    x: &Array2<f64>,
    u_hat: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<TestResults> {
    let pre = precompute(x, config.lambda)?;
    test_associations_fused(y_full, x, u_hat, &pre, config)
}

/// Full LFMM2 pipeline: estimate latent factors + test associations.
///
/// - y_est: LD-pruned subset for factor estimation (Steps 0-2)
/// - y_full: All SNPs for testing (Steps 3-4)
/// - x: Covariate matrix (n × d)
pub fn fit_lfmm2(
    y_est: &BedFile,
    y_full: &BedFile,
    x: &Array2<f64>,
    config: &Lfmm2Config,
) -> Result<TestResults> {
    let pre = precompute(x, config.lambda)?;
    let subset = SubsetSpec::All;
    let u_hat = estimate_factors_streaming(y_est, &subset, &pre, config)?;
    test_associations_fused(y_full, x, &u_hat, &pre, config)
}
