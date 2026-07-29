use anyhow::{Context, Result};
use ndarray::Array2;
use ndarray_linalg::InverseInto;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;

use crate::bed::{decode_raw_bed_chunk_into, BedFile, BimRecord, SubsetSpec};
use crate::parallel::{parallel_stream, ImputeConfig};
use crate::precompute::Precomputed;
use crate::progress::make_progress_bar;
use crate::Lfmm2Config;
use crate::timer::Timer;

/// Output format for final association results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Tab-separated text (default, backward-compatible).
    Tsv,
    /// Columnar Parquet with Zstd compression (DuckDB-ready).
    Parquet,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Tsv => write!(f, "tsv"),
            OutputFormat::Parquet => write!(f, "parquet"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tsv" => Ok(OutputFormat::Tsv),
            "parquet" => Ok(OutputFormat::Parquet),
            _ => Err(format!("unknown output format '{}', expected 'tsv' or 'parquet'", s)),
        }
    }
}

/// Configuration for streaming results output.
///
/// Each chunk writes a Snappy-compressed .parquet fragment during the
/// streaming pass (bim metadata embedded, no p-values yet).  After GIF
/// calibration, fragments are coalesced into the final file in the
/// requested format.
pub struct OutputConfig<'a> {
    pub path: &'a Path,
    pub bim: &'a [BimRecord],
    pub cov_names: &'a [String],
    pub format: OutputFormat,
}

/// Results from the LFMM2 association testing pass.
pub struct TestResults {
    /// Estimated latent factors: n x K
    pub u_hat: Array2<f64>,
    /// Genomic inflation factor: average of the *unclamped* per-trait estimates
    /// (diagnostic). P-value calibration uses per-trait GIFs clamped to >= 1.0.
    pub gif: f64,
    /// NMF cross-validation MAE per iteration (only when NMF imputation is used).
    pub nmf_cv: Option<Vec<f64>>,
    /// Mean imputation CV MAE for comparison (only when NMF imputation is used).
    pub nmf_cv_mean: Option<f64>,
    /// Per-block GWAS-phase CV MAE using the actual NmfOnTheFly estimator
    /// (H_chunk = max(0, W_pinv @ Y)), same method used during GWAS scans.
    pub nmf_gwas_cv_mae: Option<f64>,
    /// Mean-imputation baseline MAE on the same held-out positions.
    pub nmf_gwas_cv_mean_mae: Option<f64>,
    /// Number of masked positions evaluated in the GWAS-phase CV.
    pub nmf_gwas_cv_count: Option<u64>,
}

/// Configuration for per-block NMF cross-validation during the GWAS pass.
///
/// Evaluates the same `H_chunk = max(0, W_pinv @ Y_imputed)` estimator
/// used on-the-fly for imputing missing genotypes during association testing.
pub struct NmfGwasCvConfig {
    pub cv_rate: f64,
    pub seed: u64,
    pub w: Array2<f64>,
    pub w_pinv: Array2<f64>,
}

// ---------------------------------------------------------------------------
// TsqHistogram - unchanged
// ---------------------------------------------------------------------------

/// Streaming histogram for estimating the median of t^2 values per trait.
///
/// Uses fixed-width bins over [0, max_val) to avoid storing all p values.
/// Memory: d x n_bins x 8 bytes (800 KB per trait with default settings).
struct TsqHistogram {
    /// counts[trait_idx][bin_idx]
    counts: Vec<Vec<u64>>,
    n_bins: usize,
    bin_width: f64,
    /// Total count per trait (including clamped values)
    total: Vec<u64>,
}

impl TsqHistogram {
    fn new(d: usize) -> Self {
        let n_bins = 100_000;
        let bin_width = 0.001;
        TsqHistogram {
            counts: vec![vec![0u64; n_bins]; d],
            n_bins,
            bin_width,
            total: vec![0u64; d],
        }
    }

    /// Add a batch of t^2 values for all traits from a chunk.
    fn add_chunk(&mut self, tstats: &Array2<f64>, d: usize) {
        let chunk_cols = tstats.nrows();
        for col in 0..chunk_cols {
            for j in 0..d {
                let t = tstats[(col, j)];
                let t_sq = t * t;
                if t_sq.is_finite() {
                    let bin = ((t_sq / self.bin_width) as usize).min(self.n_bins - 1);
                    self.counts[j][bin] += 1;
                    self.total[j] += 1;
                }
            }
        }
    }

    /// Compute per-trait GIF from the histogram medians.
    /// Returns (gif_per_trait, avg_gif_raw):
    ///
    /// - `gif_per_trait`: clamped to a minimum of 1.0, used for p-value
    ///   calibration.  Deflation (GIF < 1) would make calibration
    ///   anti-conservative (dividing t by sqrt(GIF) < 1 inflates |z|).
    /// - `avg_gif_raw`: average of the *unclamped* per-trait estimates,
    ///   reported as a diagnostic (deflation is itself informative).
    fn compute_gif(&self) -> (Vec<f64>, f64) {
        let d = self.counts.len();
        let mut gif_per_trait = Vec::with_capacity(d);
        let mut total_gif_raw = 0.0;

        for j in 0..d {
            let median_t_sq = self.median_for_trait(j);
            let gif_raw = if median_t_sq < 1e-10 {
                1.0
            } else {
                median_t_sq / 0.4549
            };
            gif_per_trait.push(gif_raw.max(1.0));
            total_gif_raw += gif_raw;
        }

        let avg_gif_raw = total_gif_raw / d as f64;
        (gif_per_trait, avg_gif_raw)
    }

    /// Walk bins to find the median for a given trait.
    fn median_for_trait(&self, j: usize) -> f64 {
        let n = self.total[j];
        if n == 0 {
            return 0.0;
        }
        // For median, we need the value at position n/2 (0-indexed).
        // For even n, we average positions n/2-1 and n/2.
        let is_even = n % 2 == 0;
        let target = if is_even { n / 2 - 1 } else { n / 2 };

        let mut cumulative = 0u64;
        let mut first_bin = None;
        let mut second_bin = None;

        for bin in 0..self.n_bins {
            cumulative += self.counts[j][bin];
            if first_bin.is_none() && cumulative > target {
                first_bin = Some(bin);
                if !is_even {
                    // Odd count: median is the middle value
                    return (bin as f64 + 0.5) * self.bin_width;
                }
            }
            if is_even && first_bin.is_some() && cumulative > target + 1 {
                second_bin = Some(bin);
                break;
            }
            if is_even && first_bin.is_some() && cumulative == target + 1 {
                // The second median value is at exactly this position
                second_bin = Some(bin);
                // But it might be in the next non-empty bin
                if cumulative > target + 1 {
                    break;
                }
            }
        }

        match (first_bin, second_bin) {
            (Some(b1), Some(b2)) => {
                ((b1 as f64 + 0.5) * self.bin_width + (b2 as f64 + 0.5) * self.bin_width) / 2.0
            }
            (Some(b1), None) => (b1 as f64 + 0.5) * self.bin_width,
            _ => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// test_associations_fused -- Steps 3-4 fused in a single pass
// ---------------------------------------------------------------------------

/// Perform Steps 3-4 fused in a single pass over Y_full.
///
/// Step 3: B_hat^T = (X^T X + lambda*I)^{-1} X^T (Y - P_U Y)
///   where P_U = U_hat (U_hat^T U_hat)^{-1} U_hat^T is the orthogonal
///   projector onto col(U_hat).
///
/// Step 4: Per-locus OLS with C = [1 | X | U_hat], t-tests, GIF calibration.
///   For each SNP j: y_j = C gamma_j + epsilon_j, then
///   t_j = gamma_hat[1..d+1] / se(gamma_hat[1..d+1]).
///   Standard errors come from se^2(gamma_hat_j) = sigma_hat^2 * diag((C^T C)^{-1}),
///   where sigma_hat^2 = RSS / df.
///   Degrees of freedom: df = n - d - K - 1 (residual df after fitting
///   intercept + d covariates + K latent factors).
///
/// Each chunk's betas, t-stats and R^2 values are written to a Snappy-compressed
/// .parquet fragment during the streaming pass (p-values not yet computed).
/// After GIF calibration via the streaming histogram, fragments are coalesced
/// into the final output file.
///
/// No p-dimensional arrays are held in RAM - all per-SNP data flows through
/// chunk files on disk.
pub fn test_associations_fused(
    y_full: &BedFile,
    x: &Array2<f64>,
    u_hat: &Array2<f64>,
    pre: &Precomputed,
    config: &Lfmm2Config,
    output: &OutputConfig,
    impute: ImputeConfig,
    nmf_gwas_cv: Option<NmfGwasCvConfig>,
) -> Result<TestResults> {
    let n = y_full.n_samples;
    let p = y_full.n_snps;
    let d = x.ncols();
    let k = config.k;
    let chunk_size = config.chunk_size;

    // Validate degrees of freedom: df = n - 1 - d - K must be positive for a valid
    // t-test.  The -1 accounts for the intercept column in C = [1 | X | U_hat].
    // With df <= 0 the Student-t distribution is undefined, and the usize
    // subtraction would silently wrap around to a huge value.
    if n <= 1 + d + k {
        anyhow::bail!(
            "Insufficient degrees of freedom: n={} samples but 1+d+K=1+{}+{}={}. \
             Need n > 1 + d + K for valid t-tests. Reduce K or add more samples.",
            n, d, k, 1 + d + k,
        );
    }
    let df = (n - 1 - d - k) as f64;

    let timer = Timer::new("OLS hat");
    let (i_minus_pu, xtr, c, ctc_inv, h, hx) = crate::with_multithreaded_blas(config.n_workers, || -> Result<_> {
        // Precompute P_U = U_hat (U^T U)^{-1} U^T (n x n).
        let utu = u_hat.t().dot(u_hat);
        let utu_inv = safe_inv(&utu, "U_hat^T U_hat")?;
        let p_u = u_hat.dot(&utu_inv).dot(&u_hat.t());

        // XtR = (X^T X + lambda*I)^{-1} X^T (d x n) - precomputed ridge projection
        let xtr = pre.ridge_inv.dot(&x.t());

        // I - P_U for Step 3 residual: projects Y onto space orthogonal to U_hat
        let mut i_minus_pu = Array2::<f64>::eye(n);
        i_minus_pu -= &p_u;

        // Step 4 precomputes:
        // C = [1 | X | U_hat] (n x (1+d+K)) - intercept + covariate + latent
        // factor design matrix.
        let c_cols = 1 + d + k;
        let mut c = Array2::<f64>::zeros((n, c_cols));
        c.column_mut(0).fill(1.0); // intercept
        c.slice_mut(ndarray::s![.., 1..1 + d]).assign(x);
        c.slice_mut(ndarray::s![.., 1 + d..]).assign(u_hat);

        // (C^T C)^{-1}: needed for standard errors.
        let ctc = c.t().dot(&c);
        let ctc_inv = safe_inv(&ctc, "C^T C  where C = [1 | X | U_hat]")?;

        // H = (C^T C)^{-1} C^T - the OLS hat matrix for coefficient estimation
        let h = ctc_inv.dot(&c.t());

        // H_X = X (X^T X)^{-1} X^T (n x n): projector onto col(X), used for
        // the sequential (Type-I) variance decomposition.  X is centered, so
        // col(X) is orthogonal to 1, and H_X y is exactly the OLS fit of y
        // on [1 | X].
        let xtx = x.t().dot(x);
        let xtx_inv = safe_inv(&xtx, "X^T X for sequential R^2")?;
        let hx = x.dot(&xtx_inv).dot(&x.t());

        Ok((i_minus_pu, xtr, c, ctc_inv, h, hx))
    })?;
    timer.finish();

    // Diagonal of (C^T C)^{-1} for standard error computation (cov indices 1..d+1)
    let ctc_inv_diag: Vec<f64> = (0..d).map(|j| ctc_inv[(1 + j, 1 + j)]).collect();

    // Create temp dir for chunk files
    let parent = output.path.parent().unwrap_or(Path::new("."));
    let chunk_dir = tempfile::Builder::new()
        .prefix(".lfmm2_chunks_")
        .tempdir_in(parent)
        .context("Failed to create temp directory for chunk files")?;

    // Streaming histogram for GIF estimation (no p-dimensional allocations)
    let histogram = TsqHistogram::new(d);
    let mtx_histogram = Mutex::new(histogram);

    // Single fused pass over Y_full
    let subset = SubsetSpec::All;
    let n_chunks = p.div_ceil(chunk_size);
    let pb = make_progress_bar(n_chunks as u64, "Association tests", config.progress);

    // Per-block CV state using the same NmfOnTheFly estimator
    let nmf_cv_acc: Option<std::sync::Arc<std::sync::Mutex<(f64, f64, u64)>>> =
        nmf_gwas_cv.as_ref().map(|_| {
            std::sync::Arc::new(Mutex::new((0.0f64, 0.0f64, 0u64)))
        });
    let cv_bps = y_full.bytes_per_snp();
    let cv_n_physical = y_full.n_physical_samples;
    let cv_sample_keep: Option<Vec<usize>> = y_full.sample_keep.clone();

    let bim = output.bim;
    let cov_names = output.cov_names;

    {
        parallel_stream(y_full, &subset, chunk_size, config.n_workers, config.norm, impute, |_worker_id, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let chunk_cols = block.n_cols;

            // Step 3: B = (XtR @ (I - P_U) @ chunk)^T
            let residual = i_minus_pu.dot(&chunk);
            let b_chunk = xtr.dot(&residual); // d x chunk_cols
            let b_chunk_t = b_chunk.t().to_owned();

            // Step 4: OLS with C = [1 | X | U_hat]
            let coefs = h.dot(&chunk); // (1+d+K) x chunk_cols
            let fitted = c.dot(&coefs); // n x chunk_cols
            let residuals = &chunk - &fitted; // n x chunk_cols

            // Sequential (Type-I) variance decomposition with order [1, X, U]:
            // fitted_x is the OLS fit of each y_j on [1 | X] alone.  Since
            // col(fitted_x) subset of col(C) and residuals orthogonal to col(C),
            // the exact identity TSS = SS_cov + SS_latent + RSS holds, with
            // SS_cov = ||fitted_x||^2, SS_latent = ||fitted||^2 - ||fitted_x||^2.
            let fitted_x = hx.dot(&chunk); // n x chunk_cols

            let mut local_tstats = Array2::<f64>::zeros((chunk_cols, d));
            let mut local_r2_cov = vec![0.0f64; chunk_cols];
            let mut local_r2_latent = vec![0.0f64; chunk_cols];
            let mut local_r2_resid = vec![0.0f64; chunk_cols];

            for col_in_chunk in 0..chunk_cols {
                let y_col = chunk.column(col_in_chunk);
                let res_col = residuals.column(col_in_chunk);
                let rss: f64 = res_col.dot(&res_col);
                let sigma2 = rss / df;

                for j in 0..d {
                    let (t, _) = t_test(coefs[(1 + j, col_in_chunk)], sigma2, ctc_inv_diag[j], df);
                    local_tstats[(col_in_chunk, j)] = t;
                }

                // Variance decomposition (exact partition of TSS)
                let tss: f64 = y_col.dot(&y_col);
                if tss > 1e-300 {
                    let fx_col = fitted_x.column(col_in_chunk);
                    let ss_cov: f64 = fx_col.dot(&fx_col);
                    // SS_latent = ||fitted||^2 - ||fitted_x||^2 = (TSS - RSS) - SS_cov.
                    // Clamp tiny negative values from floating-point error.
                    let ss_latent: f64 = (tss - rss - ss_cov).max(0.0);
                    local_r2_cov[col_in_chunk] = ss_cov / tss;
                    local_r2_latent[col_in_chunk] = ss_latent / tss;
                    local_r2_resid[col_in_chunk] = rss / tss;
                }
                // else: monomorphic SNP, all zeros -> leave r2 as 0.0
            }

            // Feed t^2 values into streaming histogram
            mtx_histogram.lock().unwrap().add_chunk(&local_tstats, d);

            // Write chunk .parquet fragment (Snappy, bim embedded, no p-values yet)
            let start = block.seq * chunk_size;
            let bim_slice = &bim[start..start + chunk_cols];
            write_chunk_parquet(
                chunk_dir.path(), block.seq, &b_chunk_t, &local_tstats,
                &local_r2_cov, &local_r2_latent, &local_r2_resid,
                bim_slice, cov_names,
            )
                .expect("failed to write chunk file");

            pb.inc(1);

            // Per-block NMF CV using the on-the-fly estimator (same as GWAS imputation)
            if let (Some(ref cv_cfg), Some(ref acc)) = (&nmf_gwas_cv, &nmf_cv_acc) {
                let n_cols = block.n_cols;
                let cv_indices: Vec<usize> = (0..n_cols).collect();

                let mut raw_geno = Array2::<f64>::zeros((n, n_cols));
                {
                    let out_view = raw_geno.view_mut();
                    decode_raw_bed_chunk_into(
                        &block.raw, cv_bps, cv_n_physical, &cv_indices,
                        out_view, cv_sample_keep.as_deref(),
                    );
                }

                let mut masked = raw_geno.clone();
                let mut mask_rng = ChaCha8Rng::seed_from_u64(cv_cfg.seed + block.seq as u64);
                let unif_dist = rand_distr::Uniform::new(0.0f64, 1.0);

                let mut mask_positions: Vec<(usize, usize, f64)> = Vec::new();
                for col in 0..n_cols {
                    for row in 0..n {
                        let val = raw_geno[(row, col)];
                        if !val.is_nan() {
                            let r: f64 = unif_dist.sample(&mut mask_rng);
                            if r < cv_cfg.cv_rate {
                                mask_positions.push((row, col, val));
                                masked[(row, col)] = f64::NAN;
                            }
                        }
                    }
                }

                if !mask_positions.is_empty() {
                    for col in 0..n_cols {
                        let mut sum = 0.0;
                        let mut n_obs = 0u32;
                        for row in 0..n {
                            let v = masked[(row, col)];
                            if !v.is_nan() { sum += v; n_obs += 1; }
                        }
                        let mean = if n_obs > 0 { sum / n_obs as f64 } else { 0.0 };
                        for row in 0..n {
                            if masked[(row, col)].is_nan() {
                                masked[(row, col)] = mean;
                            }
                        }
                    }

                    let h_raw = cv_cfg.w_pinv.dot(&masked);
                    let h_chunk: Array2<f64> = h_raw.mapv(|v| v.max(0.0));
                    let pred = cv_cfg.w.dot(&h_chunk);

                    let mut nmf_err = 0.0f64;
                    let mut mean_err = 0.0f64;
                    for &(row, col, true_val) in &mask_positions {
                        nmf_err += (true_val - pred[(row, col)]).abs();
                        mean_err += (true_val - masked[(row, col)]).abs();
                        // masked[(row, col)] is the column-mean fill for this position
                    }

                    let mut acc_guard = acc.lock().unwrap();
                    acc_guard.0 += nmf_err;
                    acc_guard.1 += mean_err;
                    acc_guard.2 += mask_positions.len() as u64;
                }
            }
        });
    }
    pb.finish_and_clear();

    // Extract GWAS-phase CV results from the shared accumulator
    let (nmf_gwas_cv_mae, nmf_gwas_cv_mean_mae, nmf_gwas_cv_count) =
        match nmf_cv_acc {
            Some(acc) => {
                let (nmf_err, mean_err, count) = *acc.lock().unwrap();
                let nmf_mae = if count > 0 { nmf_err / count as f64 } else { 0.0 };
                let mean_mae = if count > 0 { mean_err / count as f64 } else { 0.0 };
                (Some(nmf_mae), Some(mean_mae), Some(count))
            }
            None => (None, None, None),
        };

    // GIF calibration via streaming histogram
    let (gif_per_trait, avg_gif) = mtx_histogram.into_inner().unwrap().compute_gif();

    // Coalesce chunk files into final output
    match output.format {
        OutputFormat::Tsv => coalesce_output_tsv(
            output.path, output.cov_names, chunk_dir.path(),
            n_chunks, &gif_per_trait, config.progress,
        )?,
        OutputFormat::Parquet => coalesce_output_parquet(
            output.path, output.cov_names, chunk_dir.path(),
            n_chunks, &gif_per_trait, config.progress, config.n_workers,
        )?,
    }

    Ok(TestResults {
        u_hat: u_hat.to_owned(),
        gif: avg_gif,
        nmf_cv: None,
        nmf_cv_mean: None,
        nmf_gwas_cv_mae,
        nmf_gwas_cv_mean_mae,
        nmf_gwas_cv_count,
    })
}

// ---------------------------------------------------------------------------
// Chunk writing -- Snappy-compressed Parquet
// ---------------------------------------------------------------------------

/// Write a chunk's numerical data and BIM metadata as a Parquet file.
///
/// Packs betas (chunk_cols x d), tstats (chunk_cols x d), and r2 values
/// (r2_cov, r2_latent, r2_resid) alongside chr, pos, snp_id from BIM into
/// a Snappy-compressed .parquet fragment.
///
/// Columns: chr, pos, snp_id, beta_{cov}..., t_{cov}..., r2_cov, r2_latent, r2_resid.
///
/// P-values are NOT included -- they depend on GIF which is unknown until
/// all chunks are processed.
fn write_chunk_parquet(
    dir: &Path,
    seq: usize,
    betas: &Array2<f64>,
    tstats: &Array2<f64>,
    r2_cov: &[f64],
    r2_latent: &[f64],
    r2_resid: &[f64],
    bim_slice: &[BimRecord],
    cov_names: &[String],
) -> Result<()> {
    let d = cov_names.len();
    let schema = build_chunk_schema(cov_names);

    let chr_arr = StringArray::from_iter_values(bim_slice.iter().map(|b| b.chrom.as_str()));
    let pos_arr = Int64Array::from_iter_values(bim_slice.iter().map(|b| b.pos as i64));
    let snp_id_arr = StringArray::from_iter_values(bim_slice.iter().map(|b| b.snp_id.as_str()));

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(chr_arr),
        Arc::new(pos_arr),
        Arc::new(snp_id_arr),
    ];

    for j in 0..d {
        let vals: Float64Array = betas.column(j).iter().copied().collect();
        columns.push(Arc::new(vals));
    }
    for j in 0..d {
        let vals: Float64Array = tstats.column(j).iter().copied().collect();
        columns.push(Arc::new(vals));
    }
    columns.push(Arc::new(Float64Array::from_iter_values(r2_cov.iter().copied())));
    columns.push(Arc::new(Float64Array::from_iter_values(r2_latent.iter().copied())));
    columns.push(Arc::new(Float64Array::from_iter_values(r2_resid.iter().copied())));

    let batch = RecordBatch::try_new(schema, columns)
        .map_err(|e| anyhow::anyhow!("Failed to build chunk record batch: {}", e))?;

    let path = dir.join(format!("chunk_{:06}.parquet", seq));
    let file = std::fs::File::create(&path)
        .with_context(|| format!("Failed to create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| anyhow::anyhow!("Failed to create parquet writer for {}: {}", path.display(), e))?;
    writer.write(&batch)
        .map_err(|e| anyhow::anyhow!("Failed to write chunk {}: {}", path.display(), e))?;
    writer.close()
        .map_err(|e| anyhow::anyhow!("Failed to finalise {}: {}", path.display(), e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

fn build_chunk_schema(cov_names: &[String]) -> SchemaRef {
    let mut fields = vec![
        Field::new("chr", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
        Field::new("snp_id", DataType::Utf8, false),
    ];
    for name in cov_names {
        fields.push(Field::new(format!("beta_{}", name), DataType::Float64, false));
    }
    for name in cov_names {
        fields.push(Field::new(format!("t_{}", name), DataType::Float64, false));
    }
    fields.push(Field::new("r2_cov", DataType::Float64, false));
    fields.push(Field::new("r2_latent", DataType::Float64, false));
    fields.push(Field::new("r2_resid", DataType::Float64, false));
    Arc::new(Schema::new(fields))
}

fn build_calibrated_schema(cov_names: &[String]) -> SchemaRef {
    let mut fields = vec![
        Field::new("chr", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
        Field::new("snp_id", DataType::Utf8, false),
    ];
    for name in cov_names {
        fields.push(Field::new(format!("p_{}", name), DataType::Float64, false));
    }
    for name in cov_names {
        fields.push(Field::new(format!("beta_{}", name), DataType::Float64, false));
    }
    for name in cov_names {
        fields.push(Field::new(format!("t_{}", name), DataType::Float64, false));
    }
    fields.push(Field::new("r2_cov", DataType::Float64, false));
    fields.push(Field::new("r2_latent", DataType::Float64, false));
    fields.push(Field::new("r2_resid", DataType::Float64, false));
    Arc::new(Schema::new(fields))
}

// ---------------------------------------------------------------------------
// Parquet reading helper
// ---------------------------------------------------------------------------

fn read_parquet_to_batch(path: &Path) -> Result<RecordBatch> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| anyhow::anyhow!("Failed to create parquet reader for {}: {}", path.display(), e))?;
    let reader = builder.build()
        .map_err(|e| anyhow::anyhow!("Failed to build parquet reader: {}", e))?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        let batch = batch
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", path.display(), e))?;
        batches.push(batch);
    }
    if batches.is_empty() {
        anyhow::bail!("Empty parquet file: {}", path.display());
    }
    if batches.len() == 1 {
        return Ok(batches.remove(0));
    }
    let schema = batches[0].schema();
    let refs: Vec<&RecordBatch> = batches.iter().collect();
    arrow::compute::concat_batches(&schema, refs)
        .map_err(|e| anyhow::anyhow!("Failed to concat batches from {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// Calibration - single code path for p-value computation
// ---------------------------------------------------------------------------

/// Compute GIF-calibrated p-values and insert p_* columns into a record batch.
///
/// For each trait j: p_j = 2 * Phi(-|t_j| / sqrt(GIF_j)), where Phi is the
/// standard normal CDF. This is the single shared code path used by both TSV
/// and Parquet coalescence.
///
/// Chunk schema (input):  chr, pos, snp_id, beta_0..beta_{d-1}, t_0..t_{d-1},
///                         r2_cov, r2_latent, r2_resid
/// Output schema:          chr, pos, snp_id, p_0..p_{d-1}, beta_0..beta_{d-1},
///                         t_0..t_{d-1}, r2_cov, r2_latent, r2_resid
fn calibrate_batch(
    batch: &RecordBatch,
    gif_sqrt: &[f64],
    d: usize,
    cov_names: &[String],
) -> Result<RecordBatch> {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(3 + 3 * d + 3);

    columns.push(batch.column(0).clone()); // chr
    columns.push(batch.column(1).clone()); // pos
    columns.push(batch.column(2).clone()); // snp_id

    for j in 0..d {
        let t_arr = batch.column(3 + d + j)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("t column should be Float64");
        let gs = gif_sqrt[j];
        let p_vals: Float64Array = t_arr.values().iter()
            .map(|&t_val| {
                let z_cal = t_val / gs;
                let p_cal = 2.0 * normal.cdf(-z_cal.abs());
                p_cal
            })
            .collect();
        columns.push(Arc::new(p_vals));
    }

    for j in 0..d {
        columns.push(batch.column(3 + j).clone());
    }
    for j in 0..d {
        columns.push(batch.column(3 + d + j).clone());
    }
    columns.push(batch.column(3 + 2 * d).clone());
    columns.push(batch.column(3 + 2 * d + 1).clone());
    columns.push(batch.column(3 + 2 * d + 2).clone());

    let schema = build_calibrated_schema(cov_names);
    RecordBatch::try_new(schema, columns)
        .map_err(|e| anyhow::anyhow!("Failed to build calibrated record batch: {}", e))
}

// ---------------------------------------------------------------------------
// TSV coalescence (serial)
// ---------------------------------------------------------------------------

/// Coalesce chunk .parquet files into a single TSV with calibrated p-values.
///
/// Reads each chunk via `read_parquet_to_batch`, calibrates p-values with
/// `calibrate_batch`, then writes rows as tab-separated text.
/// Column order: chr, pos, snp_id, p_{cov}, beta_{cov}, t_{cov} per covariate,
/// r2_cov, r2_latent, r2_resid.
fn coalesce_output_tsv(
    output_path: &Path,
    cov_names: &[String],
    chunk_dir: &Path,
    n_chunks: usize,
    gif_per_trait: &[f64],
    progress: bool,
) -> Result<()> {
    let d = cov_names.len();
    let gif_sqrt: Vec<f64> = gif_per_trait.iter().map(|g| g.sqrt()).collect();

    let mut out = BufWriter::new(
        std::fs::File::create(output_path)
            .with_context(|| format!("Failed to create {}", output_path.display()))?,
    );

    // Header
    write!(out, "chr\tpos\tsnp_id")?;
    for name in cov_names {
        write!(out, "\tp_{}\tbeta_{}\tt_{}", name, name, name)?;
    }
    write!(out, "\tr2_cov\tr2_latent\tr2_resid")?;
    writeln!(out)?;

    // Read each chunk .parquet and write rows with calibrated p-values
    let pb = make_progress_bar(n_chunks as u64, "Write output", progress);
    for seq in 0..n_chunks {
        let chunk_path = chunk_dir.join(format!("chunk_{:06}.parquet", seq));
        let batch = read_parquet_to_batch(&chunk_path)?;
        let calibrated = calibrate_batch(&batch, &gif_sqrt, d, cov_names)?;

        let n_rows = calibrated.num_rows();
        let chr_arr = calibrated.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let pos_arr = calibrated.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let snp_arr = calibrated.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..n_rows {
            write!(out, "{}\t{}\t{}", chr_arr.value(i), pos_arr.value(i), snp_arr.value(i))?;
            for j in 0..d {
                let p_idx = 3 + j;
                let beta_idx = 3 + d + j;
                let t_idx = 3 + 2 * d + j;

                let p_val = calibrated.column(p_idx).as_any().downcast_ref::<Float64Array>().unwrap().value(i);
                let beta = calibrated.column(beta_idx).as_any().downcast_ref::<Float64Array>().unwrap().value(i);
                let t_val = calibrated.column(t_idx).as_any().downcast_ref::<Float64Array>().unwrap().value(i);

                write!(out, "\t{:.6e}\t{:.6e}\t{:.6e}", p_val, beta, t_val)?;
            }
            let r2_base = 3 + 3 * d;
            let r2_cov = calibrated.column(r2_base).as_any().downcast_ref::<Float64Array>().unwrap().value(i);
            let r2_lat = calibrated.column(r2_base + 1).as_any().downcast_ref::<Float64Array>().unwrap().value(i);
            let r2_res = calibrated.column(r2_base + 2).as_any().downcast_ref::<Float64Array>().unwrap().value(i);
            write!(out, "\t{:.6e}\t{:.6e}\t{:.6e}", r2_cov, r2_lat, r2_res)?;
            writeln!(out)?;
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    Ok(())
}

// ---------------------------------------------------------------------------
// Parquet coalescence (parallel: workers calibrate, main writes)
// ---------------------------------------------------------------------------

/// Coalesce chunk .parquet files into a single Zstd-compressed Parquet.
///
/// Parallel over chunks: workers read chunks and calibrate p-values via
/// `calibrate_batch` concurrently. Calibrated RecordBatches are sent to the
/// main thread via a crossbeam channel. The main thread reorders by chunk
/// sequence and writes row groups sequentially to ensure correct output
/// ordering. Zstd compression runs in the main thread during row group
/// encoding.
fn coalesce_output_parquet(
    output_path: &Path,
    cov_names: &[String],
    chunk_dir: &Path,
    n_chunks: usize,
    gif_per_trait: &[f64],
    progress: bool,
    n_workers: usize,
) -> Result<()> {
    let d = cov_names.len();
    let gif_sqrt: Arc<Vec<f64>> = Arc::new(gif_per_trait.iter().map(|g| g.sqrt()).collect());
    let cov_names_arc = Arc::new(cov_names.to_owned());
    let output_schema = build_calibrated_schema(cov_names);
    let chunk_dir = chunk_dir.to_path_buf();

    let n_workers = n_workers.min(n_chunks).max(1);

    let (tx, rx) = crossbeam_channel::bounded::<(usize, RecordBatch)>(n_workers * 2);

    std::thread::scope(|s| {
        for w in 0..n_workers {
            let tx = tx.clone();
            let gif_sqrt = gif_sqrt.clone();
            let cov_names = cov_names_arc.clone();
            let chunk_dir = chunk_dir.clone();

            s.spawn(move || {
                let mut seq = w;
                while seq < n_chunks {
                    let chunk_path = chunk_dir.join(format!("chunk_{:06}.parquet", seq));
                    match read_parquet_to_batch(&chunk_path) {
                        Ok(batch) => match calibrate_batch(&batch, &gif_sqrt, d, &cov_names) {
                            Ok(calibrated) => {
                                if tx.send((seq, calibrated)).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                panic!("calibrate chunk {}: {}", seq, e);
                            }
                        },
                        Err(e) => {
                            panic!("read chunk {}: {}", seq, e);
                        }
                    }
                    seq += n_workers;
                }
            });
        }
        drop(tx);

        let file = std::fs::File::create(output_path)
            .with_context(|| format!("Failed to create {}", output_path.display()))
            .unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let mut writer = ArrowWriter::try_new(file, output_schema, Some(props)).unwrap();

        let mut pending: BTreeMap<usize, RecordBatch> = BTreeMap::new();
        let mut next_seq = 0usize;
        let pb = make_progress_bar(n_chunks as u64, "Write output", progress);

        for (seq, batch) in rx {
            pending.insert(seq, batch);
            while let Some(batch) = pending.remove(&next_seq) {
                writer.write(&batch)
                    .with_context(|| format!("Failed to write row group {}", next_seq))
                    .unwrap();
                pb.inc(1);
                next_seq += 1;
            }
        }

        writer.close()
            .map_err(|e| anyhow::anyhow!("Failed to finalise {}: {}", output_path.display(), e))
            .unwrap();
        pb.finish_and_clear();
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// safe_inv, t_test, median_sorted - unchanged
// ---------------------------------------------------------------------------

/// Invert a square matrix, falling back to diagonal regularization if singular.
///
/// Real-data edge cases (collinear covariates, over-specified K) can make
/// U^T U or C^T C singular. Rather than crashing, we add epsilon*I where
/// epsilon = 1e-8 * max(diag(A)). This is small enough to not affect
/// well-conditioned results (relative perturbation ~1e-8) but prevents
/// hard failures.
pub(crate) fn safe_inv(a: &Array2<f64>, name: &str) -> Result<Array2<f64>> {
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
                "Warning: {} is singular, adding epsilon={:.2e} diagonal regularization",
                name, eps,
            );
            let mut a_reg = a.clone();
            for i in 0..n {
                a_reg[(i, i)] += eps;
            }
            a_reg.inv_into().map_err(|e| {
                anyhow::anyhow!(
                    "{} inversion failed even with epsilon={:.2e} regularization: {}",
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
/// For a monomorphic SNP (all samples have the same genotype), the centered
/// column is all zeros. After OLS: residuals = 0, RSS = 0, sigma_hat^2 = 0,
/// se = 0. Then t = beta_hat / se = finite/0 = +/-Inf, which crashes
/// statrs::StudentsT::cdf (it passes t through the beta function which
/// requires finite input).
///
/// We return (t=0, p=1): zero variance -> no evidence of association.
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
#[cfg(test)]
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
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_median() {
        assert!((median_sorted(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert!((median_sorted(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-10);
        assert!((median_sorted(&[5.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_inv_positive_definite() {
        // A = [[2, 1], [1, 3]] - symmetric positive definite (eigenvalues ~1.38, 3.62)
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let inv = safe_inv(&a, "test_pd").unwrap();

        // A @ A^{-1} should be I
        let product = a.dot(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(product[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_safe_inv_singular() {
        // Rank-1 matrix: all rows are multiples of [1, 2] -> singular
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let inv = safe_inv(&a, "test_singular").unwrap();

        // With regularization (A + epsilon*I), the result should be approximately
        // the pseudoinverse-like solution.  Verify that (A + epsilon*I) @ inv ~= I
        // where epsilon = 1e-8 * max(diag(A)) = 1e-8 * 4.0
        let eps = 1e-8 * 4.0;
        let mut a_reg = a.clone();
        a_reg[(0, 0)] += eps;
        a_reg[(1, 1)] += eps;
        let product = a_reg.dot(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(product[(i, j)], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_tsq_histogram_basic() {
        let mut hist = TsqHistogram::new(1);
        // Feed in known t-values: [1, 2, 3, 4, 5] -> t^2=[1, 4, 9, 16, 25]
        // Median t^2 = 9.0, GIF = 9.0 / 0.4549 ~= 19.78
        let tstats = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        hist.add_chunk(&tstats, 1);
        let (gif_per_trait, _avg_gif) = hist.compute_gif();
        let expected_gif = 9.0 / 0.4549;
        assert!((gif_per_trait[0] - expected_gif).abs() < 0.1,
            "GIF mismatch: got {:.4}, expected {:.4}", gif_per_trait[0], expected_gif);
    }

    #[test]
    fn test_calibrate_batch_smoke() {
        let cov_names = vec!["x1".to_string()];
        let schema = build_chunk_schema(&cov_names);

        let chr = StringArray::from_iter_values(["1", "2"].iter().cloned());
        let pos = Int64Array::from_iter_values([100i64, 200i64].iter().copied());
        let snp = StringArray::from_iter_values(["rs001", "rs002"].iter().cloned());
        let beta = Float64Array::from_iter_values([0.5f64, -0.3f64].iter().copied());
        let t = Float64Array::from_iter_values([3.0f64, -2.0f64].iter().copied());
        let r2c = Float64Array::from_iter_values([0.1f64, 0.2f64].iter().copied());
        let r2l = Float64Array::from_iter_values([0.3f64, 0.4f64].iter().copied());
        let r2r = Float64Array::from_iter_values([0.6f64, 0.4f64].iter().copied());

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(chr), Arc::new(pos), Arc::new(snp),
                Arc::new(beta), Arc::new(t),
                Arc::new(r2c), Arc::new(r2l), Arc::new(r2r),
            ],
        ).unwrap();

        let gif_sqrt = vec![1.0];
        let calibrated = calibrate_batch(&batch, &gif_sqrt, 1, &cov_names).unwrap();

        assert_eq!(calibrated.num_columns(), 3 + 3 * 1 + 3);
        let p_arr = calibrated.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(p_arr.value(0) < p_arr.value(1),
            "|t=3| should have smaller p than |t=2|; got p0={:.6e}, p1={:.6e}",
            p_arr.value(0), p_arr.value(1));
    }
}
