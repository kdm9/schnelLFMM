use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};

use schnellfmm::bed::{BedFile, SubsetSpec};
use schnellfmm::{fit_lfmm2, Lfmm2Config, OutputConfig, SnpNorm};

#[derive(Parser)]
#[command(name = "lfmm2", about = "Latent Factor Mixed Model v2 — GWAS with latent confounders")]
struct Cli {
    /// PLINK .bed file (with .bim/.fam)
    #[arg(short = 'b', long)]
    bed: PathBuf,

    /// Covariate/phenotype file (CSV or TSV, auto-detected from extension).
    /// First row = header, first column = sample ID.
    #[arg(short = 'c', long)]
    cov: PathBuf,

    /// Number of latent factors
    #[arg(short)]
    k: usize,

    /// Ridge penalty
    #[arg(short = 'l', long, default_value = "1e-5")]
    lambda: f64,

    /// Output prefix
    #[arg(short = 'o', long, default_value = "lfmm2_out")]
    out: String,

    /// Worker threads (0 is treated as 1)
    #[arg(short = 't', long, default_value_t = default_threads())]
    threads: usize,

    /// SNPs per chunk
    #[arg(long, default_value = "10000")]
    chunk_size: usize,

    /// RSVD oversampling
    #[arg(long, default_value = "10")]
    oversampling: usize,

    /// RSVD power iterations
    #[arg(long, default_value = "2")]
    power_iter: usize,

    /// RNG seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Separate .bed for estimation (LD-pruned)
    #[arg(long)]
    est_bed: Option<PathBuf>,

    /// Thin estimation SNPs at this rate
    #[arg(long)]
    est_rate: Option<f64>,

    /// Use only samples present in both the .fam and covariate files.
    /// Without this flag, a missing sample in the covariate file is an error.
    #[arg(long)]
    intersect_samples: bool,

    /// Verbose progress output
    #[arg(short = 'v', long)]
    verbose: bool,

    /// SNP normalization mode
    #[arg(long, default_value = "eigenstrat")]
    norm: SnpNorm,

    /// Scale covariate columns to unit variance (in addition to centering).
    /// By default, covariates are centered but not scaled.
    #[arg(long)]
    scale_cov: bool,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1))
        .unwrap_or(1)
}

fn main() -> Result<()> {
    // Force BLAS single-threaded: our worker pool is the sole source of parallelism.
    schnellfmm::with_multithreaded_blas(1, || {});

    let cli = Cli::parse();

    // Load genotype data
    eprintln!("Loading BED file: {}", cli.bed.display());
    let mut bed = BedFile::open(&cli.bed)?;
    eprintln!(
        "  {} samples x {} SNPs",
        bed.n_samples, bed.n_snps
    );

    // Load covariates with sample matching
    eprintln!("Loading covariates from: {}", cli.cov.display());
    let (cov_names, x, kept_indices) =
        load_covariates(&cli.cov, &bed.fam_records, cli.intersect_samples, cli.verbose)?;
    let d = x.ncols();

    // Apply sample subsetting to main BED if needed
    let subsetted = kept_indices.len() < bed.fam_records.len();
    if subsetted {
        bed.subset_samples(kept_indices);
    }
    let n = bed.n_samples;
    eprintln!(
        "  {} samples x {} covariates: [{}]",
        n,
        d,
        cov_names.join(", ")
    );

    // Estimation subset
    if cli.est_bed.is_some() && cli.est_rate.is_some() {
        anyhow::bail!("Cannot specify both --est-bed and --est-rate");
    }

    let est_bed = if let Some(ref est_path) = cli.est_bed {
        eprintln!("Loading estimation BED file: {}", est_path.display());
        let mut eb = BedFile::open(est_path)?;

        // Validate sample identity (not just count) and align order
        align_est_bed_samples(&mut eb, &bed)?;

        eprintln!(
            "  {} samples x {} SNPs (for factor estimation)",
            eb.n_samples, eb.n_snps
        );
        Some(eb)
    } else {
        None
    };

    let est_subset = if let Some(rate) = cli.est_rate {
        if !(0.0..=1.0).contains(&rate) {
            anyhow::bail!("--est-rate must be in (0.0, 1.0], got {}", rate);
        }
        eprintln!("  Estimation subset: thinning at rate {}", rate);
        SubsetSpec::Rate(rate)
    } else {
        SubsetSpec::All
    };

    let config = Lfmm2Config {
        k: cli.k,
        lambda: cli.lambda,
        chunk_size: cli.chunk_size,
        oversampling: cli.oversampling,
        n_power_iter: cli.power_iter,
        seed: cli.seed,
        n_workers: cli.threads,
        progress: std::io::stderr().is_terminal(),
        norm: cli.norm,
        scale_cov: cli.scale_cov,
    };

    eprintln!(
        "Running LFMM2: K={}, lambda={:.1e}, chunk_size={}, threads={}, oversampling={}, power_iter={}",
        config.k, config.lambda, config.chunk_size, config.n_workers,
        config.oversampling, config.n_power_iter,
    );

    // Run LFMM2 with streaming output
    let results_path = PathBuf::from(format!("{}.tsv", cli.out));
    let summary_path = format!("{}.summary.txt", cli.out);

    let output_config = OutputConfig {
        path: &results_path,
        bim: &bed.bim_records,
        cov_names: &cov_names,
    };

    let y_est = est_bed.as_ref().unwrap_or(&bed);
    let results = fit_lfmm2(y_est, &est_subset, &bed, &x, &config, Some(&output_config))?;

    eprintln!("GIF: {:.4}", results.gif);
    eprintln!("Results written to {}", results_path.display());

    {
        let mut f = std::fs::File::create(&summary_path)
            .with_context(|| format!("Failed to create {}", summary_path))?;
        writeln!(f, "GIF: {:.6}", results.gif)?;
        writeln!(f, "n_samples: {}", bed.n_samples)?;
        writeln!(f, "n_snps: {}", bed.n_snps)?;
        writeln!(f, "K: {}", cli.k)?;
        writeln!(f, "d: {}", cov_names.len())?;
        writeln!(f, "covariates: {}", cov_names.join(", "))?;
        writeln!(f, "lambda: {}", cli.lambda)?;
        writeln!(f, "threads: {}", cli.threads)?;
    }

    eprintln!("Done. Output: {}, {}", results_path.display(), summary_path);

    Ok(())
}

/// Validate and align est-bed samples to match the main BED's FAM IIDs.
///
/// After the main BED may have been subsetted (via `--intersect-samples`),
/// this ensures the est-bed has the same samples in the same order.
///
/// - Builds an IID→index map of est-bed's FAM
/// - For each main-BED IID, looks up the est-bed index
/// - If order differs (or est-bed has extra samples), calls `subset_samples`
///   on est-bed with the reordered indices
/// - Errors if any main-BED IID is missing from est-bed
fn align_est_bed_samples(
    est_bed: &mut BedFile,
    main_bed: &BedFile,
) -> Result<()> {
    // Build IID → physical index map for est-bed
    let est_iid_map: HashMap<&str, usize> = est_bed
        .fam_records
        .iter()
        .enumerate()
        .map(|(i, r)| (r.iid.as_str(), i))
        .collect();

    let mut est_indices: Vec<usize> = Vec::with_capacity(main_bed.n_samples);
    let mut missing: Vec<String> = Vec::new();

    for fam_rec in &main_bed.fam_records {
        match est_iid_map.get(fam_rec.iid.as_str()) {
            Some(&idx) => est_indices.push(idx),
            None => missing.push(fam_rec.iid.clone()),
        }
    }

    if !missing.is_empty() {
        anyhow::bail!(
            "est-bed is missing {} sample(s) that are in the main BED file: [{}]{}",
            missing.len(),
            missing.iter().take(5).map(String::as_str).collect::<Vec<_>>().join(", "),
            if missing.len() > 5 { ", ..." } else { "" },
        );
    }

    // Check if reordering/subsetting is needed
    let needs_reorder = est_indices.len() != est_bed.n_samples
        || est_indices.iter().enumerate().any(|(i, &idx)| i != idx);

    if needs_reorder {
        eprintln!(
            "  Reordering/subsetting est-bed samples to match main BED ({} → {} samples)",
            est_bed.n_samples, est_indices.len(),
        );
        est_bed.subset_samples(est_indices);
    }

    Ok(())
}

/// Guess the field delimiter from a file's extension.
/// .csv → comma, everything else (.tsv, .txt, etc.) → tab.
fn guess_delimiter(path: &Path) -> char {
    match path.extension().and_then(|e| e.to_str()) {
        Some("csv") => ',',
        _ => '\t',
    }
}

/// Load a covariate/phenotype file and match samples to .fam order.
///
/// File format:
/// - First row: header. Cell A1 (sample ID column name) may be blank.
/// - First column: sample identifiers (matched against .fam IIDs).
/// - Remaining columns: numeric covariate values.
/// - Delimiter: comma for .csv, tab otherwise.
///
/// When `intersect` is true, FAM samples missing from the covariate file are
/// silently skipped (error only if the intersection is empty).
///
/// Returns (covariate_names, data_matrix, kept_fam_indices).
/// `kept_fam_indices` contains the original .fam row indices of the kept samples.
fn load_covariates(
    path: &Path,
    fam: &[schnellfmm::bed::FamRecord],
    intersect: bool,
    verbose: bool,
) -> Result<(Vec<String>, Array2<f64>, Vec<usize>)> {
    let delim = guess_delimiter(path);
    let delim_name = if delim == ',' { "CSV" } else { "TSV" };

    if verbose {
        eprintln!("  Detected {} format (delimiter: {:?})", delim_name, delim);
    }

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open covariate file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header row
    let header_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("Covariate file is empty"))??;
    let header_fields: Vec<&str> = header_line.split(delim).collect();
    if header_fields.len() < 2 {
        anyhow::bail!(
            "Covariate file header has only {} field(s) — expected at least a sample ID column \
             and one covariate column (delimiter: {:?})",
            header_fields.len(),
            delim,
        );
    }

    // First header field is the sample ID column name (may be blank); rest are covariate names
    let cov_names: Vec<String> = header_fields[1..]
        .iter()
        .map(|s| s.trim().to_string())
        .collect();
    let n_covs = cov_names.len();

    // Parse data rows: sample_id → values
    let mut sample_data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut file_order: Vec<String> = Vec::new();

    for (i, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(delim).collect();
        if fields.len() != n_covs + 1 {
            anyhow::bail!(
                "Line {} has {} fields, expected {} (1 sample ID + {} covariates)",
                i + 2,
                fields.len(),
                n_covs + 1,
                n_covs,
            );
        }

        let sample_id = fields[0].trim().to_string();
        let vals: Vec<f64> = fields[1..]
            .iter()
            .enumerate()
            .map(|(j, s)| {
                let v = s.trim().parse::<f64>().with_context(|| {
                    format!(
                        "Failed to parse value '{}' for covariate '{}' on line {} (sample '{}')",
                        s.trim(),
                        cov_names[j],
                        i + 2,
                        sample_id,
                    )
                })?;
                if !v.is_finite() {
                    anyhow::bail!(
                        "Non-finite value ({}) for covariate '{}' on line {} (sample '{}'): \
                         NaN/Inf will propagate through all matrix operations",
                        v,
                        cov_names[j],
                        i + 2,
                        sample_id,
                    );
                }
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;

        match sample_data.entry(sample_id) {
            Entry::Occupied(e) => {
                anyhow::bail!(
                    "Duplicate sample ID '{}' in covariate file (line {})",
                    e.key(),
                    i + 2,
                );
            }
            Entry::Vacant(e) => {
                file_order.push(e.key().clone());
                e.insert(vals);
            }
        }
    }

    if sample_data.is_empty() {
        anyhow::bail!("Covariate file has no data rows");
    }

    eprintln!(
        "  Read {} samples from covariate file",
        sample_data.len()
    );

    // Match samples to .fam order using IID
    let n = fam.len();
    let mut kept_indices: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for (row, fam_rec) in fam.iter().enumerate() {
        match sample_data.get(&fam_rec.iid) {
            Some(vals) => {
                kept_indices.push(row);
                rows.push(vals.clone());
            }
            None => {
                if intersect {
                    // Skip missing FAM samples silently when intersecting
                } else {
                    anyhow::bail!(
                        "Sample '{}' (FID='{}') from .fam file not found in covariate file.\n\
                         .fam has {} samples, covariate file has {}.\n\
                         First few covariate sample IDs: [{}]\n\
                         Hint: use --intersect-samples to use only samples present in both files.",
                        fam_rec.iid,
                        fam_rec.fid,
                        n,
                        sample_data.len(),
                        file_order.iter().take(5).map(String::as_str).collect::<Vec<_>>().join(", "),
                    );
                }
            }
        }
    }

    let matched = kept_indices.len();
    if matched == 0 {
        anyhow::bail!(
            "No samples in common between .fam file ({} samples) and covariate file ({} samples).\n\
             First few .fam IIDs: [{}]\n\
             First few covariate IDs: [{}]",
            n,
            sample_data.len(),
            fam.iter().take(5).map(|r| r.iid.as_str()).collect::<Vec<_>>().join(", "),
            file_order.iter().take(5).map(String::as_str).collect::<Vec<_>>().join(", "),
        );
    }

    let extra_cov = sample_data.len() - matched;
    let skipped_fam = n - matched;
    if extra_cov > 0 {
        eprintln!(
            "  Warning: {} sample(s) in covariate file not present in .fam file (ignored)",
            extra_cov,
        );
    }
    if skipped_fam > 0 {
        eprintln!(
            "  Warning: {} sample(s) in .fam file not present in covariate file (dropped with --intersect-samples)",
            skipped_fam,
        );
    }

    eprintln!(
        "  Matched {} samples to .fam order",
        matched,
    );

    // Build output matrix
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((matched, n_covs), flat)
        .expect("row count × col count should match flattened length");

    Ok((cov_names, x, kept_indices))
}

