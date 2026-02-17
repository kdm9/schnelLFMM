use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};

use lfmm2::bed::{BedFile, SubsetSpec};
use lfmm2::{fit_lfmm2, Lfmm2Config};

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

    /// Verbose progress output
    #[arg(short = 'v', long)]
    verbose: bool,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1))
        .unwrap_or(1)
}

extern "C" {
    fn openblas_set_num_threads(num_threads: std::ffi::c_int);
}

fn main() -> Result<()> {
    // Force BLAS single-threaded: our worker pool is the sole source of parallelism.
    // set_var as belt-and-suspenders (may not take effect if BLAS is statically linked
    // and initializes its thread pool before main), plus FFI call for runtime override.
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    unsafe { openblas_set_num_threads(1); }

    let cli = Cli::parse();

    // --- Load genotype data ---
    eprintln!("Loading BED file: {}", cli.bed.display());
    let bed = BedFile::open(&cli.bed)?;
    eprintln!(
        "  {} samples x {} SNPs",
        bed.n_samples, bed.n_snps
    );

    // --- Load covariates with sample matching ---
    eprintln!("Loading covariates from: {}", cli.cov.display());
    let (cov_names, x) = load_covariates(&cli.cov, &bed.fam_records)?;
    let n = x.nrows();
    let d = x.ncols();
    eprintln!(
        "  {} samples x {} covariates: [{}]",
        n,
        d,
        cov_names.join(", ")
    );

    // --- Estimation subset ---
    if cli.est_bed.is_some() && cli.est_rate.is_some() {
        anyhow::bail!("Cannot specify both --est-bed and --est-rate");
    }

    let est_bed = if let Some(ref est_path) = cli.est_bed {
        eprintln!("Loading estimation BED file: {}", est_path.display());
        let eb = BedFile::open(est_path)?;
        if eb.n_samples != n {
            anyhow::bail!(
                "Sample count mismatch: est-bed has {} samples, covariate file has {}",
                eb.n_samples,
                n
            );
        }
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
    };

    eprintln!(
        "Running LFMM2: K={}, lambda={:.1e}, chunk_size={}, threads={}, oversampling={}, power_iter={}",
        config.k, config.lambda, config.chunk_size, config.n_workers,
        config.oversampling, config.n_power_iter,
    );

    let y_est = est_bed.as_ref().unwrap_or(&bed);
    let results = fit_lfmm2(y_est, &est_subset, &bed, &x, &config)?;

    eprintln!("GIF: {:.4}", results.gif);

    // --- Write output ---
    let results_path = format!("{}.tsv", cli.out);
    let summary_path = format!("{}.summary.txt", cli.out);

    eprintln!("Writing results to {}...", results_path);

    write_results(
        &results_path,
        &bed.bim_records,
        &results.effect_sizes,
        &results.t_stats,
        &results.p_values,
        &cov_names,
    )?;

    {
        let mut f = std::fs::File::create(&summary_path)
            .with_context(|| format!("Failed to create {}", summary_path))?;
        writeln!(f, "GIF: {:.6}", results.gif)?;
        writeln!(f, "n_samples: {}", results.u_hat.nrows())?;
        writeln!(f, "n_snps: {}", results.p_values.nrows())?;
        writeln!(f, "K: {}", results.u_hat.ncols())?;
        writeln!(f, "d: {}", results.effect_sizes.ncols())?;
        writeln!(f, "covariates: {}", cov_names.join(", "))?;
        writeln!(f, "lambda: {}", cli.lambda)?;
        writeln!(f, "threads: {}", cli.threads)?;
    }

    eprintln!("Done. Output: {}, {}", results_path, summary_path);

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
/// Returns (covariate_names, data_matrix) where rows are in .fam order.
fn load_covariates(
    path: &Path,
    fam: &[lfmm2::bed::FamRecord],
) -> Result<(Vec<String>, Array2<f64>)> {
    let delim = guess_delimiter(path);
    let delim_name = if delim == ',' { "CSV" } else { "TSV" };

    if cli_verbose() {
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

        if sample_data.contains_key(&sample_id) {
            anyhow::bail!(
                "Duplicate sample ID '{}' in covariate file (line {})",
                sample_id,
                i + 2,
            );
        }
        file_order.push(sample_id.clone());
        sample_data.insert(sample_id, vals);
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
    let mut x = Array2::<f64>::zeros((n, n_covs));
    let mut matched = 0;

    for (row, fam_rec) in fam.iter().enumerate() {
        let vals = sample_data.get(&fam_rec.iid).ok_or_else(|| {
            anyhow::anyhow!(
                "Sample '{}' (FID='{}') from .fam file not found in covariate file.\n\
                 .fam has {} samples, covariate file has {}.\n\
                 First few covariate sample IDs: [{}]",
                fam_rec.iid,
                fam_rec.fid,
                n,
                sample_data.len(),
                file_order.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
            )
        })?;
        for (j, &v) in vals.iter().enumerate() {
            x[(row, j)] = v;
        }
        matched += 1;
    }

    let extra = sample_data.len() - matched;
    if extra > 0 {
        eprintln!(
            "  Warning: {} sample(s) in covariate file not present in .fam file (ignored)",
            extra,
        );
    }

    eprintln!(
        "  Matched {} samples to .fam order",
        matched,
    );

    Ok((cov_names, x))
}

/// Write a single results TSV with SNP annotations and per-trait statistics.
///
/// Columns: chr, pos, snp_id, then for each trait: p_$TRAIT, beta_$TRAIT, t_$TRAIT
fn write_results(
    path: &str,
    bim: &[lfmm2::bed::BimRecord],
    effect_sizes: &Array2<f64>,
    t_stats: &Array2<f64>,
    p_values: &Array2<f64>,
    cov_names: &[String],
) -> Result<()> {
    let mut f =
        std::fs::File::create(path).with_context(|| format!("Failed to create {}", path))?;
    let p = bim.len();
    let d = cov_names.len();

    // Header: chr, pos, snp_id, then p/beta/t triples per trait
    let mut header = vec!["chr".to_string(), "pos".to_string(), "snp_id".to_string()];
    for name in cov_names {
        header.push(format!("p_{}", name));
        header.push(format!("beta_{}", name));
        header.push(format!("t_{}", name));
    }
    writeln!(f, "{}", header.join("\t"))?;

    // Data rows
    for i in 0..p {
        let rec = &bim[i];
        let mut vals = vec![rec.chrom.clone(), rec.pos.to_string(), rec.snp_id.clone()];
        for j in 0..d {
            vals.push(format!("{:.6e}", p_values[(i, j)]));
            vals.push(format!("{:.6e}", effect_sizes[(i, j)]));
            vals.push(format!("{:.6e}", t_stats[(i, j)]));
        }
        writeln!(f, "{}", vals.join("\t"))?;
    }
    Ok(())
}

/// Check if --verbose was passed. Used by helper functions that don't have
/// direct access to the Cli struct.
fn cli_verbose() -> bool {
    std::env::args().any(|a| a == "--verbose" || a == "-v")
}
