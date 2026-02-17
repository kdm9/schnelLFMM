use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use lfmm2::bed::BedFile;
use lfmm2::{fit_lfmm2, Lfmm2Config};

#[derive(Parser)]
#[command(name = "lfmm2", about = "Latent Factor Mixed Model v2 — GWAS with latent confounders")]
struct Cli {
    /// PLINK .bed file (with .bim/.fam)
    #[arg(short = 'b', long)]
    bed: PathBuf,

    /// Covariate file (TSV with header, n rows × d cols)
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

    /// Worker threads (0 = sequential)
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

fn main() -> Result<()> {
    // Prevent BLAS from spawning its own threads (contention with our workers)
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let cli = Cli::parse();

    if cli.verbose {
        eprintln!("Loading covariates from {}...", cli.cov.display());
    }
    let x = load_covariates(&cli.cov)?;
    let n = x.nrows();
    let d = x.ncols();
    if cli.verbose {
        eprintln!("  {} samples × {} covariates", n, d);
    }

    if cli.verbose {
        eprintln!("Opening BED file {}...", cli.bed.display());
    }
    let bed = BedFile::open(&cli.bed)?;
    if bed.n_samples != n {
        anyhow::bail!(
            "Sample count mismatch: BED has {} samples, covariate file has {}",
            bed.n_samples,
            n
        );
    }
    if cli.verbose {
        eprintln!("  {} samples × {} SNPs", bed.n_samples, bed.n_snps);
    }

    let est_bed = if let Some(ref est_path) = cli.est_bed {
        if cli.verbose {
            eprintln!("Opening estimation BED file {}...", est_path.display());
        }
        let eb = BedFile::open(est_path)?;
        if eb.n_samples != n {
            anyhow::bail!(
                "Sample count mismatch: est-bed has {} samples, covariate file has {}",
                eb.n_samples,
                n
            );
        }
        if cli.verbose {
            eprintln!("  {} samples × {} SNPs", eb.n_samples, eb.n_snps);
        }
        Some(eb)
    } else {
        None
    };

    let config = Lfmm2Config {
        k: cli.k,
        lambda: cli.lambda,
        chunk_size: cli.chunk_size,
        oversampling: cli.oversampling,
        n_power_iter: cli.power_iter,
        seed: cli.seed,
        n_workers: cli.threads,
    };

    if cli.verbose {
        eprintln!(
            "Running LFMM2: K={}, lambda={}, chunk_size={}, threads={}, oversampling={}, power_iter={}",
            config.k, config.lambda, config.chunk_size, config.n_workers,
            config.oversampling, config.n_power_iter,
        );
    }

    let y_est = est_bed.as_ref().unwrap_or(&bed);
    let results = fit_lfmm2(y_est, &bed, &x, &config)?;

    if cli.verbose {
        eprintln!("GIF: {:.4}", results.gif);
        eprintln!("Writing output files with prefix '{}'...", cli.out);
    }

    write_tsv(
        &format!("{}.effect_sizes.tsv", cli.out),
        &results.effect_sizes,
        "beta",
    )?;
    write_tsv(
        &format!("{}.t_stats.tsv", cli.out),
        &results.t_stats,
        "t",
    )?;
    write_tsv(
        &format!("{}.p_values.tsv", cli.out),
        &results.p_values,
        "p",
    )?;

    // Write summary
    {
        let path = format!("{}.summary.txt", cli.out);
        let mut f = std::fs::File::create(&path)
            .with_context(|| format!("Failed to create {}", path))?;
        writeln!(f, "GIF: {:.6}", results.gif)?;
        writeln!(f, "n_samples: {}", results.u_hat.nrows())?;
        writeln!(f, "n_snps: {}", results.p_values.nrows())?;
        writeln!(f, "K: {}", results.u_hat.ncols())?;
        writeln!(f, "d: {}", results.effect_sizes.ncols())?;
        writeln!(f, "lambda: {}", cli.lambda)?;
        writeln!(f, "threads: {}", cli.threads)?;
    }

    if cli.verbose {
        eprintln!("Done.");
    }

    Ok(())
}

/// Load a TSV covariate file with a header row into an Array2<f64>.
fn load_covariates(path: &Path) -> Result<Array2<f64>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open covariate file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    let _header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("Covariate file is empty"))??;

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line?;
        let vals: Vec<f64> = line
            .split('\t')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .with_context(|| format!("Failed to parse value '{}' on line {}", s.trim(), i + 2))
            })
            .collect::<Result<Vec<_>>>()?;
        if !rows.is_empty() && vals.len() != rows[0].len() {
            anyhow::bail!(
                "Line {} has {} columns, expected {}",
                i + 2,
                vals.len(),
                rows[0].len()
            );
        }
        rows.push(vals);
    }

    if rows.is_empty() {
        anyhow::bail!("Covariate file has no data rows");
    }

    let n = rows.len();
    let d = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n, d), flat)?)
}

/// Write an Array2<f64> as a TSV file with a header row.
fn write_tsv(path: &str, data: &Array2<f64>, col_prefix: &str) -> Result<()> {
    let mut f =
        std::fs::File::create(path).with_context(|| format!("Failed to create {}", path))?;
    let d = data.ncols();

    // Header
    let header: Vec<String> = (0..d).map(|j| format!("{}_{}", col_prefix, j)).collect();
    writeln!(f, "{}", header.join("\t"))?;

    // Data
    for i in 0..data.nrows() {
        let vals: Vec<String> = (0..d).map(|j| format!("{:.6e}", data[(i, j)])).collect();
        writeln!(f, "{}", vals.join("\t"))?;
    }
    Ok(())
}
