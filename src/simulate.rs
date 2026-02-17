use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Binomial, Normal, Uniform};
use std::io::Write;
use std::path::Path;

use crate::bed::{write_bed_file, write_bim, write_fam, BimRecord, FamRecord};

pub struct SimConfig {
    pub n_samples: usize,
    pub n_snps: usize,
    pub n_causal: usize,
    pub k: usize,
    pub d: usize,
    pub effect_size: f64,
    pub latent_scale: f64,
    pub noise_std: f64,
    /// Target r² between covariate columns (0.0 = independent).
    /// Column 0 is drawn iid N(0,1); columns j>0 are generated as
    /// r*X[:,0] + sqrt(1-r²)*Z_j with Z_j iid N(0,1), giving pairwise r² ≈ this value.
    pub covariate_r2: f64,
    pub seed: u64,
}

pub struct SimData {
    pub genotypes: Array2<u8>,
    pub x: Array2<f64>,
    pub u_true: Array2<f64>,
    pub b_true: Array2<f64>,
    pub causal_indices: Vec<usize>,
    pub allele_freqs: Array1<f64>,
}

/// Simulate GWAS data following the LFMM2 generative model.
///
/// Y = X B^T + U V^T + E
///
/// Genotypes are sampled from Binomial(2, p_ij) where the allele
/// frequency is shifted by latent structure and covariate effects.
pub fn simulate(config: &SimConfig) -> SimData {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let n = config.n_samples;
    let p = config.n_snps;
    let k = config.k;
    let d = config.d;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let af_dist = Uniform::new(0.05, 0.5);

    // Generate U_true (n × K)
    let u_true = Array2::from_shape_fn((n, k), |_| rng.sample(normal));

    // Generate V_true (p × K) with latent_scale
    let v_scale = config.latent_scale / (k as f64).sqrt();
    let v_true = Array2::from_shape_fn((p, k), |_| rng.sample(normal) * v_scale);

    // Generate X (n × d) — covariates with pairwise correlation ≈ covariate_r2
    // Column 0: iid N(0,1)
    // Columns j>0: r*X[:,0] + sqrt(1-r²)*Z_j giving pairwise r² ≈ covariate_r2
    let mut x = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        x[(i, 0)] = rng.sample(normal);
    }
    let r = config.covariate_r2.sqrt();
    let r_orth = (1.0 - config.covariate_r2).sqrt();
    for j in 1..d {
        for i in 0..n {
            x[(i, j)] = r * x[(i, 0)] + r_orth * rng.sample(normal);
        }
    }

    // Choose causal SNP indices
    let mut all_indices: Vec<usize> = (0..p).collect();
    all_indices.partial_shuffle(&mut rng, config.n_causal);
    let causal_indices: Vec<usize> = all_indices[..config.n_causal].to_vec();

    // Build B_true (p × d): only causal SNPs have nonzero effects
    let mut b_true = Array2::<f64>::zeros((p, d));
    for &idx in &causal_indices {
        for j in 0..d {
            let sign: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            b_true[(idx, j)] = sign * config.effect_size;
        }
    }

    // Compute latent + covariate contributions
    // UV^T: n × p
    // XB^T: n × p
    // We'll compute per-SNP to avoid allocating the full n×p matrix
    let mut allele_freqs = Array1::<f64>::zeros(p);
    let mut genotypes = Array2::<u8>::zeros((n, p));

    for j in 0..p {
        let base_af = rng.sample(af_dist);
        allele_freqs[j] = base_af;
        let base_logit = logit(base_af);

        for i in 0..n {
            // Latent contribution: (U V^T)[i,j] = sum_k U[i,k] * V[j,k]
            let mut uv = 0.0;
            for kk in 0..k {
                uv += u_true[(i, kk)] * v_true[(j, kk)];
            }

            // Covariate contribution: (X B^T)[i,j] = sum_d X[i,d] * B[j,d]
            let mut xb = 0.0;
            for dd in 0..d {
                xb += x[(i, dd)] * b_true[(j, dd)];
            }

            // Shifted allele frequency
            let shifted_logit = base_logit + uv + xb;
            let p_ij = inv_logit(shifted_logit).clamp(0.01, 0.99);

            // Sample genotype ~ Binomial(2, p_ij)
            let binom = Binomial::new(2, p_ij).unwrap();
            genotypes[(i, j)] = rng.sample(binom) as u8;
        }
    }

    SimData {
        genotypes,
        x,
        u_true,
        b_true,
        causal_indices,
        allele_freqs,
    }
}

/// Write PLINK triplet (.bed, .bim, .fam) for simulated data.
pub fn write_plink(dir: &Path, prefix: &str, sim: &SimData) -> Result<()> {
    let bed_path = dir.join(format!("{}.bed", prefix));
    let bim_path = dir.join(format!("{}.bim", prefix));
    let fam_path = dir.join(format!("{}.fam", prefix));

    let n = sim.genotypes.nrows();
    let p = sim.genotypes.ncols();

    // Write .bed
    write_bed_file(&bed_path, &sim.genotypes)?;

    // Write .bim
    let bim_records: Vec<BimRecord> = (0..p)
        .map(|j| BimRecord {
            chrom: "1".to_string(),
            snp_id: format!("snp_{}", j),
            cm: 0.0,
            pos: (j + 1) as u64,
            allele1: "A".to_string(),
            allele2: "G".to_string(),
        })
        .collect();
    write_bim(&bim_path, &bim_records)?;

    // Write .fam
    let fam_records: Vec<FamRecord> = (0..n)
        .map(|i| FamRecord {
            fid: format!("FAM{}", i),
            iid: format!("IND{}", i),
            father: "0".to_string(),
            mother: "0".to_string(),
            sex: 0,
            pheno: "-9".to_string(),
        })
        .collect();
    write_fam(&fam_path, &fam_records)?;

    Ok(())
}

/// Write covariate matrix as tab-separated file with header.
pub fn write_covariates(path: &Path, x: &Array2<f64>) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let d = x.ncols();

    // Header
    let header: Vec<String> = (0..d).map(|j| format!("env_{}", j)).collect();
    writeln!(file, "{}", header.join("\t"))?;

    // Data rows
    for i in 0..x.nrows() {
        let vals: Vec<String> = (0..d).map(|j| format!("{:.6}", x[(i, j)])).collect();
        writeln!(file, "{}", vals.join("\t"))?;
    }
    Ok(())
}

/// Write ground truth for causal/null status and true effect sizes.
pub fn write_ground_truth(path: &Path, sim: &SimData) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let p = sim.b_true.nrows();
    let d = sim.b_true.ncols();

    // Header
    let mut header = vec!["snp_index".to_string(), "is_causal".to_string()];
    for j in 0..d {
        header.push(format!("beta_true_{}", j));
    }
    writeln!(file, "{}", header.join("\t"))?;

    // Data
    for j in 0..p {
        let is_causal = sim.causal_indices.contains(&j);
        let mut vals = vec![
            j.to_string(),
            if is_causal {
                "1".to_string()
            } else {
                "0".to_string()
            },
        ];
        for dd in 0..d {
            vals.push(format!("{:.6}", sim.b_true[(j, dd)]));
        }
        writeln!(file, "{}", vals.join("\t"))?;
    }
    Ok(())
}

/// Write true latent U matrix.
pub fn write_latent_u(path: &Path, u: &Array2<f64>) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let k = u.ncols();

    let header: Vec<String> = (0..k).map(|j| format!("U_{}", j)).collect();
    writeln!(file, "{}", header.join("\t"))?;

    for i in 0..u.nrows() {
        let vals: Vec<String> = (0..k).map(|j| format!("{:.6}", u[(i, j)])).collect();
        writeln!(file, "{}", vals.join("\t"))?;
    }
    Ok(())
}

/// Write R comparison script for LEA cross-reference.
///
/// Generates QQ plots (LEA vs expected, Rust vs expected, Rust vs LEA),
/// and computes power/recall, FDR, TDR, and F1 at fixed FDR thresholds.
pub fn write_r_comparison_script(dir: &Path, prefix: &str, k: usize) -> Result<()> {
    let script_path = dir.join("run_lea_comparison.R");
    let mut file = std::fs::File::create(script_path)?;

    write!(
        file,
        r#"#!/usr/bin/env Rscript
# Comprehensive validation: Rust LFMM2 vs LEA (Bioconductor) lfmm2()
#
# Produces:
#   - QQ plots (LEA vs expected, Rust vs expected, Rust vs LEA)
#   - Power/recall, FDR, TDR, F1 at fixed FDR thresholds (1%, 5%, 10%)
#   - Spearman rank correlations
#
# Usage: cd testdata && Rscript run_lea_comparison.R

library(LEA)

prefix <- "{prefix}"
K <- {k}

# ============================================================
# 1. Load data
# ============================================================
cat("Loading data...\n")
geno  <- read.table(paste0(prefix, ".lfmm"), header = FALSE)
env   <- read.table(paste0(prefix, "_covariates.txt"), header = TRUE)
truth <- read.table(paste0(prefix, "_truth.tsv"), header = TRUE, sep = "\t")

is_causal <- truth$is_causal == 1
n_samples <- nrow(geno)
n_snps    <- ncol(geno)
n_causal  <- sum(is_causal)
d         <- ncol(env)

cat(sprintf("  %d samples, %d SNPs, %d causal, %d covariates\n",
            n_samples, n_snps, n_causal, d))

# ============================================================
# 2. Run LEA lfmm2
# ============================================================
cat(sprintf("Running LEA lfmm2 with K = %d ...\n", K))
write.lfmm(as.matrix(geno), paste0(prefix, "_lea.lfmm"))

mod <- lfmm2(input = paste0(prefix, "_lea.lfmm"),
             env   = as.matrix(env),
             K     = K)

pv <- lfmm2.test(object = mod,
                  input  = paste0(prefix, "_lea.lfmm"),
                  env    = as.matrix(env),
                  full   = FALSE)

lea_pv <- as.data.frame(t(pv$pvalues))
colnames(lea_pv) <- paste0("p_", seq_len(d) - 1)
write.table(lea_pv, paste0(prefix, "_lea_pvalues.tsv"),
            sep = "\t", row.names = FALSE, col.names = TRUE)

lea_zs <- as.data.frame(t(pv$zscores))
colnames(lea_zs) <- paste0("z_", seq_len(d) - 1)
write.table(lea_zs, paste0(prefix, "_lea_zscores.tsv"),
            sep = "\t", row.names = FALSE, col.names = TRUE)

cat("LEA results written.\n")

# ============================================================
# 3. Load Rust results
# ============================================================
rust_pv_file <- paste0(prefix, "_rust_pvalues.tsv")
if (!file.exists(rust_pv_file)) {{
    stop(paste("Rust p-values not found:", rust_pv_file,
               "\nRun the Rust integration test first."))
}}
rust_pv <- read.table(rust_pv_file, header = TRUE, sep = "\t")
cat("Rust results loaded.\n")

# ============================================================
# 4. QQ plots
# ============================================================
qq_expected <- function(pvals, main, col = "black") {{
    pvals <- pvals[!is.na(pvals)]
    n     <- length(pvals)
    expected <- -log10(rev(1:n / (n + 1)))
    observed <- sort(-log10(pvals))
    lim <- max(expected, observed)
    plot(expected, observed, pch = 20, cex = 0.3, col = col,
         xlab = expression(-log[10](p[expected])),
         ylab = expression(-log[10](p[observed])),
         main = main, xlim = c(0, lim), ylim = c(0, lim))
    abline(0, 1, col = "red", lty = 2)
    # GIF (genomic inflation factor) from median chi-sq
    chisq <- qchisq(pvals, df = 1, lower.tail = FALSE)
    gif <- median(chisq, na.rm = TRUE) / qchisq(0.5, df = 1)
    legend("topleft", legend = sprintf("GIF = %.3f", gif), bty = "n", cex = 0.9)
}}

qq_vs <- function(pvals_x, pvals_y, xlab_text, ylab_text, main) {{
    x <- -log10(sort(pvals_x))
    y <- -log10(sort(pvals_y))
    lim <- max(c(x, y), na.rm = TRUE)
    plot(x, y, pch = 20, cex = 0.3,
         xlab = xlab_text, ylab = ylab_text,
         main = main, xlim = c(0, lim), ylim = c(0, lim))
    abline(0, 1, col = "red", lty = 2)
    rc <- cor(x, y, method = "spearman")
    legend("topleft", legend = sprintf("rho = %.4f", rc), bty = "n", cex = 0.9)
}}

pdf_file <- paste0(prefix, "_validation_plots.pdf")
pdf(pdf_file, width = 12, height = 4 * d)
par(mfrow = c(d, 3), mar = c(4, 4, 3, 1))

for (j in 1:d) {{
    qq_expected(lea_pv[, j],
                paste0("LEA QQ - Covariate ", j),
                col = "steelblue")
    qq_expected(rust_pv[, j],
                paste0("Rust QQ - Covariate ", j),
                col = "darkorange")
    qq_vs(lea_pv[, j], rust_pv[, j],
          expression(-log[10](p[LEA])),
          expression(-log[10](p[Rust])),
          paste0("Rust vs LEA - Covariate ", j))
}}
dev.off()
cat(sprintf("QQ plots saved to %s\n", pdf_file))

# ============================================================
# 5. Discovery metrics at fixed FDR thresholds
# ============================================================
compute_metrics <- function(pvals, is_causal, fdr_cut) {{
    qvals      <- p.adjust(pvals, method = "BH")
    discovered <- qvals < fdr_cut
    tp  <- sum( discovered &  is_causal)
    fp  <- sum( discovered & !is_causal)
    fn_ <- sum(!discovered &  is_causal)

    n_disc <- tp + fp
    power  <- if (sum(is_causal) > 0) tp / sum(is_causal) else NA   # recall
    fdr_obs <- if (n_disc > 0)  fp / n_disc            else 0       # FP / (TP+FP)
    tdr     <- if (n_disc > 0)  tp / n_disc            else NA      # precision = 1-FDR
    f1      <- if (!is.na(power) && !is.na(tdr) && (power + tdr) > 0)
                   2 * power * tdr / (power + tdr) else 0

    data.frame(fdr_cut = fdr_cut, n_disc = n_disc,
               tp = tp, fp = fp, fn_ = fn_,
               power = power, fdr_obs = fdr_obs, tdr = tdr, f1 = f1)
}}

fdr_thresholds <- c(0.01, 0.05, 0.10)

cat("\n============================================================\n")
cat("Power vs. FDR at fixed BH thresholds\n")
cat("  Power  = TP / (TP + FN)           (recall)\n")
cat("  TDR    = TP / (TP + FP)           (precision = 1 - FDR)\n")
cat("  F1     = 2 * Power * TDR / (Power + TDR)\n")
cat("============================================================\n")

metrics_all <- data.frame()

for (j in 1:d) {{
    cat(sprintf("\n--- Covariate %d ---\n", j))

    for (method_name in c("Rust", "LEA")) {{
        pvals <- if (method_name == "Rust") rust_pv[, j] else lea_pv[, j]

        cat(sprintf("\n  %s:\n", method_name))
        cat(sprintf("  %-10s %7s %5s %5s %5s %8s %8s %8s %8s\n",
                    "FDR_cut", "n_disc", "TP", "FP", "FN", "Power", "FDR_obs", "TDR", "F1"))

        for (thr in fdr_thresholds) {{
            m <- compute_metrics(pvals, is_causal, thr)
            cat(sprintf("  %-10.2f %7d %5d %5d %5d %8.4f %8.4f %8.4f %8.4f\n",
                        m$fdr_cut, m$n_disc, m$tp, m$fp, m$fn_,
                        m$power, m$fdr_obs, m$tdr, m$f1))
            m$method    <- method_name
            m$covariate <- j
            metrics_all <- rbind(metrics_all, m)
        }}
    }}
}}

# Write metrics table
write.table(metrics_all,
            paste0(prefix, "_validation_metrics.tsv"),
            sep = "\t", row.names = FALSE, col.names = TRUE)

# ============================================================
# 6. Rank correlations
# ============================================================
cat("\n============================================================\n")
cat("Spearman rank correlations of -log10(p): Rust vs LEA\n")
cat("============================================================\n")
for (j in 1:d) {{
    rc <- cor(rank(-log10(rust_pv[, j])),
              rank(-log10(lea_pv[, j])),
              method = "spearman")
    cat(sprintf("  Covariate %d: rho = %.4f\n", j, rc))
}}

cat("Sanity check: all(rust_pv == lea_pv):")
print(table(rust_pv == lea_pv))
cat("\nhead(rust_pv):\n")
print(head(rust_pv))
cat("head(lea_pv):\n")
print(head(lea_pv))

cat(sprintf("\nAll outputs written with prefix '%s'.\n", prefix))
"#,
        prefix = prefix,
        k = k,
    )?;

    Ok(())
}

/// Write genotype matrix in .lfmm format (space-separated, one row per sample)
pub fn write_lfmm_format(path: &Path, genotypes: &Array2<u8>) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let n = genotypes.nrows();
    let p = genotypes.ncols();

    for i in 0..n {
        let vals: Vec<String> = (0..p).map(|j| genotypes[(i, j)].to_string()).collect();
        writeln!(file, "{}", vals.join(" "))?;
    }
    Ok(())
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn inv_logit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
