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

    // Generate X (n × d) — covariates
    let x = Array2::from_shape_fn((n, d), |_| rng.sample(normal));

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
pub fn write_r_comparison_script(dir: &Path, prefix: &str, k: usize) -> Result<()> {
    let script_path = dir.join("run_lea_comparison.R");
    let mut file = std::fs::File::create(script_path)?;

    write!(
        file,
        r#"#!/usr/bin/env Rscript
# Cross-reference: compare Rust LFMM2 against LEA (Bioconductor) lfmm2()
#
# Usage: cd testdata && Rscript run_lea_comparison.R

library(LEA)

prefix <- "{prefix}"

# Read genotype data (PLINK -> lfmm format)
# LEA expects space-separated genotype matrix, one row per sample
geno <- read.table(paste0(prefix, ".lfmm"), header=FALSE)
env <- read.table(paste0(prefix, "_covariates.txt"), header=TRUE)

cat("Genotype matrix:", nrow(geno), "samples x", ncol(geno), "SNPs\n")
cat("Running lfmm2 with K={k}...\n")

# Write temp files in LEA format
write.lfmm(as.matrix(geno), paste0(prefix, "_lea.lfmm"))

mod <- lfmm2(input = paste0(prefix, "_lea.lfmm"),
             env = as.matrix(env),
             K = {k})

pv <- lfmm2.test(object = mod,
                  input = paste0(prefix, "_lea.lfmm"),
                  env = as.matrix(env),
                  full = TRUE)

# Write p-values
write.table(pv$pvalues, paste0(prefix, "_lea_pvalues.tsv"),
            sep="\t", row.names=FALSE, col.names=TRUE)

# Write z-scores
write.table(pv$zscores, paste0(prefix, "_lea_zscores.tsv"),
            sep="\t", row.names=FALSE, col.names=TRUE)

cat("Done. Output written to:\n")
cat("  ", paste0(prefix, "_lea_pvalues.tsv"), "\n")
cat("  ", paste0(prefix, "_lea_zscores.tsv"), "\n")

# Quick comparison if Rust results exist
rust_pv_file <- paste0(prefix, "_rust_pvalues.tsv")
if (file.exists(rust_pv_file)) {{
    rust_pv <- read.table(rust_pv_file, header=TRUE, sep="\t")
    lea_pv <- pv$pvalues

    # Rank correlation
    for (j in 1:ncol(lea_pv)) {{
        rc <- cor(rank(-log10(rust_pv[,j])), rank(-log10(lea_pv[,j])), method="spearman")
        cat(sprintf("Covariate %d: Spearman rank correlation of -log10(p) = %.4f\n", j, rc))
    }}
}}
"#,
        prefix = prefix,
        k = k,
    )?;

    // Also write the .lfmm format file (space-separated genotype matrix)
    // This will be called from the test after simulation
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
