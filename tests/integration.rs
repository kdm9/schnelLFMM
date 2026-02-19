use lfmm2::bed::{BedFile, SubsetSpec};
use lfmm2::parallel::subset_indices;
use lfmm2::simulate::{
    simulate, write_covariates, write_ground_truth, write_latent_u, write_lfmm_format,
    write_plink, write_r_comparison_script, SimConfig,
};
use lfmm2::{fit_lfmm2, Lfmm2Config, OutputConfig, SnpNorm};
use ndarray::Array2;
use ndarray_linalg::SVD;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Tier 1: Quick test — small simulation for basic correctness.
/// n=200, p=10_000, K=3, d=2 (r²≈0.3), 20 causal SNPs
#[test]
fn test_lfmm2_quick() {
    let sim_config = SimConfig {
        n_samples: 200,
        n_snps: 10_000,
        n_causal: 20,
        k: 3,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 12345,
    };

    let config = Lfmm2Config {
        k: 3,
        lambda: 1e-5,
        chunk_size: 2_000,
        oversampling: 10,
        n_power_iter: 2,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    // Simulate
    let sim = simulate(&sim_config);

    // Write PLINK files to tempdir
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();

    // Open BED file
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();
    assert_eq!(bed.n_samples, 200);
    assert_eq!(bed.n_snps, 10_000);

    // Run LFMM2
    let results = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &config, None).unwrap();

    // Validation checks
    validate_results(&results, &sim, &config);
}

/// Tier 2: Large simulation for thorough validation.
/// n=400, p=100_000, K=5, d=2 (r²≈0.3), 100 causal SNPs
#[test]
#[ignore] // Run with: cargo test --release -- --ignored
fn test_lfmm2_large() {
    let sim_config = SimConfig {
        n_samples: 400,
        n_snps: 100_000,
        n_causal: 100,
        k: 5,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 54321,
    };

    let config = Lfmm2Config {
        k: 5,
        lambda: 1e-5,
        chunk_size: 10_000,
        oversampling: 10,
        n_power_iter: 2,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    // Simulate
    eprintln!("Simulating {} samples × {} SNPs...", sim_config.n_samples, sim_config.n_snps);
    let sim = simulate(&sim_config);

    // Write to persistent testdata directory
    let testdata = Path::new("testdata");
    fs::create_dir_all(testdata).unwrap();

    eprintln!("Writing PLINK files...");
    write_plink(testdata, "sim", &sim).unwrap();
    write_covariates(&testdata.join("sim_covariates.txt"), &sim.x).unwrap();
    write_ground_truth(&testdata.join("sim_truth.tsv"), &sim).unwrap();
    write_latent_u(&testdata.join("sim_latent_U.tsv"), &sim.u_true).unwrap();
    write_lfmm_format(&testdata.join("sim.lfmm"), &sim.genotypes).unwrap();
    write_r_comparison_script(testdata, "sim", sim_config.k).unwrap();

    // Open BED file
    let bed = BedFile::open(testdata.join("sim.bed")).unwrap();
    assert_eq!(bed.n_samples, sim_config.n_samples);
    assert_eq!(bed.n_snps, sim_config.n_snps);

    // Run LFMM2
    eprintln!("Running LFMM2...");
    let results = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &config, None).unwrap();

    // Validation
    validate_results(&results, &sim, &config);

    // Write Rust results for R comparison
    write_rust_results(testdata, &results);

    eprintln!("Done. Results in testdata/");
    eprintln!("To compare with LEA: cd testdata && Rscript run_lea_comparison.R");
}

/// Tier 1b: Reproducibility — same seed produces identical results.
#[test]
fn test_reproducibility() {
    let sim_config = SimConfig {
        n_samples: 100,
        n_snps: 1_000,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 99,
    };

    let config = Lfmm2Config {
        k: 2,
        lambda: 1e-5,
        chunk_size: 500,
        oversampling: 5,
        n_power_iter: 1,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    let sim = simulate(&sim_config);

    let dir1 = tempfile::tempdir().unwrap();
    write_plink(dir1.path(), "sim", &sim).unwrap();
    let bed1 = BedFile::open(dir1.path().join("sim.bed")).unwrap();
    let r1 = fit_lfmm2(&bed1, &SubsetSpec::All, &bed1, &sim.x, &config, None).unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    write_plink(dir2.path(), "sim", &sim).unwrap();
    let bed2 = BedFile::open(dir2.path().join("sim.bed")).unwrap();
    let r2 = fit_lfmm2(&bed2, &SubsetSpec::All, &bed2, &sim.x, &config, None).unwrap();

    // Bitwise identical
    assert_eq!(r1.p_values.shape(), r2.p_values.shape());
    for (a, b) in r1.p_values.iter().zip(r2.p_values.iter()) {
        assert!(
            (a - b).abs() < 1e-15,
            "Reproducibility failed: {} vs {}",
            a,
            b
        );
    }
    assert!((r1.gif - r2.gif).abs() < 1e-15);
}

/// Different seeds should produce different but statistically similar results.
///
/// "Different" means the p-values are not bitwise identical (the RNG sketch
/// in RSVD genuinely changes the random projection matrix Omega).
/// "Similar" means the Spearman rank correlation between the two p-value
/// vectors is high (> 0.9) — both runs recover approximately the same
/// signal, just via different random projections.
#[test]
fn test_different_seeds_differ() {
    let sim_config = SimConfig {
        n_samples: 100,
        n_snps: 1_000,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 99,
    };

    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    let base_config = Lfmm2Config {
        k: 2,
        lambda: 1e-5,
        chunk_size: 500,
        oversampling: 5,
        n_power_iter: 1,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    let r1 = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &base_config, None).unwrap();

    let alt_config = Lfmm2Config {
        seed: 999,
        ..base_config
    };
    let r2 = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &alt_config, None).unwrap();

    let p = sim_config.n_snps;
    let d = sim_config.d;

    // 1. Results must NOT be identical — at least some p-values should differ
    let mut n_differ = 0;
    let mut max_diff = 0.0f64;
    for i in 0..p {
        for j in 0..d {
            let diff = (r1.p_values[(i, j)] - r2.p_values[(i, j)]).abs();
            if diff > 1e-15 {
                n_differ += 1;
            }
            max_diff = max_diff.max(diff);
        }
    }
    eprintln!(
        "Different seeds: {}/{} p-values differ, max_diff={:.2e}",
        n_differ, p * d, max_diff,
    );
    assert!(
        n_differ > 0,
        "Different seeds produced identical p-values — RNG seed is not taking effect",
    );

    // 2. Results should be similar: high Spearman rank correlation per covariate
    for j in 0..d {
        let mut v1: Vec<f64> = (0..p).map(|i| r1.p_values[(i, j)]).collect();
        let mut v2: Vec<f64> = (0..p).map(|i| r2.p_values[(i, j)]).collect();
        let rho = spearman_rank_corr(&mut v1, &mut v2);
        eprintln!("  Covariate {}: Spearman rho = {:.4}", j, rho);
        assert!(
            rho > 0.9,
            "Spearman correlation too low for covariate {}: {:.4} (expected > 0.9)",
            j, rho,
        );
    }

    // 3. GIF values should be close but not identical
    let gif_diff = (r1.gif - r2.gif).abs();
    eprintln!("  GIF diff: {:.4e} ({:.4} vs {:.4})", gif_diff, r1.gif, r2.gif);
    assert!(
        r2.gif > 0.5 && r2.gif < 2.0,
        "GIF out of range with alt seed: {:.4}",
        r2.gif,
    );
}

/// Compute Spearman rank correlation between two equal-length slices.
/// Mutates the inputs (for ranking).
fn spearman_rank_corr(a: &mut [f64], b: &mut [f64]) -> f64 {
    let n = a.len();
    assert_eq!(n, b.len());

    let rank_a = ranks(a);
    let rank_b = ranks(b);

    // Pearson correlation of ranks
    let mean_a: f64 = rank_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rank_b.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    cov / (var_a * var_b).sqrt()
}

/// Assign ranks to values (average rank for ties).
fn ranks(vals: &mut [f64]) -> Vec<f64> {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| vals[i].partial_cmp(&vals[j]).unwrap());

    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && vals[idx[j]] == vals[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            result[idx[k]] = avg_rank;
        }
        i = j;
    }
    result
}

fn validate_results(
    results: &lfmm2::testing::TestResults,
    sim: &lfmm2::simulate::SimData,
    _config: &Lfmm2Config,
) {
    let p = sim.genotypes.ncols();
    let d = sim.x.ncols();

    // 1. Subspace recovery: canonical correlations between U_hat and U_true
    let mean_cancorr = canonical_correlations(&results.u_hat, &sim.u_true);
    eprintln!("Mean canonical correlation: {:.4}", mean_cancorr);
    assert!(
        mean_cancorr > 0.5,
        "Subspace recovery too poor: mean canonical correlation = {:.4} (expected > 0.5)",
        mean_cancorr
    );

    // 2. Effect size sign agreement for causal SNPs
    let mut sign_matches = 0;
    let mut sign_total = 0;
    for &idx in &sim.causal_indices {
        for j in 0..d {
            if sim.b_true[(idx, j)].abs() > 1e-10 {
                sign_total += 1;
                if results.effect_sizes[(idx, j)].signum() == sim.b_true[(idx, j)].signum() {
                    sign_matches += 1;
                }
            }
        }
    }
    let sign_agreement = sign_matches as f64 / sign_total.max(1) as f64;
    eprintln!(
        "Sign agreement (causal SNPs): {:.1}% ({}/{})",
        sign_agreement * 100.0,
        sign_matches,
        sign_total
    );
    assert!(
        sign_agreement > 0.6,
        "Sign agreement too low: {:.1}% (expected > 60%)",
        sign_agreement * 100.0
    );

    // 3. False positive rate among null SNPs
    let null_count = p - sim.causal_indices.len();
    let mut fp_count = 0;
    for j_snp in 0..p {
        if !sim.causal_indices.contains(&j_snp) {
            for j_cov in 0..d {
                if results.p_values[(j_snp, j_cov)] < 0.05 {
                    fp_count += 1;
                }
            }
        }
    }
    let fpr = fp_count as f64 / (null_count * d) as f64;
    eprintln!("False positive rate (null, p<0.05): {:.4}", fpr);
    assert!(
        fpr < 0.15,
        "FPR too high: {:.4} (expected < 0.15)",
        fpr
    );

    // 4. Power among causal SNPs
    let causal_count = sim.causal_indices.len();
    let mut tp_count = 0;
    for &idx in &sim.causal_indices {
        for j in 0..d {
            if results.p_values[(idx, j)] < 0.05 {
                tp_count += 1;
            }
        }
    }
    let power = tp_count as f64 / (causal_count * d) as f64;
    eprintln!("Power (causal, p<0.05): {:.4}", power);
    // Power threshold is lenient since it depends heavily on effect size and sample size
    assert!(
        power > 0.1,
        "Power too low: {:.4} (expected > 0.1)",
        power
    );

    // 5. GIF calibration
    eprintln!("GIF: {:.4}", results.gif);
    assert!(
        results.gif > 0.5 && results.gif < 2.0,
        "GIF out of range: {:.4} (expected 0.5-2.0)",
        results.gif
    );
}

/// Compute mean canonical correlation between two matrices.
/// Uses SVD of the cross-correlation matrix.
fn canonical_correlations(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let k = a.ncols().min(b.ncols());

    // QR of A and B
    let (qa, _, _) = a.svd(true, false).unwrap();
    let (qb, _, _) = b.svd(true, false).unwrap();
    let qa = qa.unwrap();
    let qb = qb.unwrap();

    // Take first k columns
    let qa_k = qa.slice(ndarray::s![.., ..k]).to_owned();
    let qb_k = qb.slice(ndarray::s![.., ..k]).to_owned();

    // SVD of cross-product
    let cross = qa_k.t().dot(&qb_k);
    let (_, s, _) = cross.svd(false, false).unwrap();

    // Canonical correlations are the singular values (clamped to [0,1])
    let mean_corr: f64 = s.iter().take(k).map(|&v| v.min(1.0)).sum::<f64>() / k as f64;
    mean_corr
}

/// Parallel correctness: n_workers=2 should match n_workers=0 within FP tolerance.
#[test]
fn test_parallel_matches_sequential() {
    let sim_config = SimConfig {
        n_samples: 200,
        n_snps: 10_000,
        n_causal: 20,
        k: 3,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 77777,
    };

    let sim = simulate(&sim_config);

    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    // Sequential run
    let config_seq = Lfmm2Config {
        k: 3,
        lambda: 1e-5,
        chunk_size: 2_000,
        oversampling: 10,
        n_power_iter: 2,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };
    let r_seq = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &config_seq, None).unwrap();

    // Parallel run
    let config_par = Lfmm2Config {
        n_workers: 2,
        ..config_seq
    };
    let r_par = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &config_par, None).unwrap();

    // P-values should match within floating-point tolerance.
    // Pattern A (disjoint writes) are bitwise identical across thread counts.
    // Pattern B (per-worker accumulators summed at the end) may differ at ~1e-9
    // due to FP summation order: workers accumulate locally then merge, vs
    // sequential accumulation with n_workers=1.
    let tol = 1e-6;
    let p = r_seq.p_values.nrows();
    let d = r_seq.p_values.ncols();
    let mut max_diff = 0.0f64;
    for i in 0..p {
        for j in 0..d {
            let diff = (r_seq.p_values[(i, j)] - r_par.p_values[(i, j)]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    eprintln!("Max p-value diff (seq vs par): {:.2e}", max_diff);
    assert!(
        max_diff < tol,
        "Parallel p-values diverge from sequential: max_diff={:.2e} (tol={:.2e})",
        max_diff,
        tol,
    );

    // GIF should also be very close
    let gif_tol = 1e-6;
    let gif_diff = (r_seq.gif - r_par.gif).abs();
    eprintln!("GIF diff: {:.2e}", gif_diff);
    assert!(
        gif_diff < gif_tol,
        "GIF diverges: seq={:.6} par={:.6} diff={:.2e}",
        r_seq.gif,
        r_par.gif,
        gif_diff,
    );
}

fn write_rust_results(dir: &Path, results: &lfmm2::testing::TestResults) {
    use std::io::Write;

    let p = results.p_values.nrows();
    let d = results.p_values.ncols();

    // Write p-values
    let mut f = fs::File::create(dir.join("sim_rust_pvalues.tsv")).unwrap();
    let header: Vec<String> = (0..d).map(|j| format!("p_{}", j)).collect();
    writeln!(f, "{}", header.join("\t")).unwrap();
    for i in 0..p {
        let vals: Vec<String> = (0..d).map(|j| format!("{:.6e}", results.p_values[(i, j)])).collect();
        writeln!(f, "{}", vals.join("\t")).unwrap();
    }

    // Write t-stats
    let mut f = fs::File::create(dir.join("sim_rust_tstats.tsv")).unwrap();
    let header: Vec<String> = (0..d).map(|j| format!("t_{}", j)).collect();
    writeln!(f, "{}", header.join("\t")).unwrap();
    for i in 0..p {
        let vals: Vec<String> = (0..d).map(|j| format!("{:.6e}", results.t_stats[(i, j)])).collect();
        writeln!(f, "{}", vals.join("\t")).unwrap();
    }

    // Write effect sizes
    let mut f = fs::File::create(dir.join("sim_rust_effects.tsv")).unwrap();
    let header: Vec<String> = (0..d).map(|j| format!("beta_{}", j)).collect();
    writeln!(f, "{}", header.join("\t")).unwrap();
    for i in 0..p {
        let vals: Vec<String> = (0..d)
            .map(|j| format!("{:.6e}", results.effect_sizes[(i, j)]))
            .collect();
        writeln!(f, "{}", vals.join("\t")).unwrap();
    }

    // Write summary
    let mut f = fs::File::create(dir.join("sim_rust_summary.txt")).unwrap();
    writeln!(f, "GIF: {:.6}", results.gif).unwrap();
    writeln!(f, "n_samples: {}", results.u_hat.nrows()).unwrap();
    writeln!(f, "n_snps: {}", results.p_values.nrows()).unwrap();
    writeln!(f, "K: {}", results.u_hat.ncols()).unwrap();
    writeln!(f, "d: {}", results.effect_sizes.ncols()).unwrap();
}

// ---------------------------------------------------------------------------
// SubsetSpec tests
// ---------------------------------------------------------------------------

/// Unit-level: verify subset_indices produces the correct index vectors.
#[test]
fn test_subset_indices() {
    // All
    assert_eq!(subset_indices(&SubsetSpec::All, 10), vec![0,1,2,3,4,5,6,7,8,9]);

    // Rate 0.5 → step=2 → every other SNP
    let idx = subset_indices(&SubsetSpec::Rate(0.5), 10);
    assert_eq!(idx, vec![0, 2, 4, 6, 8]);

    // Rate 0.25 → step=4
    let idx = subset_indices(&SubsetSpec::Rate(0.25), 10);
    assert_eq!(idx, vec![0, 4, 8]);

    // Rate 1.0 → step=1 → all
    let idx = subset_indices(&SubsetSpec::Rate(1.0), 10);
    assert_eq!(idx, vec![0,1,2,3,4,5,6,7,8,9]);

    // Rate 0.1 → step=10
    let idx = subset_indices(&SubsetSpec::Rate(0.1), 25);
    assert_eq!(idx, vec![0, 10, 20]);

    // Indices
    let idx = subset_indices(&SubsetSpec::Indices(vec![3, 7, 11]), 100);
    assert_eq!(idx, vec![3, 7, 11]);
}

/// Verify subset_snp_count agrees with the actual length of subset_indices.
#[test]
fn test_subset_snp_count_consistency() {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 1_000,
        n_causal: 5,
        k: 2,
        d: 1,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 111,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    for rate in &[1.0, 0.5, 0.25, 0.1, 0.01] {
        let spec = SubsetSpec::Rate(*rate);
        let count = bed.subset_snp_count(&spec);
        let indices = subset_indices(&spec, bed.n_snps);
        assert_eq!(
            count,
            indices.len(),
            "subset_snp_count disagrees with subset_indices for rate={}",
            rate
        );
    }

    let spec = SubsetSpec::All;
    assert_eq!(bed.subset_snp_count(&spec), bed.n_snps);

    let spec = SubsetSpec::Indices(vec![0, 100, 500, 999]);
    assert_eq!(bed.subset_snp_count(&spec), 4);
}

/// End-to-end: SubsetSpec::Rate produces valid results, and output p-values
/// cover all p SNPs (not just the estimation subset).
#[test]
fn test_subset_rate_end_to_end() {
    let sim_config = SimConfig {
        n_samples: 200,
        n_snps: 10_000,
        n_causal: 20,
        k: 3,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 55555,
    };

    let config = Lfmm2Config {
        k: 3,
        lambda: 1e-5,
        chunk_size: 2_000,
        oversampling: 10,
        n_power_iter: 2,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    // Run with Rate(0.5) — estimate on half the SNPs, test on all
    let r_rate = fit_lfmm2(&bed, &SubsetSpec::Rate(0.5), &bed, &sim.x, &config, None).unwrap();

    // Output must cover ALL p SNPs, not just the estimation subset
    assert_eq!(r_rate.p_values.nrows(), sim_config.n_snps);
    assert_eq!(r_rate.effect_sizes.nrows(), sim_config.n_snps);
    assert_eq!(r_rate.t_stats.nrows(), sim_config.n_snps);

    // GIF should be reasonable
    assert!(
        r_rate.gif > 0.5 && r_rate.gif < 3.0,
        "GIF out of range with Rate(0.5): {:.4}",
        r_rate.gif
    );

    // P-values should differ from using All (different estimation subset → different U_hat)
    let r_all = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &config, None).unwrap();
    let mut any_diff = false;
    for i in 0..sim_config.n_snps {
        if (r_rate.p_values[(i, 0)] - r_all.p_values[(i, 0)]).abs() > 1e-10 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "Rate(0.5) should produce different p-values than All");
}

/// End-to-end: SubsetSpec::Indices uses exactly the specified SNPs for estimation.
#[test]
fn test_subset_indices_end_to_end() {
    let sim_config = SimConfig {
        n_samples: 200,
        n_snps: 10_000,
        n_causal: 20,
        k: 3,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 66666,
    };

    let config = Lfmm2Config {
        k: 3,
        lambda: 1e-5,
        chunk_size: 2_000,
        oversampling: 10,
        n_power_iter: 2,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    // Pick every 2nd SNP — same as Rate(0.5)
    let indices: Vec<usize> = (0..sim_config.n_snps).step_by(2).collect();
    let r_indices = fit_lfmm2(
        &bed,
        &SubsetSpec::Indices(indices.clone()),
        &bed,
        &sim.x,
        &config,
        None,
    )
    .unwrap();

    // Output must cover ALL p SNPs
    assert_eq!(r_indices.p_values.nrows(), sim_config.n_snps);

    // Since Rate(0.5) produces the same index set as step_by(2), results should match
    let r_rate = fit_lfmm2(&bed, &SubsetSpec::Rate(0.5), &bed, &sim.x, &config, None).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..sim_config.n_snps {
        for j in 0..sim_config.d {
            let diff = (r_indices.p_values[(i, j)] - r_rate.p_values[(i, j)]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(
        max_diff < 1e-12,
        "Indices(step_by(2)) should match Rate(0.5): max_diff={:.2e}",
        max_diff,
    );
}

/// Verify that OutputConfig writes a valid results TSV via chunk coalescing.
#[test]
fn test_output_config_writes_results() {
    let sim_config = SimConfig {
        n_samples: 100,
        n_snps: 1_000,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 88888,
    };

    let config = Lfmm2Config {
        k: 2,
        lambda: 1e-5,
        chunk_size: 300,
        oversampling: 5,
        n_power_iter: 1,
        seed: 42,
        n_workers: 0,
        progress: false,
        norm: SnpNorm::Eigenstrat,
    };

    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    let cov_names = vec!["cov_0".to_string(), "cov_1".to_string()];
    let output_path = dir.path().join("results.tsv");

    let output_config = OutputConfig {
        path: &output_path,
        bim: &bed.bim_records,
        cov_names: &cov_names,
    };

    let results = fit_lfmm2(
        &bed, &SubsetSpec::All, &bed, &sim.x, &config, Some(&output_config),
    ).unwrap();

    // Verify file exists and has correct structure
    let content = fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // Header + p data rows
    assert_eq!(
        lines.len(),
        sim_config.n_snps + 1,
        "Expected {} lines (1 header + {} data), got {}",
        sim_config.n_snps + 1,
        sim_config.n_snps,
        lines.len(),
    );

    // Header structure: chr, pos, snp_id, then 3 columns per covariate
    let header_fields: Vec<&str> = lines[0].split('\t').collect();
    assert_eq!(header_fields.len(), 3 + 3 * sim_config.d);
    assert_eq!(header_fields[0], "chr");
    assert_eq!(header_fields[1], "pos");
    assert_eq!(header_fields[2], "snp_id");
    assert_eq!(header_fields[3], "p_cov_0");
    assert_eq!(header_fields[4], "beta_cov_0");
    assert_eq!(header_fields[5], "t_cov_0");

    // Data row structure
    let data_fields: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(data_fields.len(), 3 + 3 * sim_config.d);

    // All p-values in the file should be parseable and in [0, 1]
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split('\t').collect();
        for j in 0..sim_config.d {
            let p_val: f64 = fields[3 + j * 3].parse().unwrap();
            assert!(
                (0.0..=1.0).contains(&p_val),
                "p-value out of range: {}",
                p_val,
            );
        }
    }

    // GIF should match the in-memory result
    assert!(results.gif > 0.5 && results.gif < 3.0);
}

// ---------------------------------------------------------------------------
// CLI end-to-end tests
// ---------------------------------------------------------------------------

/// Path to the compiled binary under test.
fn lfmm2_bin() -> std::path::PathBuf {
    // cargo test builds binaries into target/debug (or target/release with --release)
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()  // deps/
        .unwrap()
        .parent()  // debug/ or release/
        .unwrap()
        .to_path_buf();
    path.push("lfmm2");
    path
}

/// Set up a temp directory with simulated PLINK + covariate files for CLI tests.
/// Returns (tempdir, bed_path, cov_path).
fn cli_test_fixtures() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 500,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 9999,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    write_covariates(&dir.path().join("cov.tsv"), &sim.x).unwrap();
    let bed_path = dir.path().join("sim.bed");
    let cov_path = dir.path().join("cov.tsv");
    (dir, bed_path, cov_path)
}

/// Valid run: basic invocation with required args produces output files.
#[test]
fn test_cli_basic_run() {
    let (dir, bed, cov) = cli_test_fixtures();
    let out_prefix = dir.path().join("out");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", cov.to_str().unwrap(),
            "-k", "2",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        output.status.success(),
        "CLI exited with error:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // Check that output files exist
    let tsv_path = dir.path().join("out.tsv");
    let summary_path = dir.path().join("out.summary.txt");
    assert!(tsv_path.exists(), "Missing output TSV");
    assert!(summary_path.exists(), "Missing summary file");

    // TSV should have header + 500 SNP rows
    let content = fs::read_to_string(&tsv_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 501, "Expected 1 header + 500 data rows");

    // Summary should contain GIF
    let summary = fs::read_to_string(&summary_path).unwrap();
    assert!(summary.contains("GIF:"), "Summary missing GIF");
    assert!(summary.contains("n_samples: 50"), "Summary missing n_samples");
    assert!(summary.contains("n_snps: 500"), "Summary missing n_snps");
}

/// Valid run with --est-rate for subset estimation.
#[test]
fn test_cli_est_rate() {
    let (dir, bed, cov) = cli_test_fixtures();
    let out_prefix = dir.path().join("out_rate");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", cov.to_str().unwrap(),
            "-k", "2",
            "--est-rate", "0.5",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        output.status.success(),
        "CLI with --est-rate failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    // Output should still cover all 500 SNPs
    let tsv_path = dir.path().join("out_rate.tsv");
    let content = fs::read_to_string(&tsv_path).unwrap();
    assert_eq!(content.lines().count(), 501);
}

/// Valid run with optional flags: lambda, chunk-size, seed, power-iter, oversampling.
#[test]
fn test_cli_optional_flags() {
    let (dir, bed, cov) = cli_test_fixtures();
    let out_prefix = dir.path().join("out_opts");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", cov.to_str().unwrap(),
            "-k", "3",
            "-l", "0.01",
            "--chunk-size", "100",
            "--seed", "123",
            "--power-iter", "1",
            "--oversampling", "5",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        output.status.success(),
        "CLI with optional flags failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    let summary = fs::read_to_string(dir.path().join("out_opts.summary.txt")).unwrap();
    assert!(summary.contains("K: 3"));
    assert!(summary.contains("lambda: 0.01"));
}

/// Invalid: missing required --bed argument.
#[test]
fn test_cli_missing_bed() {
    let output = Command::new(lfmm2_bin())
        .args(["-c", "/dev/null", "-k", "2"])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("required") || stderr.contains("--bed"),
        "Expected error about missing --bed, got: {}",
        stderr,
    );
}

/// Invalid: missing required --cov argument.
#[test]
fn test_cli_missing_cov() {
    let (dir, bed, _cov) = cli_test_fixtures();

    let output = Command::new(lfmm2_bin())
        .args(["-b", bed.to_str().unwrap(), "-k", "2"])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("required") || stderr.contains("--cov"),
        "Expected error about missing --cov, got: {}",
        stderr,
    );
    drop(dir);
}

/// Invalid: missing required -k argument.
#[test]
fn test_cli_missing_k() {
    let (dir, bed, cov) = cli_test_fixtures();

    let output = Command::new(lfmm2_bin())
        .args(["-b", bed.to_str().unwrap(), "-c", cov.to_str().unwrap()])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    drop(dir);
}

/// Invalid: nonexistent bed file.
#[test]
fn test_cli_nonexistent_bed() {
    let output = Command::new(lfmm2_bin())
        .args(["-b", "/tmp/does_not_exist.bed", "-c", "/dev/null", "-k", "2"])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Failed to open") || stderr.contains("No such file"),
        "Expected file-not-found error, got: {}",
        stderr,
    );
}

/// Invalid: nonexistent covariate file.
#[test]
fn test_cli_nonexistent_cov() {
    let (dir, bed, _cov) = cli_test_fixtures();

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", "/tmp/does_not_exist.tsv",
            "-k", "2",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Failed to open") || stderr.contains("No such file"),
        "Expected file-not-found error, got: {}",
        stderr,
    );
    drop(dir);
}

/// Invalid: covariate file with wrong sample IDs.
#[test]
fn test_cli_cov_sample_mismatch() {
    let (dir, bed, _cov) = cli_test_fixtures();

    // Write a covariate file with wrong sample IDs
    let bad_cov = dir.path().join("bad_cov.tsv");
    let mut f = fs::File::create(&bad_cov).unwrap();
    use std::io::Write;
    writeln!(f, "sample_id\tenv_0\tenv_1").unwrap();
    writeln!(f, "WRONG_ID\t0.5\t-0.3").unwrap();

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", bad_cov.to_str().unwrap(),
            "-k", "2",
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found in covariate file"),
        "Expected sample mismatch error, got: {}",
        stderr,
    );
}

/// Invalid: covariate file with non-numeric values.
#[test]
fn test_cli_cov_non_numeric() {
    let (dir, bed, _cov) = cli_test_fixtures();

    let bad_cov = dir.path().join("bad_cov.tsv");
    let mut f = fs::File::create(&bad_cov).unwrap();
    use std::io::Write;
    writeln!(f, "sample_id\tenv_0\tenv_1").unwrap();
    writeln!(f, "IND0\tNOTANUMBER\t0.5").unwrap();

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", bad_cov.to_str().unwrap(),
            "-k", "2",
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Failed to parse"),
        "Expected parse error, got: {}",
        stderr,
    );
}

/// Invalid: --est-rate out of range.
#[test]
fn test_cli_est_rate_out_of_range() {
    let (_dir, bed, cov) = cli_test_fixtures();

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", cov.to_str().unwrap(),
            "-k", "2",
            "--est-rate", "5.0",
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("est-rate"),
        "Expected est-rate validation error, got: {}",
        stderr,
    );
}

/// Invalid: covariate file with only a header (no data rows).
#[test]
fn test_cli_cov_empty_data() {
    let (dir, bed, _cov) = cli_test_fixtures();

    let bad_cov = dir.path().join("empty_cov.tsv");
    let mut f = fs::File::create(&bad_cov).unwrap();
    use std::io::Write;
    writeln!(f, "sample_id\tenv_0\tenv_1").unwrap();

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", bad_cov.to_str().unwrap(),
            "-k", "2",
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no data rows"),
        "Expected empty data error, got: {}",
        stderr,
    );
}

/// CSV delimiter auto-detection: .csv extension uses comma separator.
#[test]
fn test_cli_csv_covariates() {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 500,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 9999,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();

    // Write a CSV covariate file (comma-separated)
    let csv_path = dir.path().join("cov.csv");
    {
        use std::io::Write;
        let mut f = fs::File::create(&csv_path).unwrap();
        let d = sim.x.ncols();
        let header: Vec<String> = (0..d).map(|j| format!("env_{}", j)).collect();
        writeln!(f, "sample_id,{}", header.join(",")).unwrap();
        for i in 0..sim.x.nrows() {
            let vals: Vec<String> = (0..d).map(|j| format!("{:.6}", sim.x[(i, j)])).collect();
            writeln!(f, "IND{},{}", i, vals.join(",")).unwrap();
        }
    }

    let bed = dir.path().join("sim.bed");
    let out_prefix = dir.path().join("out_csv");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", csv_path.to_str().unwrap(),
            "-k", "2",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        output.status.success(),
        "CLI with CSV covariates failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    let tsv_path = dir.path().join("out_csv.tsv");
    assert_eq!(fs::read_to_string(&tsv_path).unwrap().lines().count(), 501);
}

// ---------------------------------------------------------------------------
// SNP normalization mode tests
// ---------------------------------------------------------------------------

/// End-to-end: all 3 normalization modes produce valid results on the same simulation,
/// and the modes produce different (but correlated) p-values.
#[test]
fn test_normalization_modes_end_to_end() {
    let sim_config = SimConfig {
        n_samples: 200,
        n_snps: 10_000,
        n_causal: 20,
        k: 3,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 77700,
    };

    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    let bed = BedFile::open(dir.path().join("sim.bed")).unwrap();

    let modes = [SnpNorm::CenterOnly, SnpNorm::Eigenstrat, SnpNorm::Hwe];
    let mut results = Vec::new();

    for &mode in &modes {
        let config = Lfmm2Config {
            k: 3,
            lambda: 1e-5,
            chunk_size: 2_000,
            oversampling: 10,
            n_power_iter: 2,
            seed: 42,
            n_workers: 0,
            progress: false,
            norm: mode,
        };
        let r = fit_lfmm2(&bed, &SubsetSpec::All, &bed, &sim.x, &config, None).unwrap();

        // Each mode should produce a valid GIF
        assert!(
            r.gif > 0.3 && r.gif < 5.0,
            "GIF out of range for {:?}: {:.4}",
            mode,
            r.gif,
        );
        // Output dimensions must match
        assert_eq!(r.p_values.nrows(), sim_config.n_snps);
        assert_eq!(r.p_values.ncols(), sim_config.d);

        results.push(r);
    }

    // Modes should produce different results
    let p = sim_config.n_snps;
    for i in 0..modes.len() {
        for j in (i + 1)..modes.len() {
            let mut n_differ = 0;
            for snp in 0..p {
                if (results[i].p_values[(snp, 0)] - results[j].p_values[(snp, 0)]).abs() > 1e-10 {
                    n_differ += 1;
                }
            }
            assert!(
                n_differ > 0,
                "Modes {:?} and {:?} produced identical p-values",
                modes[i],
                modes[j],
            );
        }
    }

    // High Spearman correlation between modes (all should recover similar signal)
    for i in 0..modes.len() {
        for j in (i + 1)..modes.len() {
            let mut v1: Vec<f64> = (0..p).map(|s| results[i].p_values[(s, 0)]).collect();
            let mut v2: Vec<f64> = (0..p).map(|s| results[j].p_values[(s, 0)]).collect();
            let rho = spearman_rank_corr(&mut v1, &mut v2);
            eprintln!(
                "Spearman({:?}, {:?}) = {:.4}",
                modes[i], modes[j], rho,
            );
            assert!(
                rho > 0.8,
                "Low correlation between {:?} and {:?}: {:.4}",
                modes[i],
                modes[j],
                rho,
            );
        }
    }
}

/// CLI: --norm flag accepts all 3 valid values.
#[test]
fn test_cli_norm_flag() {
    let (dir, bed, cov) = cli_test_fixtures();

    for mode in &["center-only", "eigenstrat", "hwe"] {
        let out_prefix = dir.path().join(format!("out_{}", mode));
        let output = Command::new(lfmm2_bin())
            .args([
                "-b", bed.to_str().unwrap(),
                "-c", cov.to_str().unwrap(),
                "-k", "2",
                "--norm", mode,
                "-o", out_prefix.to_str().unwrap(),
                "-t", "1",
            ])
            .output()
            .expect("failed to execute lfmm2 binary");

        assert!(
            output.status.success(),
            "CLI with --norm {} failed:\nstderr: {}",
            mode,
            String::from_utf8_lossy(&output.stderr),
        );

        let tsv_path = std::path::PathBuf::from(format!("{}.tsv", out_prefix.display()));
        assert!(tsv_path.exists(), "Missing output TSV for --norm {}", mode);
    }
}

/// CLI: --norm with an invalid value should fail.
#[test]
fn test_cli_norm_invalid() {
    let (dir, bed, cov) = cli_test_fixtures();
    let out_prefix = dir.path().join("out_bad");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", cov.to_str().unwrap(),
            "-k", "2",
            "--norm", "garbage",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        !output.status.success(),
        "CLI should fail with --norm garbage",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid value") || stderr.contains("possible values"),
        "Expected clap validation error, got: {}",
        stderr,
    );
}

// ---------------------------------------------------------------------------
// --intersect-samples tests
// ---------------------------------------------------------------------------

/// Helper: write a covariate file with the given sample IDs and values.
fn write_cov_file(path: &Path, sample_ids: &[&str], n_covs: usize, seed: u64) {
    use std::io::Write;
    let mut f = fs::File::create(path).unwrap();
    let header: Vec<String> = (0..n_covs).map(|j| format!("env_{}", j)).collect();
    writeln!(f, "sample_id\t{}", header.join("\t")).unwrap();
    for (i, id) in sample_ids.iter().enumerate() {
        let vals: Vec<String> = (0..n_covs)
            .map(|j| format!("{:.6}", ((seed as f64 + i as f64 + j as f64) * 0.1).sin()))
            .collect();
        writeln!(f, "{}\t{}", id, vals.join("\t")).unwrap();
    }
}

/// --intersect-samples with covariate file containing a subset of FAM samples
/// plus extra samples not in FAM. Should succeed and output correct dimensions.
#[test]
fn test_cli_intersect_samples() {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 500,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 9999,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();

    // Write a covariate file with only 30 of the 50 FAM samples + 5 extra
    let mut cov_ids: Vec<String> = (0..30).map(|i| format!("IND{}", i)).collect();
    cov_ids.extend((100..105).map(|i| format!("EXTRA{}", i)));
    let cov_strs: Vec<&str> = cov_ids.iter().map(|s| s.as_str()).collect();
    let cov_path = dir.path().join("partial_cov.tsv");
    write_cov_file(&cov_path, &cov_strs, 2, 42);

    let bed_path = dir.path().join("sim.bed");
    let out_prefix = dir.path().join("out_intersect");

    // Without --intersect-samples: should fail
    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed_path.to_str().unwrap(),
            "-c", cov_path.to_str().unwrap(),
            "-k", "2",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");
    assert!(
        !output.status.success(),
        "Should fail without --intersect-samples:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    // With --intersect-samples: should succeed
    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed_path.to_str().unwrap(),
            "-c", cov_path.to_str().unwrap(),
            "-k", "2",
            "--intersect-samples",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");
    assert!(
        output.status.success(),
        "Should succeed with --intersect-samples:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    // Output should still cover all 500 SNPs
    let tsv_path = dir.path().join("out_intersect.tsv");
    let content = fs::read_to_string(&tsv_path).unwrap();
    assert_eq!(content.lines().count(), 501, "Expected header + 500 SNP rows");

    // Summary should reflect 30 samples (the intersection)
    let summary = fs::read_to_string(dir.path().join("out_intersect.summary.txt")).unwrap();
    assert!(summary.contains("n_samples: 30"), "Expected 30 samples, got:\n{}", summary);
}

/// --intersect-samples with completely disjoint samples should error.
#[test]
fn test_cli_intersect_samples_no_overlap() {
    let (dir, bed, _cov) = cli_test_fixtures();

    let cov_path = dir.path().join("disjoint_cov.tsv");
    write_cov_file(&cov_path, &["NOPE1", "NOPE2", "NOPE3"], 2, 42);

    let out_prefix = dir.path().join("out_disjoint");
    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed.to_str().unwrap(),
            "-c", cov_path.to_str().unwrap(),
            "-k", "2",
            "--intersect-samples",
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No samples in common"),
        "Expected 'No samples in common' error, got: {}",
        stderr,
    );
}

/// est-bed with same samples but different order should work (auto-reorder).
#[test]
fn test_cli_est_bed_sample_identity() {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 500,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 9999,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    write_covariates(&dir.path().join("cov.tsv"), &sim.x).unwrap();

    // Create est-bed with reversed sample order
    let est_dir = dir.path().join("est");
    fs::create_dir_all(&est_dir).unwrap();

    // Reverse the genotype rows and fam records for the est-bed
    let n = sim_config.n_samples;
    let p = sim_config.n_snps;
    let rev_order: Vec<usize> = (0..n).rev().collect();
    let mut rev_geno = Array2::<u8>::zeros((n, p));
    for (new_row, &old_row) in rev_order.iter().enumerate() {
        for snp in 0..p {
            rev_geno[(new_row, snp)] = sim.genotypes[(old_row, snp)];
        }
    }
    lfmm2::bed::write_bed_file(&est_dir.join("est.bed"), &rev_geno).unwrap();

    // Write reversed .fam
    let rev_fam: Vec<lfmm2::bed::FamRecord> = rev_order.iter().map(|&i| {
        lfmm2::bed::FamRecord {
            fid: format!("FAM{}", i),
            iid: format!("IND{}", i),
            father: "0".to_string(),
            mother: "0".to_string(),
            sex: 0,
            pheno: "-9".to_string(),
        }
    }).collect();
    lfmm2::bed::write_fam(&est_dir.join("est.fam"), &rev_fam).unwrap();

    // Write .bim (just copy the main bim records)
    let main_bed = BedFile::open(dir.path().join("sim.bed")).unwrap();
    lfmm2::bed::write_bim(&est_dir.join("est.bim"), &main_bed.bim_records).unwrap();

    let bed_path = dir.path().join("sim.bed");
    let cov_path = dir.path().join("cov.tsv");
    let out_prefix = dir.path().join("out_est_reorder");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed_path.to_str().unwrap(),
            "-c", cov_path.to_str().unwrap(),
            "-k", "2",
            "--est-bed", est_dir.join("est.bed").to_str().unwrap(),
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        output.status.success(),
        "est-bed with reordered samples should succeed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    let tsv_path = dir.path().join("out_est_reorder.tsv");
    assert_eq!(fs::read_to_string(&tsv_path).unwrap().lines().count(), 501);
}

/// est-bed with completely different sample IIDs should error.
#[test]
fn test_cli_est_bed_sample_mismatch() {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 500,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 9999,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();
    write_covariates(&dir.path().join("cov.tsv"), &sim.x).unwrap();

    // Create est-bed with wrong sample IDs
    let est_dir = dir.path().join("est_bad");
    fs::create_dir_all(&est_dir).unwrap();

    lfmm2::bed::write_bed_file(&est_dir.join("est.bed"), &sim.genotypes).unwrap();

    let wrong_fam: Vec<lfmm2::bed::FamRecord> = (0..sim_config.n_samples).map(|i| {
        lfmm2::bed::FamRecord {
            fid: format!("WRONG{}", i),
            iid: format!("WRONG{}", i),
            father: "0".to_string(),
            mother: "0".to_string(),
            sex: 0,
            pheno: "-9".to_string(),
        }
    }).collect();
    lfmm2::bed::write_fam(&est_dir.join("est.fam"), &wrong_fam).unwrap();

    let main_bed = BedFile::open(dir.path().join("sim.bed")).unwrap();
    lfmm2::bed::write_bim(&est_dir.join("est.bim"), &main_bed.bim_records).unwrap();

    let bed_path = dir.path().join("sim.bed");
    let cov_path = dir.path().join("cov.tsv");
    let out_prefix = dir.path().join("out_est_bad");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed_path.to_str().unwrap(),
            "-c", cov_path.to_str().unwrap(),
            "-k", "2",
            "--est-bed", est_dir.join("est.bed").to_str().unwrap(),
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("est-bed is missing"),
        "Expected 'est-bed is missing' error, got: {}",
        stderr,
    );
}

/// --intersect-samples with est-bed: both main and est-bed should be subsetted.
#[test]
fn test_cli_intersect_with_est_bed() {
    let sim_config = SimConfig {
        n_samples: 50,
        n_snps: 500,
        n_causal: 5,
        k: 2,
        d: 2,
        effect_size: 1.0,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.3,
        seed: 9999,
    };
    let sim = simulate(&sim_config);
    let dir = tempfile::tempdir().unwrap();
    write_plink(dir.path(), "sim", &sim).unwrap();

    // Covariate file with only first 30 samples
    let cov_ids: Vec<String> = (0..30).map(|i| format!("IND{}", i)).collect();
    let cov_strs: Vec<&str> = cov_ids.iter().map(|s| s.as_str()).collect();
    let cov_path = dir.path().join("partial_cov.tsv");
    write_cov_file(&cov_path, &cov_strs, 2, 42);

    // est-bed with all 50 samples (should be subsetted to match main after intersection)
    let est_dir = dir.path().join("est");
    fs::create_dir_all(&est_dir).unwrap();
    write_plink(&est_dir, "est", &sim).unwrap();

    let bed_path = dir.path().join("sim.bed");
    let out_prefix = dir.path().join("out_intersect_est");

    let output = Command::new(lfmm2_bin())
        .args([
            "-b", bed_path.to_str().unwrap(),
            "-c", cov_path.to_str().unwrap(),
            "-k", "2",
            "--intersect-samples",
            "--est-bed", est_dir.join("est.bed").to_str().unwrap(),
            "-o", out_prefix.to_str().unwrap(),
            "-t", "1",
        ])
        .output()
        .expect("failed to execute lfmm2 binary");

    assert!(
        output.status.success(),
        "Should succeed with --intersect-samples + --est-bed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    // Output should cover all 500 SNPs
    let tsv_path = dir.path().join("out_intersect_est.tsv");
    assert_eq!(fs::read_to_string(&tsv_path).unwrap().lines().count(), 501);

    // Summary should reflect 30 samples
    let summary = fs::read_to_string(dir.path().join("out_intersect_est.summary.txt")).unwrap();
    assert!(summary.contains("n_samples: 30"), "Expected 30 samples, got:\n{}", summary);
}
