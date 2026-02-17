use lfmm2::bed::BedFile;
use lfmm2::simulate::{
    simulate, write_covariates, write_ground_truth, write_latent_u, write_lfmm_format,
    write_plink, write_r_comparison_script, SimConfig,
};
use lfmm2::{fit_lfmm2, Lfmm2Config};
use ndarray::Array2;
use ndarray_linalg::SVD;
use std::fs;
use std::path::Path;

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
    let results = fit_lfmm2(&bed, &bed, &sim.x, &config).unwrap();

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
    let results = fit_lfmm2(&bed, &bed, &sim.x, &config).unwrap();

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
    };

    let sim = simulate(&sim_config);

    let dir1 = tempfile::tempdir().unwrap();
    write_plink(dir1.path(), "sim", &sim).unwrap();
    let bed1 = BedFile::open(dir1.path().join("sim.bed")).unwrap();
    let r1 = fit_lfmm2(&bed1, &bed1, &sim.x, &config).unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    write_plink(dir2.path(), "sim", &sim).unwrap();
    let bed2 = BedFile::open(dir2.path().join("sim.bed")).unwrap();
    let r2 = fit_lfmm2(&bed2, &bed2, &sim.x, &config).unwrap();

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
    };
    let r_seq = fit_lfmm2(&bed, &bed, &sim.x, &config_seq).unwrap();

    // Parallel run
    let config_par = Lfmm2Config {
        n_workers: 2,
        ..config_seq
    };
    let r_par = fit_lfmm2(&bed, &bed, &sim.x, &config_par).unwrap();

    // P-values should match within floating-point tolerance.
    // Pattern A loops are bitwise identical; Pattern B may differ at ~1e-14
    // due to FP summation order via Mutex.
    let tol = 1e-10;
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
    let gif_diff = (r_seq.gif - r_par.gif).abs();
    eprintln!("GIF diff: {:.2e}", gif_diff);
    assert!(
        gif_diff < tol,
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
