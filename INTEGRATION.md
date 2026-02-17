# Integration Tests

## Quick test (Tier 1)

200 samples × 10k SNPs. Runs in ~2 seconds.

```sh
cargo test --release
```

## Large simulation (Tier 2)

1000 samples × 1M SNPs. Runs in a few minutes. Writes results to `testdata/`.

```sh
cargo test --release -- --ignored
```

## R cross-reference

After running the large simulation:

```sh
cd testdata
Rscript run_lea_comparison.R
```

Requires the [LEA](https://bioconductor.org/packages/LEA/) Bioconductor package. Compares Rust p-values against LEA's `lfmm2()` via Spearman rank correlation.

## Output files (Tier 2)

All written to `testdata/`:

| File | Contents |
|------|----------|
| `sim.bed/.bim/.fam` | Simulated PLINK genotypes |
| `sim_covariates.txt` | Covariate matrix |
| `sim_truth.tsv` | Ground truth (causal status, true betas) |
| `sim_latent_U.tsv` | True latent factors |
| `sim.lfmm` | Genotypes in LEA format |
| `sim_rust_pvalues.tsv` | Rust calibrated p-values |
| `sim_rust_tstats.tsv` | Rust t-statistics |
| `sim_rust_effects.tsv` | Rust effect size estimates |
| `sim_rust_summary.txt` | GIF and run metadata |
| `run_lea_comparison.R` | R script for LEA comparison |
