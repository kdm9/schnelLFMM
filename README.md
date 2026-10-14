# schnelLFMM

A fast streaming implementation of LFMM2 (Latent Factor Mixed Model) for GWAS
and GEA on large whole-genome datasets


## Overview

LFMM2 is a common approach for Genotype-Environment associations, initially
described in [Caye et al. (2019)](https://academic.oup.com/mbe/article/36/4/852/5290100),
and implemented in the [LEA R package
](https://bioconductor.org/packages//release/bioc/html/LEA.html). This tool is
a reimplementation of the LFMM2 model in Rust that uses a randomised streaming
approach to both compute genome blocks in parallel, and reduce memory usage
for colossal datasets.

schnelLFMM fits the model $\mathbf{Y} = \mathbf{X} \mathbf{B}^\top + \mathbf{U}
\mathbf{V}^\top + \mathbf{E}$ where $\mathbf{Y}$ is a genotype matrix,
$\mathbf{X}$ contains environmental variables/phenotypes, $\mathbf{U}
\mathbf{V}^\top$ captures latent population structure, and $\mathbf{B}$ holds
the per-SNP effect sizes to be tested.

schnelLFMM scales to massive SNP datasets by streaming PLINK .bed files from disk,
so that the genotype matrix is never fully loaded into RAM. Even on smaller
datasets, it dramatically outperforms LEA, taking only about 2 seconds to run a
single-trait GWAS in Arabidopsis, compared to a few minutes for LEA.

Optionally, we support latent-factor-based data imputation. In summary, this
imputes missing SNPs based on the expected population differences, using the
same latent factors $\hat{U}$ that LFMM already fits to correct for population
structure. Each missing genotype is filled with its fitted value under the
latent-factor term alone, so imputed values follow the underlying structure
rather than sitting at the global mean.


Why schnelLFMM? Schnell is German for fast, and I can't help myself from making
dumb punny tool names. Sorry.

## Install

Download a pre-built binary from [Releases](https://github.com/kdm9/schnelLFMM/releases),
or build from source:

```sh
# Requires Rust toolchain (https://rustup.rs), and gcc/gfortran to compile OpenBLAS
cargo install --git https://github.com/kdm9/schnelLFMM
```

Only Linux is officially supported, but schnelLFMM compiles for Mac M-series
machines, and in theory on Windows and other supported rust targets, if anyone
works out how to get OpenBLAS to behave.

## Usage

```
schnellfmm --bed <genotypes.bed> --cov <covariates.csv> --out <results> -k <K> [OPTIONS]
```

For example...

```sh
# Basic run with 12 latent factors
schnellfmm -b genotypes.bed -c phenotypes.csv -k 12

# LD-pruned estimation subset, intersect samples, verbose
schnellfmm -b all_snps.bed -c pheno.tsv -k 20 \
  --est-bed pruned.bed --intersect-samples -v
```

### Required arguments

| Flag | Description |
|------|-------------|
| `-b, --bed` | PLINK .bed file (with matching .bim/.fam) |
| `-c, --cov` | Covariate/phenotype file (CSV or TSV, header row, first column = sample ID) |
| `-k` | Number of latent factors |

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `-l, --lambda` | `1e-5` | Ridge penalty ($\lambda > 0$ required for identifiability) |
| `-o, --out` | `lfmm2_out` | Output file prefix |
| `-t, --threads` | N cpus - 1 | Worker threads |
| `--est-bed` |  | Separate LD-pruned .bed for factor estimation |
| `--est-rate` |  | Create an estimation subset of SNPs as a subsample of all snps from --bed at this rate (e.g. `0.01`) |
| `--intersect-samples` | off | Use only samples present in both .fam and covariate files |
| `--norm` | `eigenstrat` | SNP variance normalization mode. Eigenstrat scales by expected variance under HWE, see Reich et al |
| `--scale-cov` | off | Scale covariates to unit variance |
| `--chunk-size` | `10000` | SNPs per processing chunk |
| `--power-iter` | `2` | Number of RSVD power iterations |
| `--oversampling` | `10` | How many additional singular values should be estimated during RSVD? |
| `--seed` | `42` | RNG seed |
| `--latent-impute` | off | Impute missing genotypes from the fitted latent-factor term (default: mean imputation). The imputation rank is `-k` |
| `--impute-cv-rate` | `0.0005` | Fraction of observed genotypes held out to measure imputation accuracy against the mean-imputation baseline |
| `-v, --verbose` | off | Verbose progress output |

### Example

We have an example in `examples/ath` that reanalyses the days to flowering at
10C trait from AraGWAS.

### Output

Results are written to `<out>.tsv` (per-SNP effect sizes, t-statistics, and
calibrated p-values for each trait, plus per-SNP $R^2$ fractions for
covariates, latent factors, and residual variation) and `<out>.summary.txt`
with a summary of run arguments and details: the overall genomic inflation
factor (`GIF:`), the per-covariate GIFs (`GIF[<covariate>]:`, unclamped, so
values below 1 indicate deflation), and held-out imputation accuracy when
`--latent-impute` is used.

Because these results can be quite large, schnelLFMM supports writing to
parquet files instead of tsv. These files can be directly read by `nanoparquet`
in R, or better yet `duckplyr` can use DuckDB to do out-of-core
filtering/manipulation on these datasets without even needing to read them all
into RAM. To enable this, use `--format parquet`. NB: for efficiency, when
writing to parquet files results are not necessarily sorted by genome position.
`duckplyr` can return sorted results (potentially after filtering), or just
sort it in R.


## Differences to R's LEA::lfmm2()

Overall, our implementation tries to closely follow Caye et al's paper and
implementation in the R package LEA. However, there are some differences to
increase scalability.

Firstly, we accept standard PLINK 1.9 BED+FAM files for genotypes. No other
reformatting of genotypes should be needed, and analyses leading to GEA or GWAS
can use standard PLINK tooling for efficiency.

Like Caye et al, we estimate genotype latent factors ($\mathbf{U}$) using
randomised SVD. We use the Halko-Martinsson-Tropp algorithm
(https://arxiv.org/abs/0909.4061) which computes only a sparse subset of the
SVD. Unlike Caye, we do this estimation on a random subset of the data
(`--est-bed` or `--est-rate`) for efficiency, and use an implementation that
allows streaming over the BED file without loading $\mathbf{Y}$ into memory.

We then compute the effect sizes $\mathbf{B}$ with ridge regression and do
per-locus statistical tests following Caye et al, however we do this in an
additional streaming pass over all SNPs. We finally do a per-trait GIF
correction of test statistics to avoid inflated p-values.

These multiple passes over the BED file might seem inefficient, however with
modern SSDs computation time will vastly exceed the IO speed, so the minor
increase in IO is worth the vastly improved memory efficiency (and, because we
can parallelise over chunks of genotypes, ends up much faster regardless).

## Latent-factor imputation

Missing genotypes are filled with their fitted value under the latent-factor
term of the LFMM model itself. Importantly, this imputation is limited in
scope, in that we are only trying to enable and improve the association itself,
not produce an accurate complete SNP matrix (use BEAGLE or other tools for
that, where possible). 

Take a case of GEA across two diverged populations: at some SNP with a few
missing calls, whose genotypes broadly follow population structure, we have two
alternatives: we can fill missing values with the global mean, or fill with the
population-specific mean values. If we fill with a global mean, we introduce
more error as imputed values fall between the two population-level means,
whereas if we predict genotypes from where each sample sits in the latent
structure, the imputed values follow that structure. Given the mixed-effect
LFMM term corrects for this background population structure, latent imputation
effectively ensures that imputed values don't induce bias in the estimation of
fixed covariate effects. Conversely, global-mean imputation may leave some
residual signal after population structure correction (as values don't follow
population structure), and may induce error during estimation of fixed effects.

In all cases, check the summary file. `gwas_cv_mae` is the held-out mean
absolute error of the imputer and `gwas_cv_mean_mae` is the mean-imputation
baseline on the same positions; if the former is not lower, re-run with
`--mean-impute`. This can happen in populations with very little structure,
where the correct K for imputation purposes is 1, i.e. mean imputation,
regardless of what the correct K for LFMM estimation is.
