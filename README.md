# schnelLFMM

A fast streaming implementation of LFMM2 (Latent Factor Mixed Model) for GWAS
and GEA.


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

Optionally, we support non-negative matrix factorisation-based data imputation.
In summary, this imputes missing SNPs based on the expected population
differences, as estimated by the NNMF algorithm. Conceptually, this is similar
to STRUCTURE and ADMIXTURE, where each sample's genotypes are modeled as a
mixture of ancestral population allele states combined according to the
sample's admixture proportions. We use NNMF to impute genotypes by first
estimating the sample's admixture proportions (by fitting NNMF: $Y = W H$),
then use these to predict any missing genotypes per SNP.

Importantly, this imputation is limited in scope, in that we are only
trying to enable and improve the association itself, not produce an accurate
complete SNP matrix (use BEAGLE or other tools for that, where possible). Take
a case of GEA across two diverged populations: at some SNP with a few missing
calls, whose genotypes broadly follow population structure, we have two
alternatives: we can fill missing values with the global mean, or fill with the
population-specific mean values. If we fill with a global mean, we introduce
more error as imputed values fall between the two population-level means,
whereas if predict mean genotypes based on sample admixture proportions, then
the imputed values follow underlying population structure. Given the
mixed-effect LFMM term corrects for this background population structure, NMF
effectively ensures that population-imputed values don't induce bias in the
estimation of fixed covariate effects. Conversely, global-mean imputation may
leave some residual signal after population structure correction (as values
don't follow population structure), and may induce error during estimation of
fixed effects.

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
| `--nmf-impute` | off | Use NMF low-rank imputation for missing genotypes (default: mean imputation) |
| `--nmf-k` | same as `-k` | NMF rank (number of ancestral components) |
| `--nmf-iter` | `10` | NMF multiplicative update iterations; monitor the per-iteration CV MAE in the summary file for convergence |
| `--nmf-cv-rate` | `0.0005` | Fraction of genotypes held out per NMF iteration for cross-validation |
| `-v, --verbose` | off | Verbose progress output |

### Example

We have an example in `examples/ath` that reanalyses the days to flowering at
10C trait from AraGWAS.

### Output

Results are written to `<out>.tsv` (per-SNP effect sizes, t-statistics, and
calibrated p-values for each trait, plus per-SNP $R^2$ fractions for
covariates, latent factors, and residual variation) and `<out>.summary.txt`
with a summary of run arguments and details (including NMF cross-validation
errors when `--nmf-impute` is used).

## Differences to R's LEA::lfmm2()

Overall, our implementation tries to closely follow Caye et al's paper and
implementation in the R package LEA. However, there are some differences to
increase scalability.

Firstly, we accept standard PLINK 1.9 BED+FAM files for genotypes. No other
reformatting of genotypes should be needed, and analyses leading to GEA or GWAS
can use standard PLINK tooling for efficiency.

We estimate genotype latent factors ($\mathbf{U}$) using randomised SVD. We use
the Halko-Martinsson-Tropp algorithm (https://arxiv.org/abs/0909.4061) which
computes only a sparse subset of the SVD, and allows streaming over the BED
file without loading $\mathbf{Y}$ into memory.

We then compute the effect sizes $\mathbf{B}$ with ridge regression and do
per-locus statistical tests in an additional streaming pass over SNPs. We
finally do a per-trait GIF correction of test statistics to avoid inflated
p-values.

These multiple passes of the BED file might seem inefficient, however with
modern SSDs computation time will vastly exceed the IO speed, so the overall
contribution to runtime is worth the vastly improved memory efficiency.

Our NMF imputation algorithm is inspired by those in LEA, rather than directly
implementing their exact approach. This is mostly to avoid multiple full passes
over the entire genome file (as opposed to the estimation subset). Provided the
--est parameters (either the independent estimation BED or subsampling rate)
produce a snp subset from which an accurate picture of genome-wide relatedness
can be learnt, then there is should be no practical difference.
