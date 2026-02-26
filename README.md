# schnelLFMM

A fast streaming implementation of LFMM2 (Latent Factor Mixed Model) for GWAS
and GEA, orignally from Caye et al. (2019) "LFMM 2: Fast and Accurate Inference
of Gene-Environment Associations in Genome-Wide Studies." *Molecular Biology
and Evolution*, 36(4), 852–860.


## Overview

schnelLFMM fits the model **Y = X B' + U V' + E** where $\mathbf{Y}$ is a genotype matrix,
X contains environmental variables/phenotypes, U V' captures latent population structure, and
B holds the per-SNP effect sizes to be tested.

It scales to billions of SNPs by streaming PLINK .bed files from disk, so that the
genotype matrix is never fully loaded into RAM.


### Differences to R's LEA::lfmm2()

Overall, our implementation tries to closely follow Caye et al's paper and
implementation in the R package LEA. However, there are some differences to
increase scalability.

Firstly, we accept standard PLINK 1.9 BED+FAM files for genotypes. No other
reformatting of genotypes should be needed, and analyses leading to GEA or GWAS
can use standard PLINK tooling for efficiency.

We estimate genotype latent factors ($U$) using randomised SVD. We use the
Halko-Martinsson-Tropp algorithm (https://arxiv.org/abs/0909.4061) which allows
streaming over the BED file without loading Y into memory. 

We then compute the effect sizes B with ridge regression, and do per-locus
statistical tests in an addtional streaming pass over SNPs. We finally do a
per-trait GIF correction of test statistics to avoid inflated p-values.

These multiple passes of the BED file might seem inefficient, however with
modern SSDs computation time will vastly exceed the IO speed, so the overall
contribution to runtime is worth the vastly improved memory efficiency.


## Install

Download a pre-built binary from
[Releases](https://github.com/kdm9/schnelLFMM/releases), or build from
source:

```sh
# Requires Rust toolchain, gfortran
cargo install --git https://github.com/kdm9/schnelLFMM
```

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
| `-l, --lambda` | `1e-5` | Ridge penalty (λ > 0 required for identifiability) |
| `-o, --out` | `lfmm2_out` | Output file prefix |
| `-t, --threads` | N cpus - 1 | Worker threads |
| `--est-bed` |  | Separate LD-pruned .bed for factor estimation |
| `--est-rate` |  | Create an estimation subset of SNPs as a subsample of all snps from --bed at this rate (e.g. `0.01`) |
| `--intersect-samples` | off | Use only samples present in both .fam and covariate files |
| `--norm` | `eigenstrat` | SNP variance normalization mode. Eigenstrat scales by expected variance under HWE, see Reich et al |
| `--scale-cov` | off | Scale covariates to unit variance |
| `--chunk-size` | `10000` | SNPs per processing chunk |
| `--power-iter` | `2` | Number of RSVD power iterations |
| `--oversampling` | `10` | How many addtional singular values should be estimated during RSVD? |
| `--seed` | `42` | RNG seed |
| `-v, --verbose` | off | Verbose progress output |

### Example

We have an example in `examples/ath` that reanalyses the days to flowering at
10C trait from AraGWAS.

### Output

Results are written to `<out>.tsv` (per-SNP effect sizes, t-statistics, and
calibrated p-values for each trait, and a total R2 for covariates, latent
factors, and residual variation) and `<out>.summary.txt` with a summary of run
arguments and details.
