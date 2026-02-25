python3 simulate_gwas.py \
    --out massive \
    --traits traits.csv \
    --n-samples 200 \
    --n-snps 10000000 \
    --k-pops 5 \
    --fst 0.3 \
    --pop-props 0.01,0.04,0.15,0.3,0.5 \
    --sim-method mvnorm

LFMM_K=20

cargo build --release --bin schnellfmm

time ../../target/release/schnellfmm \
    --bed massive.bed \
    --cov massive_phenotypes.tsv \
    -k $LFMM_K \
    --out out_massive_k$LFMM_K \
    -t 12 \
    --norm center-only

time Rscript compare.R $LFMM_K
