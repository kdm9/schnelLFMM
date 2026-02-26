python3 simulate_gwas.py \
    --out ath_sim \
    --traits traits.csv \
    --n-samples 200 \
    --n-snps 1000000 \
    --k-pops 5 \
    --fst 0.3 \
    --pop-props 0.01,0.04,0.15,0.3,0.5 \
    --sim-method real_genotypes \
    --bed ../ath/1k1g.bed

LFMM_K=6

cargo build --release --bin schnellfmm

time ../../target/release/schnellfmm \
    --bed massive.bed \
    --cov massive_phenotypes.tsv \
    -k $LFMM_K \
    --out out_massive_k$LFMM_K \
    -t 12 \
    --norm center-only

time Rscript compare.R $LFMM_K
