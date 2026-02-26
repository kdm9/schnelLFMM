python3 simulate_gwas.py \
    --out ath_sim \
    --traits traits.csv \
    --sim-method real_genotypes \
    --bed ../ath/1k1g.bed

cargo build --release --bin schnellfmm

LFMM_K=12
time ../../target/release/schnellfmm \
    --bed ../ath/1k1g.bed \
    --cov ath_sim_phenotypes.tsv \
    -k $LFMM_K \
    --out out_ath_k$LFMM_K \
    -t 12 \
    --norm center-only

time Rscript compare.R ath_sim_causal.tsv out_ath_k12.tsv

time ../../target/release/schnellfmm \
    --bed massive.bed \
    --cov massive_phenotypes.tsv \
    -k $LFMM_K \
    --out out_massive_k$LFMM_K \
    -t 12 \
    --norm center-only

