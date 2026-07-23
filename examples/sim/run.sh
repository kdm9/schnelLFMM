xz -dc ath_sim_genes_simple.gff.xz > ath_sim_genes.gff
wc -l ath_sim_genes.gff

python3 simulate_gwas.py \
    --out ath_sim \
    --traits traits.csv \
    --sim-method real_genotypes \
    --causal-regions ath_sim_genes.gff \
    --bed ../ath/1k1g.bed

cargo build --release 
LFMM_K=12
time ../../target/release/schnellfmm \
    --bed ../ath/1k1g.bed \
    --nmf-impute \
    --nmf-iter 30 \
    --est-rate 0.05 \
    --cov ath_sim_phenotypes.tsv \
    -k $LFMM_K \
    --out out_ath_k$LFMM_K \
    -t 8 \
    --norm eigenstrat

time Rscript compare.R ath_sim_causal.tsv out_ath_k12.tsv
