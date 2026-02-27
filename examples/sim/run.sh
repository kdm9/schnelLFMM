awk '$3 == "gene"{print $0}' \
    < ~/ws/data/tair10/Araport11_GFF3_genes_transposons.current.gff \
    | grep 'locus_type=protein_coding' \
    | sed -e 's/^Chr//' \
    > ath_sim_genes.gff
wc -l ath_sim_genes.gff

python3 simulate_gwas.py \
    --out ath_sim \
    --traits traits.csv \
    --sim-method real_genotypes \
    --causal-regions ath_sim_genes.gff \
    --bed ../ath/1k1g.bed

cargo build --release --bin schnellfmm

LFMM_K=12
time ../../target/release/schnellfmm \
    --bed ../ath/1k1g.bed \
    --cov ath_sim_phenotypes.tsv \
    -k $LFMM_K \
    --out out_ath_k$LFMM_K \
    -t 12 \
    --norm eigenstrat

time Rscript compare.R ath_sim_causal.tsv out_ath_k12.tsv
