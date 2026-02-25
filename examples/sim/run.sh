python3 simulate_gwas.py \
    --out massive \
    --traits traits.csv \
    --n-samples 200 \
    --n-snps 10000000 \
    --k-pops 5 \
    --fst 0.3 \
    --pop-props 0.01,0.04,0.15,0.3,0.5 \
    --sim-method mvnorm
