# wget -O 261.hdf5.1 https://aragwas.1001genomes.org/api/studies/261/download/
# wget -O 1001genomes_snp-short-indel_only_ACGTN.vcf.gz https://1001genomes.org/data/GMI-MPI/releases/current/1001genomes_snp-short-indel_only_ACGTN.vcf.gz


plink1.9 --make-bed  \
    --vcf 1001genomes_snp-short-indel_only_ACGTN.vcf.gz \
    --set-missing-var-ids '@:#' \
    --out 1k1g \
    --maf 0.05 \
    --double-id \
    --biallelic-only

comm -12 \
    <(cut -f 1 pheno.tsv | tail -n +2 | grep -v 8424  | sort  -u ) \
    <(cut -f 1 1k1g.fam -d ' ' | sort -u) \
    >intersection.samples

csvtk grep -f 1 -P intersection.samples pheno.csv | csvtk sort -k 1:n -T > pheno.tsv
wc -l pheno.tsv

paste intersection.samples intersection.samples > intersection.fam
wc -l intersection.fam
rm -v 1k1g_intersect*
plink1.9 --bfile 1k1g --out 1k1g_intersect --keep intersection.fam --indep-pairwise 1000 500 0.4 --make-bed
plink1.9 --bfile 1k1g_intersect --out 1k1g_intersect_ldprune --extract 1k1g_intersect.prune.in --make-bed
plink1.9 --recode ped --out 1k1g_intersect_ldprune --bfile 1k1g_intersect_ldprune

cargo build --release --bin lfmm2
time ../target/release/lfmm2 --bed 1k1g_intersect_ldprune.bed --cov pheno.tsv -k 5 --out ath_ft10_rs_k5 -t 12
time ../target/release/lfmm2 --bed 1k1g_intersect_ldprune.bed --cov pheno.tsv -k 20 --out ath_ft10_rs_k20 -t 12
time ../target/release/lfmm2 --bed 1k1g_intersect_ldprune.bed --cov pheno.tsv -k 12 --out ath_ft10_rs_k12 -t 12
