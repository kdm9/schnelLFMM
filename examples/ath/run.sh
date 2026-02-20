LFMM_K=12
if [ ! -f 261.hdf5 ]
then
    wget -O 261.hdf5 https://aragwas.1001genomes.org/api/studies/261/download/
fi

if [ ! -f 1001genomes_snp-short-indel_only_ACGTN.vcf.gz ]
then
    wget -O 1001genomes_snp-short-indel_only_ACGTN.vcf.gz https://1001genomes.org/data/GMI-MPI/releases/current/1001genomes_snp-short-indel_only_ACGTN.vcf.gz
fi

if [ ! -f 1k1g.bed ]
then
    plink1.9 --make-bed  \
        --vcf 1001genomes_snp-short-indel_only_ACGTN.vcf.gz \
        --set-missing-var-ids '@:#' \
        --out 1k1g \
        --maf 0.05 \
        --double-id \
        --biallelic-only
fi

# 8424 is excluded as it's there twice
comm -12 \
    <(cut -f 1 pheno.tsv | tail -n +2 | grep -v 8424  | sort  -u ) \
    <(cut -f 1 1k1g.fam -d ' ' | sort -u) \
    >intersection.samples

csvtk tab2csv  pheno.tsv | \
    csvtk grep -f 1 -P intersection.samples | \
    csvtk sort -k 1:n -T > pheno_intersect.tsv
wc -l pheno_intersect.tsv

paste intersection.samples intersection.samples > intersection.fam
wc -l intersection.fam
plink1.9 --bfile 1k1g --out 1k1g_intersect --keep intersection.fam --indep-pairwise 1000 500 0.4 --make-bed
plink1.9 --bfile 1k1g_intersect --out 1k1g_intersect_ldprune --extract 1k1g_intersect.prune.in --make-bed
plink1.9 --recode ped --out 1k1g_intersect_ldprune --bfile 1k1g_intersect_ldprune

cargo build --release --bin schnelfmm
time ../../target/release/schnelfmm --bed 1k1g_intersect_ldprune.bed --cov pheno_intersect.tsv -k $LFMM_K --out ath_ft10_rs_k$LFMM_K -t 12
time Rscript compare.R $LFMM_K
