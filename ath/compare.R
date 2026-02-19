library(tidyverse)
library(rhdf5)
library(LEA)
LFMM_K = 1

# --- LEA LFMM2 ---
lea_lfmm_file = sprintf("lea_lfmm2_k%d.tsv", LFMM_K)
if (!file.exists(lea_lfmm_file)) {

    message("Running LEA lfmm2 (K=", LFMM_K, ")...")
    lfmm_file = "1k1g_intersect_ldprune.lfmm"
    if (!file.exists(lfmm_file)) {
        message("  ped2lfmm")
        lfmm_file = ped2lfmm("1k1g_intersect_ldprune.ped")
    }

    message("  read.lfmm")

    Y = read.lfmm(lfmm_file)
    Y[Y==9 | Y==-9] = NA
    col_means = colMeans(Y, na.rm = TRUE)
    na_mask = is.na(Y)
    Y[na_mask] = col_means[col(Y)[na_mask]]
    table(is.na(Y))

    pheno = read.delim("pheno.tsv")
    X = pheno$FT10

    message("  lfmm2")
    mod = lfmm2(Y, X, K = LFMM_K)
    message("  lfmm2.test")
    pv = lfmm2.test(mod, Y, X, full = F)$pvalues
    # pv is n_snps x 1 matrix

    message("  lfmm2 GIF")
    median(qchisq(pv, df=1, lower.tail=FALSE)) / qchisq(0.5, df=1)

    bim = read.delim("1k1g_intersect_ldprune.bim", header = FALSE,
                      col.names = c("chr", "snp_id", "cm", "pos", "a1", "a2"))
    lea_results = tibble(chr = bim$chr, pos = bim$pos, p_lea = as.numeric(t(pv)))
    write_tsv(lea_results, lea_lfmm_file)
    message("LEA results written to ", lea_lfmm_file)
} else {
    message("Loading cached LEA results from ", lea_lfmm_file)
}
lea_results = read_tsv(lea_lfmm_file) |>
    mutate(chrom = sprintf("chr%d", chr)) |>
    glimpse()

# --- AraGWAS ---
h5f = H5Fopen("261.hdf5")

aragwas = dplyr::bind_rows(h5f$pvalues, .id = "chrom") |>
    mutate(across(beta:variance_explained, as.numeric)) |>
    glimpse()

# --- Rust LFMM-OOC ---
lfmmooc = read_tsv(sprintf("ath_ft10_rs_k%d.tsv", LFMM_K)) |>
    mutate(chrom=sprintf("chr%d", chr)) |>
    glimpse()

# --- Join all three ---
all_results = lfmmooc |>
    transmute(chrom, pos, score_rs=-log10(p_FT10)) |>
    full_join(
        aragwas |> select(chrom, pos=positions, score_aragwas=scores),
        by=join_by(chrom, pos),
    ) |>
    left_join(
        lea_results |> transmute(chrom, pos, score_lea=-log10(p_lea)),
        by=join_by(chrom, pos),
    ) |> glimpse()

sig = log10(nrow(all_results))
sig = 3
all_results |>
    pivot_longer(starts_with("score_")) |>
    ggplot(aes(x=pos, y=value, colour=name, alpha=value<sig)) +
        geom_point() +
        scale_alpha_manual(values=c(1, 0.1)) +
        facet_grid(name~chrom, space="free_x", scale="free_x") +
        theme_bw()

all_results |>
    pivot_longer(starts_with("score_")) |>
    filter(chrom=="chr1", pos < 25e6, pos > 24e6) |>
    ggplot(aes(x=pos, y=value, colour=name, alpha=value<sig)) +
        geom_point() +
        scale_alpha_manual(values=c(1, 0.8)) +
        facet_grid(name~chrom, space="free_x", scale="free_x") +
        theme_bw()


all_results |>
    pivot_longer(starts_with("score_")) |>
    filter(chrom=="chr5", pos >= 17e6, pos <= 19e6) |>
    ggplot(aes(x=pos, y=value, colour=name, alpha=value<sig)) +
        geom_point() +
        scale_alpha_manual(values=c(1, 0.8)) +
        facet_grid(name~chrom, space="free_x", scale="free_x") +
        theme_bw()

all_results |>
    pivot_longer(starts_with("score_")) |>
    filter(chrom=="chr5", pos >= 17e6, pos <= 19e6) |>
    ggplot(aes(x=pos, y=value, colour=name, alpha=value<sig)) +
        geom_point() +
        scale_alpha_manual(values=c(1, 0.8)) +
        facet_grid(name~chrom, space="free_x", scale="free_x") +
        theme_bw()
