library(tidyverse)
library(rhdf5)

h5f = H5Fopen("261.hdf5")

aragwas = dplyr::bind_rows(h5f$pvalues, .id = "chrom") |> 
    mutate(across(beta:variance_explained, as.numeric)) |> 
    glimpse()

lfmmooc = read_tsv("ath_ft10_rs_k5.tsv") |>
    mutate(chrom=sprintf("chr%d", chr)) |>
    glimpse()

both = lfmmooc |>
    transmute(chrom, pos, score_rs=-log10(p_FT10)) |>
    inner_join(
        aragwas |> select(chrom, pos=positions, score_aragwas=scores),
        by=join_by(chrom, pos),
    ) |> glimpse()

ggplot(both, aes(x=score_aragwas, y=score_rs)) +
    geom_point()

sig = log10(nrow(both))
sig = 4
both |>
    pivot_longer(starts_with("score_")) |>
    ggplot(aes(x=pos, y=value, colour=name, alpha=value<sig)) +
        geom_point() +
        scale_alpha_manual(values=c(1, 0.1)) +
        scale_colour_manual(values=c("blue", "red")) +
        facet_grid(name~chrom, space="free_x", scale="free_x") +
        theme_bw()

both |>
    pivot_longer(starts_with("score_")) |>
    filter(chrom=="chr1", pos < 25e6, pos > 22e6) |>
    ggplot(aes(x=pos, y=value, colour=name, alpha=value<sig)) +
        geom_point() +
        scale_alpha_manual(values=c(1, 0.8)) +
        scale_colour_manual(values=c("blue", "red")) +
        facet_grid(name~chrom, space="free_x", scale="free_x") +
        theme_bw()
