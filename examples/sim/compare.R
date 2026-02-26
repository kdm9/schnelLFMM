library(tidyverse)

args = commandArgs(trailingOnly=TRUE)
args = list("ath_sim_causal.tsv", "out_ath_k12.tsv")
source("~/work/src/gwashelpers/R/localscore.R")
source("~/work/src/kdm9.R/R/kview.R")

sim = read_tsv(args[[1]], col_types=cols(chr="c")) |>
    glimpse()
gwas = read_tsv(args[[2]], col_types=cols(chr="c", snp_id="c", .default="d")) |>
    glimpse()

sim |>
    count(trait)

res = gwas |>
    pivot_longer(-c(chr:snp_id, starts_with("r2_")), names_to=c("metric", "trait"), names_pattern="(p|beta|t)_(.+)") |>
    pivot_wider(names_from="metric") |>
    #group_by(trait) |>
    #group_modify(function(df, key) {
    #    bind_cols(df, local_score(df$p, xi=3))
    #}) |>
    #ungroup() |>
    full_join(sim, by=join_by(trait, chr, pos==position)) |>
    mutate(causal=!is.na(effect_size)) |>
    glimpse()

res |> count(causal)


ggplot(res |> filter(p<0.01) |> arrange(causal), aes(x=pos, y=-log10(p))) +
    geom_point(aes(colour=causal, size=causal)) +
    scale_colour_manual(values=c("black", "red")) +
    facet_grid(trait~chr, space="free_x", scales="free") +
    theme_bw()


ggplot(res, aes(x=effect_size, y=-log10(p))) +
    geom_point() +
    theme_bw()

ggplot(res, aes(x=effect_size, y=t)) +
    geom_point() +
    theme_bw()


res |>
    filter(!is.na(lindley_window)|!is.na(effect_size)) |>
    kview()
    
