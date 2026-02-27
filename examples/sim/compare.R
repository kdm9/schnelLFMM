library(tidyverse)

args = commandArgs(trailingOnly=TRUE)
#args = list("ath_sim_causal.tsv", "out_ath_k12.tsv")

base = sprintf("out_%s", sub("_causal.tsv", "", args[[1]]))

sim = read_tsv(args[[1]], col_types=cols(chr="c")) |>
    glimpse()
gwas = read_tsv(args[[2]], col_types=cols(chr="c", snp_id="c", .default="d")) |>
    glimpse()

sim |>
    count(trait)

res = gwas |>
    pivot_longer(-c(chr:snp_id, starts_with("r2_")), names_to=c("metric", "trait"), names_pattern="(p|beta|t)_(.+)") |>
    pivot_wider(names_from="metric") |>
    full_join(sim, by=join_by(trait, chr, pos==position)) |>
    mutate(causal=!is.na(effect_size)) |>
    glimpse()

ggplot(res |> filter(p<0.01) |> arrange(causal), aes(x=pos, y=-log10(p))) +
    geom_point(size=0.5, alpha=0.3) +
    geom_point(aes(size=abs(effect_size)^2), colour="red") +
    facet_grid(trait~chr, space="free_x", scales="free_x") +
    labs(size="Effect Magnitude") +
    theme_bw()
ggsave(sprintf("%s_manhattan.pdf", base), width=10, height=10, dpi=600)
ggsave(sprintf("%s_manhattan.png", base), width=10, height=10, dpi=600)

ggplot(res, aes(x=effect_size, y=-log10(p))) +
    geom_point() +
    geom_hline(yintercept=3) +
    theme_bw()
ggsave(sprintf("%s_eff-v-p.pdf", base), width=8, height=5, dpi=600)
ggsave(sprintf("%s_eff-v-p.png", base), width=8, height=5, dpi=600)
