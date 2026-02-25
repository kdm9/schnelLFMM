library(tidyverse)

sim = read_tsv("massive_causal.tsv") |>
    glimpse()

gwas = read_tsv("massive_gwas.tsv") |>
    pivot_longer(-c(chr:snp_id), names_to=c("metric", "trait"), names_sep="_") |>
    pivot_wider(names_from="metric") |>
    glimpse()

res = gwas |>
    inner_join(sim, by=join_by(trait, chr, pos==position, snp_id)) |>
    glimpse()

ggplot(res, aes(x=pos, y=-log10(p))) +
    geom_point() +
    facet_grid(~chr, space="free_x", scales="free_x") +
    theme_bw()

ggplot(res, aes(x=effect_size, y=-log10(p))) +
    geom_point() +
    theme_bw()

ggplot(res, aes(x=effect_size, y=t)) +
    geom_point() +
    theme_bw()
