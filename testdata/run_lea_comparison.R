#!/usr/bin/env Rscript
# Comprehensive validation: Rust LFMM2 vs LEA (Bioconductor) lfmm2()
#
# Produces:
#   - QQ plots (LEA vs expected, Rust vs expected, Rust vs LEA)
#   - Power/recall, FDR, TDR, F1 at fixed FDR thresholds (1%, 5%, 10%)
#   - Spearman rank correlations
#
# Usage: cd testdata && Rscript run_lea_comparison.R

library(LEA)

prefix <- "sim"
K <- 5

# ============================================================
# 1. Load data
# ============================================================
cat("Loading data...\n")
geno  <- read.table(paste0(prefix, ".lfmm"), header = FALSE)
env   <- read.table(paste0(prefix, "_covariates.txt"), header = TRUE)
truth <- read.table(paste0(prefix, "_truth.tsv"), header = TRUE, sep = "\t")

is_causal <- truth$is_causal == 1
n_samples <- nrow(geno)
n_snps    <- ncol(geno)
n_causal  <- sum(is_causal)
d         <- ncol(env)

cat(sprintf("  %d samples, %d SNPs, %d causal, %d covariates\n",
            n_samples, n_snps, n_causal, d))

# ============================================================
# 2. Run LEA lfmm2
# ============================================================
cat(sprintf("Running LEA lfmm2 with K = %d ...\n", K))
write.lfmm(as.matrix(geno), paste0(prefix, "_lea.lfmm"))

mod <- lfmm2(input = paste0(prefix, "_lea.lfmm"),
             env   = as.matrix(env),
             K     = K)

pv <- lfmm2.test(object = mod,
                  input  = paste0(prefix, "_lea.lfmm"),
                  env    = as.matrix(env),
                  full   = FALSE)

lea_pv <- as.data.frame(t(pv$pvalues))
colnames(lea_pv) <- paste0("p_", seq_len(d) - 1)
write.table(lea_pv, paste0(prefix, "_lea_pvalues.tsv"),
            sep = "\t", row.names = FALSE, col.names = TRUE)

lea_zs <- as.data.frame(t(pv$zscores))
colnames(lea_zs) <- paste0("z_", seq_len(d) - 1)
write.table(lea_zs, paste0(prefix, "_lea_zscores.tsv"),
            sep = "\t", row.names = FALSE, col.names = TRUE)

cat("LEA results written.\n")

# ============================================================
# 3. Load Rust results
# ============================================================
rust_pv_file <- paste0(prefix, "_rust_pvalues.tsv")
if (!file.exists(rust_pv_file)) {
    stop(paste("Rust p-values not found:", rust_pv_file,
               "\nRun the Rust integration test first."))
}
rust_pv <- read.table(rust_pv_file, header = TRUE, sep = "\t")
cat("Rust results loaded.\n")

# ============================================================
# 4. QQ plots
# ============================================================
qq_expected <- function(pvals, main, col = "black") {
    pvals <- pvals[!is.na(pvals)]
    n     <- length(pvals)
    expected <- -log10(rev(1:n/(n+1)))
    observed <- sort(-log10(pvals))
    lim <- max(expected, observed)
    plot(expected, observed, pch = 20, cex = 0.3, col = col,
         xlab = expression(-log[10](p[expected])),
         ylab = expression(-log[10](p[observed])),
         main = main, xlim = c(0, lim), ylim = c(0, lim))
    abline(0, 1, col = "red", lty = 2)
    # GIF (genomic inflation factor) from median chi-sq
    chisq <- qchisq(pvals, df = 1, lower.tail = FALSE)
    gif <- median(chisq, na.rm = TRUE) / qchisq(0.5, df = 1)
    legend("topleft", legend = sprintf("GIF = %.3f", gif), bty = "n", cex = 0.9)
}


qq_vs <- function(pvals_x, pvals_y, xlab_text, ylab_text, main) {
    x <- -log10(sort(pvals_x))
    y <- -log10(sort(pvals_y))
    lim <- max(c(x, y), na.rm = TRUE)
    plot(x, y, pch = 20, cex = 0.3,
         xlab = xlab_text, ylab = ylab_text,
         main = main, xlim = c(0, lim), ylim = c(0, lim))
    abline(0, 1, col = "red", lty = 2)
    rc <- cor(x, y, method = "spearman")
    legend("topleft", legend = sprintf("rho = %.4f", rc), bty = "n", cex = 0.9)
}

pdf_file <- paste0(prefix, "_validation_plots.pdf")
pdf(pdf_file, width = 12, height = 4 * d)
par(mfrow = c(d, 3), mar = c(4, 4, 3, 1))

for (j in 1:d) {
    qq_expected(lea_pv[, j],
                paste0("LEA QQ - Covariate ", j),
                col = "steelblue")
    qq_expected(rust_pv[, j],
                paste0("Rust QQ - Covariate ", j),
                col = "darkorange")
    qq_vs(lea_pv[, j], rust_pv[, j],
          expression(-log[10](p[LEA])),
          expression(-log[10](p[Rust])),
          paste0("Rust vs LEA - Covariate ", j))
}
dev.off()
cat(sprintf("QQ plots saved to %s\n", pdf_file))

# ============================================================
# 5. Discovery metrics at fixed FDR thresholds
# ============================================================
compute_metrics <- function(pvals, is_causal, fdr_cut) {
    qvals      <- p.adjust(pvals, method = "BH")
    discovered <- qvals < fdr_cut
    tp  <- sum( discovered &  is_causal)
    fp  <- sum( discovered & !is_causal)
    fn_ <- sum(!discovered &  is_causal)

    n_disc <- tp + fp
    power  <- if (sum(is_causal) > 0) tp / sum(is_causal) else NA   # recall
    fdr_obs <- if (n_disc > 0)  fp / n_disc            else 0       # FP / (TP+FP)
    tdr     <- if (n_disc > 0)  tp / n_disc            else NA      # precision = 1-FDR
    f1      <- if (!is.na(power) && !is.na(tdr) && (power + tdr) > 0)
                   2 * power * tdr / (power + tdr) else 0

    data.frame(fdr_cut = fdr_cut, n_disc = n_disc,
               tp = tp, fp = fp, fn_ = fn_,
               power = power, fdr_obs = fdr_obs, tdr = tdr, f1 = f1)
}

fdr_thresholds <- c(0.01, 0.05, 0.10)

cat("\n============================================================\n")
cat("Power vs. FDR at fixed BH thresholds\n")
cat("  Power  = TP / (TP + FN)           (recall)\n")
cat("  TDR    = TP / (TP + FP)           (precision = 1 - FDR)\n")
cat("  F1     = 2 * Power * TDR / (Power + TDR)\n")
cat("============================================================\n")

metrics_all <- data.frame()

for (j in 1:d) {
    cat(sprintf("\n--- Covariate %d ---\n", j))

    for (method_name in c("Rust", "LEA")) {
        pvals <- if (method_name == "Rust") rust_pv[, j] else lea_pv[, j]

        cat(sprintf("\n  %s:\n", method_name))
        cat(sprintf("  %-10s %7s %5s %5s %5s %8s %8s %8s %8s\n",
                    "FDR_cut", "n_disc", "TP", "FP", "FN", "Power", "FDR_obs", "TDR", "F1"))

        for (thr in fdr_thresholds) {
            m <- compute_metrics(pvals, is_causal, thr)
            cat(sprintf("  %-10.2f %7d %5d %5d %5d %8.4f %8.4f %8.4f %8.4f\n",
                        m$fdr_cut, m$n_disc, m$tp, m$fp, m$fn_,
                        m$power, m$fdr_obs, m$tdr, m$f1))
            m$method    <- method_name
            m$covariate <- j
            metrics_all <- rbind(metrics_all, m)
        }
    }
}

# Write metrics table
write.table(metrics_all,
            paste0(prefix, "_validation_metrics.tsv"),
            sep = "\t", row.names = FALSE, col.names = TRUE)

# ============================================================
# 6. Rank correlations
# ============================================================
cat("\n============================================================\n")
cat("Spearman rank correlations of -log10(p): Rust vs LEA\n")
cat("============================================================\n")
for (j in 1:d) {
    rc <- cor(rank(-log10(rust_pv[, j])),
              rank(-log10(lea_pv[, j])),
              method = "spearman")
    cat(sprintf("  Covariate %d: rho = %.4f\n", j, rc))
}

cat("And just to check, did we stuff up and use the same file twice? all(rust pv == lea pv):")
table(rust_pv == lea_pv)
cat("\nhead(rust_pv):\n")
head(rust_pv)
cat("head(lea_pv):\n")
head(lea_pv)

cat(sprintf("\nAll outputs written with prefix '%s'.\n", prefix))
