#!/usr/bin/env Rscript
#
# Cross-validation of genotype imputation methods against held-out genotypes.
#
# Compares four approaches on the same simulated admixed data:
#   1. Mean imputation (R baseline)
#   2. LEA::snmf (R reference NMF)
#   3. Rust schnellfmm --mean-impute
#   4. Rust schnellfmm --nmf-impute
#
# Usage:
#   Rscript tests/nmf_validation.R [--rust-bin <path-to-schnellfmm>]

suppressMessages({
    library(LEA)
    library(MCMCpack)
})

args <- commandArgs(trailingOnly = TRUE)
rust_bin <- if (length(args) >= 2 && args[1] == "--rust-bin") args[2] else NA

# --- Parameters ---
n         <- 300
p         <- 3000
K         <- 3
miss_rate <- 0.05
seed      <- 42
set.seed(seed)

cat(sprintf("=== %d samples x %d SNPs, K=%d, %.0f%% MCAR missing ===\n\n",
    n, p, K, 100 * miss_rate))

# --- Simulate admixed genotypes ---
cat("Simulating ... ")
f <- matrix(rbeta(K * p, 0.15, 0.15), K, p)
f[f < 0.005] <- 0.005; f[f > 0.995] <- 0.995
pi_true <- rdirichlet(n, rep(0.3, K))
pmat <- pi_true %*% f
Y <- matrix(rbinom(n * p, 2, as.vector(pmat)), n, p)

mask <- matrix(runif(n * p) < miss_rate, n, p)
Y_miss <- Y; Y_miss[mask] <- NA
true_vals <- Y[mask]
n_masked <- sum(mask)
cat(sprintf("done (%d held out)\n", n_masked))

# --- Shared helpers ---
mae_mean <- function(Ymiss, mask, true_vals) {
    yf <- Ymiss
    for (j in 1:ncol(Ymiss)) {
        m <- mean(Ymiss[, j], na.rm = TRUE)
        yf[is.na(Ymiss[, j]), j] <- m
    }
    mean(abs(true_vals - yf[mask]))
}

write_geno <- function(Ymiss, path) {
    conn <- file(path, "w")
    for (j in 1:ncol(Ymiss)) {
        line <- Ymiss[, j]; line[is.na(line)] <- 9
        writeLines(paste(as.integer(line), collapse = ""), conn)
    }
    close(conn)
}

write_plink_bed <- function(Ymat, stem) {
    n <- nrow(Ymat); p <- ncol(Ymat)
    bed_con <- file(paste0(stem, ".bed"), "wb")
    writeBin(as.raw(c(0x6C, 0x1B, 0x01)), bed_con)
    mult <- as.integer(c(1, 4, 16, 64))
    for (snp in 1:p) {
        bytes <- raw(ceiling(n / 4))
        for (i in 0:(n - 1)) {
            byte_idx <- i %/% 4
            pos <- (i %% 4) + 1
            g <- Ymat[i + 1, snp]
            code <- if (g == 0L) 0L else if (g == 1L) 2L else if (g == 2L) 3L else 1L
            bytes[byte_idx + 1] <- as.raw(as.integer(bytes[byte_idx + 1]) + code * mult[pos])
        }
        writeBin(bytes, bed_con)
    }
    close(bed_con)
    write.table(data.frame(chr = rep(1, p), snp = sprintf("snp_%d", 1:p),
        cm = 0, pos = 1:p, a1 = "A", a2 = "G"),
        paste0(stem, ".bim"), quote = FALSE, row.names = FALSE,
        col.names = FALSE, sep = "\t")
    write.table(data.frame(fid = sprintf("s%d", 1:n), iid = sprintf("s%d", 1:n),
        father = 0, mother = 0, sex = 1, pheno = 0),
        paste0(stem, ".fam"), quote = FALSE, row.names = FALSE,
        col.names = FALSE, sep = "\t")
}

# === 1. Mean imputation (R baseline) ===
cat("Mean imputation ... ")
mae_mean_r <- mae_mean(Y_miss, mask, true_vals)
cat(sprintf("MAE = %.5f\n", mae_mean_r))

# === 2a. LEA snmf on COMPLETE data (no missingness) ===
# This tests whether snmf can recover the underlying admixture structure at all.
cat("LEA snmf (complete data) ...\n")
tf_complete <- tempfile(fileext = ".geno")
Y_complete <- Y  # no missingness
write_geno(Y_complete, tf_complete)

lea_c <- snmf(tf_complete, K = K, project = "new", repetitions = 1, CPU = 1,
              alpha = 0, tolerance = 1e-6, iterations = 2000, ploidy = 2,
              seed = seed)

Qlc <- Q(lea_c, K = K)
Glc_raw <- G(lea_c, K = K)
Glc <- Glc_raw[1:p, , drop = FALSE]
Y_lea_comp <- 2 * Qlc %*% t(Glc)
Y_lea_comp[Y_lea_comp < 0] <- 0; Y_lea_comp[Y_lea_comp > 2] <- 2
mae_lea_comp <- mean(abs(true_vals - Y_lea_comp[mask]))

# Also check: how well does snmf recover the TRUE admixture proportions?
# Align Q columns to pi_true columns: match highest absolute correlation
q_cors <- numeric(K)
used <- logical(K)
for (k1 in 1:K) {
    best_cor <- 0
    best_k2 <- NA
    for (k2 in 1:K) {
        if (used[k2]) next
        cc <- tryCatch(cor(pi_true[,k1], Qlc[,k2]), error = function(e) NA)
        if (is.na(cc)) cc <- 0
        if (abs(cc) > best_cor) { best_cor <- abs(cc); best_k2 <- k2 }
    }
    if (!is.na(best_k2)) {
        used[best_k2] <- TRUE
        q_cors[k1] <- best_cor
    }
}
cat(sprintf("  Q: %dx%d, G: %dx%d\n", nrow(Qlc), ncol(Qlc), nrow(Glc), ncol(Glc)))
cat(sprintf("  Q-pi correlations: %.3f %.3f %.3f\n", q_cors[1], q_cors[2], q_cors[3]))
cat(sprintf("  MAE = %.5f (complete data snmf)\n", mae_lea_comp))

# === 2b. LEA snmf on data WITH missing values ===
cat("LEA snmf (with missing) ...\n")
tf_mar <- tempfile(fileext = ".geno")
write_geno(Y_miss, tf_mar)

lea <- snmf(tf_mar, K = K, project = "new", repetitions = 1, CPU = 1,
            alpha = 0, tolerance = 1e-6, iterations = 2000, ploidy = 2,
            seed = seed)

Ql <- Q(lea, K = K)
Gl_raw <- G(lea, K = K)
Gl <- Gl_raw[1:p, , drop = FALSE]
Y_lea_snmf <- 2 * Ql %*% t(Gl)
Y_lea_snmf[Y_lea_snmf < 0] <- 0; Y_lea_snmf[Y_lea_snmf > 2] <- 2
mae_lea_miss <- mean(abs(true_vals - Y_lea_snmf[mask]))
cat(sprintf("  MAE = %.5f (with-missing snmf)\n", mae_lea_miss))

# === 3. Rust schnellfmm ===
mae_rust_mean  <- NA
mae_rust_nmf   <- NA
nmf_iters      <- NA

if (!is.na(rust_bin) && file.exists(rust_bin)) {
    tf_stem <- file.path(tempdir(), "lfmmv")

    # Write complete data as .bed (no missingness — CV is done internally by Rust)
    write_plink_bed(Y, tf_stem)
    cov_path <- paste0(tf_stem, "_cov.csv")
    write.table(data.frame(id = sprintf("s%d", 1:n), trait = rnorm(n)),
        cov_path, quote = FALSE, row.names = FALSE, sep = ",")

    # --- Rust with NMF ---
    cat("Rust NMF ...\n")
    system2(rust_bin, c(
        "-b", paste0(tf_stem, ".bed"),
        "-c", cov_path,
        "-k", K,
        "--nmf-impute", "--nmf-iter", "3", "--nmf-cv-rate", "0.05",
        "--norm", "center-only",
        "-o", paste0(tf_stem, "_nmf"),
        "-t", "2"
    ), stdout = FALSE, stderr = FALSE)

    summ_file <- paste0(tf_stem, "_nmf.summary.txt")
    if (file.exists(summ_file)) {
        lines <- readLines(summ_file)
        for (line in lines) {
            if (grepl("nmf_cv_mae_iter_", line)) {
                parts <- strsplit(line, ": ")[[1]]
                mae_rust_nmf <- as.numeric(parts[2])
                nmf_iters <- max(nmf_iters, as.integer(sub(".*_(\\d+)$", "\\1", parts[1])), na.rm=TRUE)
            }
            if (grepl("mean_impute_cv_mae:", line)) {
                mae_rust_mean <- as.numeric(strsplit(line, ": ")[[1]][2])
            }
        }
    }
} else {
    cat(sprintf("Rust binary not found at '%s', skipping Rust comparison.\n",
        if (is.na(rust_bin)) "(not specified)" else rust_bin))
}

# === Summary Table ===
cat("\n")
cat(sprintf("%-24s %9s %9s\n", "Method", "MAE", "vs_R_mean%"))
cat(sprintf("%-24s %9s %9s\n", "------------------------", "---------", "---------"))
cat(sprintf("%-24s %9.5f %9s\n", "R mean imputation", mae_mean_r, "–"))

# LEA
if (!is.na(mae_lea_comp)) {
    pct <- (1 - mae_lea_comp / mae_mean_r) * 100
    cat(sprintf("%-24s %9.5f %+9.1f%%\n",
        "LEA snmf (complete data)", mae_lea_comp, pct))
}
if (!is.na(mae_lea_miss)) {
    pct <- (1 - mae_lea_miss / mae_mean_r) * 100
    cat(sprintf("%-24s %9.5f %+9.1f%%\n",
        "LEA snmf (with missing)", mae_lea_miss, pct))
}

# Rust
if (!is.na(mae_rust_mean)) {
    cat(sprintf("%-24s %9.5f %9s\n",
        "Rust mean impute (CV)", mae_rust_mean, "–"))
}
if (!is.na(mae_rust_nmf)) {
    pct <- (1 - mae_rust_nmf / mae_rust_mean) * 100
    cat(sprintf("%-24s %9.5f %+9.1f%%\n",
        "Rust NMF (CV)", mae_rust_nmf, pct))
}

cat("\nNote: Rust NMF is trained on raw dosages and its CV metrics are on the raw {0,1,2} scale,\n")
cat("so they are directly comparable to the R metrics above (also raw dosages).\n")
