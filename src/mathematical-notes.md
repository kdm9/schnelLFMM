# Mathematical Notes: LFMM2 with Latent-Factor Imputation

## 1. Genotype Encoding and Preprocessing

### 1.1 PLINK BED Format

Genotypes are stored in PLINK 1 BED format with the standard 2-bit encoding:

| PLINK Code | Genotype       | Decoded Value |
|------------|---------------|---------------|
| `00`       | Homozygous reference | 0 |
| `01`       | Missing              | `NaN` |
| `10`       | Heterozygous         | 1 |
| `11`       | Homozygous alternate | 2 |

A BED file consists of a 3-byte magic header (`0x6C 0x1B 0x01`), followed by SNP-major packed data. Each SNP occupies $\lceil n_{\text{phys}}/4 \rceil$ bytes where $n_{\text{phys}}$ is the number of physical samples on disk. Optionally, the software subsets to $n$ output samples via `sample_keep`.

### 1.2 Genotype Normalization

Each SNP column $y_j \in \mathbb{R}^n$ (after raw decode to $\{0, 1, 2\}$) is centered and optionally scaled. Let $\bar{y}_j$ be the column mean computed over non-missing entries.

**Mean-based centering:**
$$
\tilde{y}_{ij} = y_{ij} - \bar{y}_j
$$

**Eigenstrat normalization:** additionally divide by $\sqrt{2 p_j (1-p_j)}$ where $p_j = \bar{y}_j / 2$:

$$
\tilde{y}_{ij}^{\text{eigen}} = \frac{y_{ij} - 2p_j}{\sqrt{2p_j(1-p_j)}}
$$

This yields approximately unit-variance genotypes under Hardy-Weinberg equilibrium. Monomorphic SNPs ($2p(1-p) < 10^{-10}$) are set to zero to avoid division by zero.

### 1.3 Missing Data Strategies

The software supports two imputation strategies for handling missing genotypes (NaN entries in the raw decode):

1. **Mean imputation** (`ImputeConfig::Mean`): Missing values are filled with the column mean before centering — after centering, these become exactly 0.

2. **Latent-factor imputation** (`ImputeConfig::LatentFill`): Missing values are filled with their fitted value under the latent-factor term, $\text{clamp}(\bar{y}_j + (P_{\hat{U}}(Y^{\text{filled}} - \bar{y}))_{ij}, 0, 2)$, before centering (see §4 and §8).

---

## 2. Covariate Preprocessing

Given a covariate matrix $X \in \mathbb{R}^{n \times d}$:

$$
X \leftarrow \text{center}(X)
$$

Optionally, if `--scale-cov` is set, each column is further divided by its standard deviation:

$$
x_{ij} \leftarrow \frac{x_{ij} - \bar{x}_j}{\text{sd}(x_j)}
$$

---

## 3. Step 0: Precomputation of X-Related Quantities

Let $X \in \mathbb{R}^{n \times d}$ be the centered (and optionally scaled) covariate matrix. Perform:

### 3.1 SVD of X

Compute the full SVD:

$$
X = Q \Sigma R^T
$$

where $Q \in \mathbb{R}^{n \times n}$ is orthogonal, $\Sigma$ contains singular values $\sigma_1, \ldots, \sigma_d$ (for $n > d$, only $\min(n, d)$ are non-zero), and $R \in \mathbb{R}^{d \times d}$ is orthogonal.

### 3.2 Ridge-Regularized Weight Matrix

For ridge penalty $\lambda > 0$, define the diagonal matrix $D_\lambda \in \mathbb{R}^{n \times n}$ with entries:

$$
d_\lambda[j] = \begin{cases}
\sqrt{\frac{\lambda}{\lambda + \sigma_j^2}}, & j < \min(n, d) \\
1, & j \geq d
\end{cases}
$$

and its inverse $D_\lambda^{-1}$ with entries $d_\lambda^{-1}[j] = 1 / d_\lambda[j]$.

**Note (square-root convention).** Caye et al. (2019) define $D_\lambda$ *without* the square root, $d_\lambda[j] = \lambda / (\lambda + \sigma_j^2)$, which is the weighting that arises exactly when profiling the ridge objective: $\|Y - XB^T - W\|_F^2 + \lambda\|B\|_2^2$ minimised over $B$ gives the residual maker $I - H_\lambda = Q\, \mathrm{diag}(\lambda/(\lambda+\sigma_j^2))\, Q^T$. We instead follow the LEA reference implementation (`LEA::lfmm2()`), which uses the square-root form above. At the default $\lambda = 10^{-5}$ both conventions shrink the covariate directions by many orders of magnitude and give indistinguishable results; the difference only becomes relevant for substantially larger $\lambda$.

Define the operator $M \in \mathbb{R}^{n \times n}$:

$$
M = D_\lambda Q^T
$$

which is the key matrix passed to the randomized SVD step.

### 3.3 Ridge Inverse

Precompute the ridge-regularized inverse:

$$
(X^T X + \lambda I_d)^{-1} \in \mathbb{R}^{d \times d}
$$

This is used for effect size estimation in Step 3 (Equation 16).

### 3.4 Fallback for Singular Matrices

If any matrix inversion fails (near-singularity from collinear covariates or over-specified $K$), Tikhonov regularization is added: $\varepsilon I$ where $\varepsilon = 10^{-8} \cdot \max(|\text{diag}(A)|)$.

---

## 4. Latent-Factor Imputation of Missing Genotypes

When `--latent-impute` is active, missing genotypes are filled with their fitted value under the latent-factor term of the LFMM model itself. No separate factorisation is trained: the basis is $\hat{U}$, which the pipeline already computes for Steps 1–2.

### 4.1 Model

For SNP column $j$ with per-SNP mean $\bar{y}_j$ over *observed* entries, let $Y^{\text{filled}}$ be the chunk with missing entries replaced by $\bar{y}_j$. The fill is the orthogonal projection of the centered column onto $\text{col}(\hat{U})$, returned to the dosage scale and clamped to the valid dosage range:

$$
\hat{y}_{ij} = \text{clamp}\!\left(\bar{y}_j + \left(P_{\hat{U}} \left(Y^{\text{filled}} - \bar{y}\right)\right)_{ij},\; 0,\; 2\right),
\qquad
P_{\hat{U}} = \hat{U}(\hat{U}^T \hat{U})^{-1}\hat{U}^T .
$$

$\hat{U}$ need not be orthonormalised — a projection is invariant to the choice of basis for $\text{col}(\hat{U})$ — and $(\hat{U}^T \hat{U})^{-1}$ is $K \times K$, inverted once up front rather than per chunk.

### 4.2 The basis must be $\text{col}(\hat{U})$ alone

**The projection basis must never be widened to include $X$.** Imputing with the full fitted model $[\mathbf{1} \mid X \mid \hat{U}]$ writes the effect under test into the imputed genotypes, which is circular: measured on simulated null data ($n = 400$, $F_{ST} = 0.10$, $\text{corr}(X, \text{structure}) = 0.6$, 20% missing), it drives the GIF to 1.50 and the null false-positive rate at $\alpha = 0.05$ to 0.107. Projecting onto $\text{col}(\hat{U})$ alone gives GIF 1.03 and FPR 0.050. This invariant is pinned by `test_impute_does_not_inflate_null_calibration`.

An explicit intercept column is unnecessary: the chunk is centered on $\bar{y}_j$ before projection, so the mean is already carried outside the projection.

### 4.3 One EM step

$\hat{U}$ is itself estimated from genotypes that need imputing, so the pipeline takes a single EM step:

1. Estimate $\hat{U}^{(1)}$ by RSVD (§5) with mean imputation.
2. Re-run the RSVD over the estimation subset with `LatentFill` using $\hat{U}^{(1)}$, giving $\hat{U}^{(2)}$.
3. The GWAS pass imputes from, and tests against, $\hat{U}^{(2)}$.

The fill alone (without step 2) already delivers the full imputation-accuracy gain; the re-estimation is what restores calibration. Measured GIF on the simulation above: 1.065 mean-imputed, 1.053 with fill but no re-estimation, 1.032 after re-estimation. A second EM step was tested and gains essentially nothing (1.029), so the step count is fixed at one and is not exposed as an option.

### 4.4 Why not NMF

Earlier versions trained a non-negative matrix factorisation $Y \approx W H$ on raw dosages and imputed with $W H_c$. It was removed in 0.4.0 because it performed no better than mean imputation at moderate structure ($F_{ST}$ 5–25%), for two compounding reasons:

- **Initialization.** $W^{(0)} = |\Omega V_Z[:, 1{:}K]|$ used a sketch with no power iterations, and the entrywise absolute value folds signed contrast singular vectors onto the all-positive direction. Measured, $\text{col}(W^{(0)})$ captured 0.9% of the true population structure — with $K/n = 1.25\%$, no better than $K$ random directions. The default 10 multiplicative updates reached 7.6%; roughly 100 were needed to reach 99%.
- **Parameterisation.** Non-negativity on raw dosages spends one component representing the per-SNP mean, and the $\max(0, \cdot)$ clamp biases each per-SNP fit. Handed the *true* top-$K$ subspace, the non-negative raw-scale path recovered +1.2% / +3.6% MAE over the mean baseline at $F_{ST}$ 0.05 / 0.10, versus +4.1% / +8.7% for the same subspace fit signed on centered data.

Non-negativity bought nothing downstream, since the fill is centered immediately afterwards (§1.2); it was required only by the Lee–Seung multiplicative updates. A fully converged NMF merely matched the signed rank-$K$ projection that $\hat{U}$ supplies for free.

---

## 5. Steps 1-2: Randomized SVD for Latent Factor Estimation

The LFMM2 model assumes the genotype matrix decomposes as:

$$
Y = \mathbf{1} \mu^T + X B + U V^T + E
$$

where $U \in \mathbb{R}^{n \times K}$ are unobserved latent factors, $V \in \mathbb{R}^{p \times K}$ their loadings, and $E$ is noise. To estimate $U$ without iteratively fitting $V$, the LFMM2 approach notes that under the ridge-regularized least-squares framework, the latent factors can be recovered from the top $K$ singular vectors of $M Y_{\text{est}}$ where $M$ is from §3.2.

### 5.1 The M Operator

The matrix $M = D_\lambda Q^T$ acts as a generalized projection onto the subspace where the covariates $X$ contribute most. Recovering latent factors from $M Y_{\text{est}}$ rather than raw $Y_{\text{est}}$ removes X-confounded signal from the first singular vectors.

### 5.2 Randomized SVD Algorithm

Given $M \in \mathbb{R}^{n \times n}$ and $Y_{\text{est}} \in \mathbb{R}^{n \times p_{\text{est}}}$, we seek the top $K$ left singular vectors of $A = M Y_{\text{est}}$.

**Sketch dimension:** $l = K + l_{\text{oversample}}$ (default $l_{\text{oversample}} = 10$).

**Step 1a: Initial Sketch**

- Generate random Gaussian matrix $\Omega \in \mathbb{R}^{n \times l}$ with $\omega_{ij} \sim \mathcal{N}(0, 1)$.
- Precompute $M^T \Omega \in \mathbb{R}^{n \times l}$ via BLAS.
- Streaming sketch: $Z = Y_{\text{est}}^T (M^T \Omega)$. For each chunk $Y_c$:

  $$Z_c = Y_c^T (M^T \Omega)$$

  accumulate into $Z \in \mathbb{R}^{p_{\text{est}} \times l}$.
- Compute the thin QR decomposition $Z = Q_z R_z$, obtaining an orthonormal basis $Q_z$ for the approximate column space of $A^T$.

**Step 1b: Power Iterations** (repeated $n_{\text{power}}$ times, default 2)

For each iteration:

1. **Forward pass** ($A Q_z$): For each chunk $Y_c$ with corresponding slice $Q_{z,c}$:

   $$A Q_{z,c} = M \cdot (Y_c Q_{z,c}) \in \mathbb{R}^{n \times l}$$

   Accumulated across workers using per-worker buffers, summed after.

2. QR decompose $A Q_z$ to get $Q_{aqz}$.
3. Compute $M^T Q_{aqz}$ via BLAS.
4. **Backward pass** ($A^T Q_{aqz}$): For each chunk:

   $$Z_c = Y_c^T (M^T Q_{aqz}) \in \mathbb{R}^{p_c \times l}$$

5. QR decompose the accumulated $Z$ to update $Q_z$.

This alternation sharpens the approximation of the left singular subspace: each power iteration contracts the subspace error by a factor of roughly $(\sigma_{K+1}/\sigma_K)^2$, giving a total contraction of $(\sigma_{K+1}/\sigma_K)^{2 n_{\text{power}}}$ after $n_{\text{power}}$ iterations.

**Step 2: Project and Recover**

- Compute $B_{\text{svd}} = A Q_z = M Y_{\text{est}} Q_z \in \mathbb{R}^{n \times l}$ via one final streaming pass.
- Thin SVD of $B_{\text{svd}}$ (only $U$ needed):

  $$B_{\text{svd}} = U_{\text{small}} \Sigma_{\text{small}} V_{\text{small}}^T$$

  with $U_{\text{small}} \in \mathbb{R}^{n \times l}$.

- Recover latent factors via:

  $$U_{\text{hat}} = Q \cdot \left(D_\lambda^{-1} \odot U_{\text{small}}[:, 1:K]\right)$$

  where $\odot$ denotes row-wise scaling: $(D_\lambda^{-1} U_{[1:K]})_{ik} = d_\lambda^{-1}[i] \cdot U_{\text{small}}[i, k]$.

The output is $U_{\text{hat}} \in \mathbb{R}^{n \times K}$, the estimated latent factor matrix.

---

## 6. Step 3: Covariate Effect Size Estimation

Given $U_{\text{hat}}$ from §5, the goal is to estimate the covariate effects $B \in \mathbb{R}^{d \times p}$ while correcting for latent structure. The approach projects out the latent subspace before estimating $B$.

### 6.1 Orthogonal Projector Onto Latent Subspace

Define the orthogonal projector onto the column space of $U_{\text{hat}}$:

$$
P_U = U_{\text{hat}} (U_{\text{hat}}^T U_{\text{hat}})^{-1} U_{\text{hat}}^T \in \mathbb{R}^{n \times n}
$$

Then $I - P_U$ projects onto the orthogonal complement, removing latent-factor signal from the genotypes.

### 6.2 Ridge-Regularized Least Squares

The covariate effects $\hat{B}$ are estimated column-by-column via ridge regression on the residualised genotypes:

For each SNP column $y_j \in \mathbb{R}^n$:

$$
r_j = (I - P_U) y_j
$$

$$
\hat{\beta}_j = (X^T X + \lambda I_d)^{-1} X^T r_j
$$

In matrix form, for all SNPs simultaneously:

$$
\hat{B}^T = (X^T X + \lambda I_d)^{-1} X^T (I - P_U) Y_{\text{full}}
$$

This is computed chunk-by-chunk in the fused pass. Define $X_t R = (X^T X + \lambda I_d)^{-1} X^T \in \mathbb{R}^{d \times n}$, precomputed in §3.3. For each chunk $Y_c$:

$$
\hat{B}_c^T = X_t R \cdot (I - P_U) \cdot Y_c
$$

The betas are written directly to disk as chunk fragments; no $p$-dimensional array is held in RAM.

### 6.3 Remark: relation to the paper's Eq. 4 and to the Step-4 test statistic

The paper defines $\hat{B}^T = (X^T X + \lambda I)^{-1} X^T (Y - \hat{W})$ with $\hat{W} = \hat{U}\hat{V}^T$ the rank-$K$ reconstruction of $Y$. We instead project out the whole estimated latent subspace via $(I - P_U)Y$. The two agree when $\hat{V}^T$ equals the OLS fit of $Y$ onto the columns of $\hat{U}$, which holds approximately when the rank-$K$ SVD captures the latent signal well: $(I - P_U)Y$ removes *every* component of $Y$ lying in $\operatorname{col}(\hat{U})$, whereas $Y - \hat{W}$ removes only the fitted rank-$K$ component.

Note also that the Step-3 effect sizes $\hat{\beta}_j$ and the Step-4 t-statistics (§7) arise from **different estimators**: Step 3 is ridge regression of $(I - P_U) y_j$ on $X$ alone, whereas Step 4 is joint OLS of $y_j$ on $[\mathbf{1} \; X \; U_{\text{hat}}]$. The two coincide when $X \perp U_{\text{hat}}$ and $\lambda \to 0$, but not in general — in particular not when covariates correlate with population structure, which is precisely the setting LFMM addresses. Consequently the reported `beta` column does not exactly satisfy $t = \beta / \mathrm{SE}(\beta)$: the t-statistic derives from the Step-4 joint fit. (The LEA reference has the same split: effect sizes from the ridge-residualised fit, z-scores from a per-locus `lm`.)

---

## 7. Step 4: Per-SNP Association Testing

For each SNP $j$, we test the null hypothesis that the covariate effects equal zero ($\beta_j = 0$), while adjusting for both observed covariates and estimated latent factors.

### 7.1 Full Model Specification

For SNP $j$, the full linear model is:

$$
y_j = \mathbf{1} \alpha_j + X \beta_j + U_{\text{hat}} \gamma_j + \varepsilon_j
$$

where $\alpha_j$ is the intercept, $\beta_j \in \mathbb{R}^d$ are covariate effects of interest, $\gamma_j \in \mathbb{R}^K$ are latent factor effects, and $\varepsilon_j \sim \mathcal{N}(0, \sigma_j^2 I_n)$.

Define the design matrix $C = [\mathbf{1} \;|\; X \;|\; U_{\text{hat}}] \in \mathbb{R}^{n \times (1+d+K)}$.

### 7.2 Ordinary Least Squares

For each column $y_j$, the OLS coefficient vector is:

$$
\hat{\theta}_j = (C^T C)^{-1} C^T y_j = H y_j
$$

where $H = (C^T C)^{-1} C^T \in \mathbb{R}^{(1+d+K) \times n}$ is precomputed once. For each chunk $Y_c$, this is applied as a single matrix multiplication:

$$
\hat{\Theta}_c = H Y_c
$$

yielding coefficients $\hat{\Theta}_c \in \mathbb{R}^{(1+d+K) \times p_c}$.

### 7.3 Residual Variance

The residual sum of squares for SNP $j$:

$$
\text{RSS}_j = \|y_j - C \hat{\theta}_j\|^2 = \varepsilon_j^T \varepsilon_j
$$

The residual variance estimate:

$$
\hat{\sigma}_j^2 = \frac{\text{RSS}_j}{n - 1 - d - K}
$$

with degrees of freedom $\text{df} = n - 1 - d - K$.

### 7.4 t-Tests for Covariate Effects

The covariate coefficients occupy indices $1, \ldots, d$ in $\hat{\theta}_j$ (index 0 is the intercept). For covariate $m$:

$$
\text{SE}(\hat{\beta}_{j,m}) = \sqrt{\hat{\sigma}_j^2 \cdot [(C^T C)^{-1}]_{m+1, m+1}}
$$

$$
t_{j,m} = \frac{\hat{\beta}_{j,m}}{\text{SE}(\hat{\beta}_{j,m})}
$$

The two-sided p-value is computed from Student's $t$ distribution with df degrees of freedom:

$$
p_{j,m} = 2 \cdot F_{t_{\text{df}}}(-|t_{j,m}|)
$$

where $F_{t_{\text{df}}}$ is the CDF. Zero-variance (monomorphic) SNPs yield $t = 0, p = 1$.

### 7.5 Variance Decomposition

The total sum of squares for SNP $j$ is partitioned by a **sequential (Type-I) decomposition** with term order $[\mathbf{1}, X, U_{\text{hat}}]$:

$$
\text{TSS}_j = \|y_j\|^2 = \text{SS}_{\text{cov}}(j) + \text{SS}_{\text{latent}}(j) + \text{RSS}_j
$$

where, writing $H_X = X (X^T X)^{-1} X^T$ for the orthogonal projector onto $\operatorname{col}(X)$ (since $X$ is centered, $H_X y_j$ is exactly the OLS fit of $y_j$ on $[\mathbf{1} \; X]$),

$$
\text{SS}_{\text{cov}}(j) = \|H_X y_j\|^2, \qquad
\text{SS}_{\text{latent}}(j) = \|C \hat{\theta}_j\|^2 - \|H_X y_j\|^2 = \text{TSS}_j - \text{RSS}_j - \text{SS}_{\text{cov}}(j)
$$

and the $R^2$ fractions are $r^2_{\text{cov}} = \text{SS}_{\text{cov}} / \text{TSS}$, etc. The identity is **exact** (up to floating-point error): because $H_X y_j \in \operatorname{col}(C)$ and the OLS residual $\hat{\varepsilon}_j$ is orthogonal to $\operatorname{col}(C)$, the fitted vector decomposes orthogonally as $\|C\hat{\theta}_j\|^2 = \|H_X y_j\|^2 + \|C\hat{\theta}_j - H_X y_j\|^2$, so the three reported fractions sum to 1. Monomorphic SNPs ($\text{TSS} \approx 0$) report all zeros.

**Ordering caveat:** as with any sequential decomposition, variance jointly explainable by $X$ and $U_{\text{hat}}$ is attributed to $X$, the term entering first. $r^2_{\text{cov}}$ is thus the *marginal* variance explained by the covariates (ignoring the latent factors), while $r^2_{\text{latent}}$ is the *additional* variance explained by the latent factors beyond the covariates. These are reported per-SNP in the output.

---

## 8. Latent-Factor Imputation (GWAS Phase)

During the GWAS/testing phase the genotype matrix $Y_{\text{full}} \in \mathbb{R}^{n \times p}$ (all SNPs, not just the estimation subset) must be imputed. Because the fill is a projection onto a fixed $n$-dimensional subspace, nothing per-SNP has to be carried over from the estimation phase — the per-chunk coefficients are computed on the fly.

### 8.1 Per-Chunk Procedure

For each chunk $c$ of $Y_{\text{full}}$, with $\hat{U} \in \mathbb{R}^{n \times K}$ fixed from §4.3:

- **Pass 1:** Raw-decode genotypes to $\{0, 1, 2, \text{NaN}\}$. Fill NaN entries with the per-column mean $\bar{y}_j$ over observed entries, giving $Y^{\text{filled}}_c$.

- **Fit the latent coefficients:**

  $$C_c = (\hat{U}^T \hat{U})^{-1} \hat{U}^T \left(Y^{\text{filled}}_c - \bar{y}\right) \in \mathbb{R}^{K \times p_c}$$

- **Compute fill values:** $\hat{Y}_c = \text{clamp}(\hat{U} C_c + \bar{y},\, 0,\, 2)$, on the raw dosage scale.

- **Pass 2:** Re-decode from raw bytes. NaN entries are filled with the corresponding $\hat{Y}_{ij}$ **before** centering and normalization; then standard centering (± Eigenstrat scaling) is applied.

As with any fill, imputed genotypes participate in the centering mean computation — the column mean used for centering is taken over all entries including the filled ones, which differs slightly from the $\bar{y}_j$ used to build the fill.

### 8.2 Shrinkage toward the mean

The per-chunk fit sees $\bar{y}_j$ at the missing positions rather than the (unknown) true dosage. Writing $P = P_{\hat{U}}$, the prediction at a missing entry $(i,j)$ relates to the leave-one-out fit by

$$
\hat{y}^{\text{fill}}_{ij} = (1 - P_{ii})\, \hat{y}^{\text{LOO}}_{ij} + P_{ii}\, \bar{y}_j ,
$$

i.e. it is shrunk toward the mean-imputation baseline with weight $P_{ii}$, which averages $K/n$. This is a mild and *conservative* approximation: it can only pull the estimate back toward the mean baseline, never past it. It is also why latent imputation degrades gracefully to parity, rather than to a loss, when there is no structure to exploit.

No further regularisation is applied. Ridge shrinkage on $C_c$ was tested and monotonically erodes the gain (penalty 1 / 5 / 20 gives +5.3% / +2.0% / +0.6% versus +8.7% unregularised at $F_{ST} = 0.10$); $K/n$ is small enough that the unregularised projection is the right estimator.

---

## 9. Held-Out Imputation Cross-Validation

Imputation accuracy is measured during the GWAS pass, using exactly the estimator that fills genotypes there. This is an honest held-out estimate: the evaluated positions are masked before the coefficients are fitted, so they cannot leak into their own prediction.

### 9.1 Per-Block Procedure

For each chunk $c$ during the GWAS streaming pass:

1. **Raw decode** genotypes from `block.raw` into $Y^{\text{raw}}_c \in \{0, 1, 2, \text{NaN}\}^{n \times p_c}$.

2. **Generate mask:** each *observed* entry is selected with probability $\text{cv\_rate}$ (default $5 \times 10^{-4}$). Seed: $\text{seed} + \text{chunk\_seq}$.

3. **Create leave-out matrix** $Y^{\text{masked}}_c$: a copy of $Y^{\text{raw}}_c$ with masked positions set to NaN.

4. **Mean-impute** $Y^{\text{masked}}_c$: per column, replace all NaN entries (truly missing + masked) with the mean over the remaining observed entries. Masked positions are excluded from this mean, so the baseline they are scored against does not see them.

5. **Fit and predict** with the §8.1 estimator on the masked matrix:

   $$\hat{Y}^{\text{cv}}_c = \text{clamp}\!\left(\hat{U} (\hat{U}^T \hat{U})^{-1} \hat{U}^T \left(Y^{\text{masked}}_c - \bar{y}\right) + \bar{y},\; 0,\; 2\right)$$

6. **Score on masked positions** (raw dosage scale), paired against the mean-imputation baseline:

   $$
   e^{\text{impute}} = \sum_{(i,j) \in \text{mask}} \left|Y^{\text{raw}}_{ij} - \hat{Y}^{\text{cv}}_{ij}\right|,
   \qquad
   e^{\text{mean}} = \sum_{(i,j) \in \text{mask}} \left|Y^{\text{raw}}_{ij} - \bar{y}_j\right|
   $$

7. Accumulate across chunks; after the pass, report $\text{MAE} = e / N_{\text{masked}}$ for each.

Both are evaluated on the same positions (a paired comparison) and on the raw dosage scale — the scale on which the fill is applied. The two MAEs and the count are written to the summary file as `gwas_cv_mae`, `gwas_cv_mean_mae` and `gwas_cv_count`. If the imputer fails to beat the baseline, a warning is printed to stderr and recorded as `impute_warning` in the summary; the requested imputation strategy is used either way (there is no automatic fallback).

---

## 10. Genomic Inflation Factor (GIF) Calibration

Population stratification and cryptic relatedness can inflate test statistics. The genomic inflation factor (GIF) is estimated from the empirical distribution of $t^2$ statistics across all SNPs and covariates.

### 10.1 Streaming Histogram

A streaming histogram over $t^2 \in [0, 100]$ with bin width $0.001$ (100,000 bins) is maintained per covariate trait across all chunks. Let $h_m[k]$ be the count of $t^2$ values falling in bin $k$ for trait $m$. After processing all $p$ SNPs:

- Total tested SNPs for trait $m$: $N_m = \sum_k h_m[k]$.
- Median $t^2$: the value at position $N_m/2$ (or average of positions $N_m/2 - 1$ and $N_m/2$ for even $N_m$), found by walking cumulative bin counts.

### 10.2 GIF Computation

Under the null hypothesis and proper calibration, each $t^2$ follows approximately a scaled $\chi^2_1$ distribution. The median of a $\chi^2_1$ distribution is approximately $0.4549$. The raw GIF estimate for trait $m$ is:

$$
\text{GIF}^{\text{raw}}_m = \frac{\text{median}(t^2_m)}{0.4549}
$$

For calibration, the divisor is clamped to a minimum of 1.0: $\text{GIF}_m = \max(\text{GIF}^{\text{raw}}_m, 1)$, since dividing by $\sqrt{\text{GIF}} < 1$ would inflate $|z|$ and make the calibration anti-conservative. The average of the *unclamped* estimates, $\overline{\text{GIF}}^{\text{raw}} = \frac{1}{d} \sum_{m=1}^d \text{GIF}^{\text{raw}}_m$, is reported in the summary output as a diagnostic (deflation below 1 is itself informative about model fit).

### 10.3 GIF Calibration of p-values

In the final output, each SNP's t-statistic is calibrated and converted to a two-sided p-value using the standard normal distribution:

$$
z_{j,m} = \frac{t_{j,m}}{\sqrt{\text{GIF}_m}}
$$

$$
p^{\text{cal}}_{j,m} = 2 \cdot \Phi(-|z_{j,m}|)
$$

where $\Phi$ is the standard normal CDF. This is equivalent to dividing the test statistic by the GIF estimate to correct for inflation.

**Reporting note.** The output's `p_*` columns are these GIF-calibrated p-values, while the `t_*` columns are the raw (uncalibrated) t-statistics of §7.4. Hence $p \ne 2\, F_{t_{\text{df}}}(-|t|)$ for the printed values whenever $\text{GIF}_m > 1$; the uncalibrated p-value can be recovered from the printed $t$ directly. The `beta_*` columns are the Step-3 ridge effect sizes — see the §6.3 remark for why they need not satisfy $t = \beta/\mathrm{SE}$ exactly.

---

## 11. Complete Pipeline Summary

The full LFMM2 pipeline proceeds as follows:

1. **Load data:** Read BED, FAM, BIM, and covariate files. Align and subset samples.

2. **Step 0:** Precompute SVD of $X$, build $M = D_\lambda Q^T$, $\text{ridge\_inv}$, and $D_\lambda^{-1}$ (§3).

3. **Steps 1-2 (Latent factor estimation via RSVD):**
   - Generate random sketch matrix $\Omega$.
   - Compute initial sketch $Z = Y_{\text{est}}^T (M^T \Omega)$.
   - QR decompose to get $Q_z$.
   - Run $n_{\text{power}}$ power iterations (forward $M Y_{\text{est}} Q_z$, QR, backward $Y_{\text{est}}^T (M^T Q_{\text{aqz}})$, QR).
   - Final projection $B = M Y_{\text{est}} Q_z$, SVD, recover $\hat{U}$ (§5).

4. **(If `--latent-impute`) One EM step:** repeat step 3 over the estimation subset with `LatentFill` using $\hat{U}^{(1)}$, giving $\hat{U}^{(2)}$ (§4.3). Precompute $(\hat{U}^T \hat{U})^{-1}$ for per-chunk fits.

5. **Fused Steps 3-4 (GWAS/testing):**
   - Precompute OLS hat matrices: $I - P_U$, $X_t R$, $H = (C^T C)^{-1} C^T$.
   - For each chunk $c$:
     - **(If latent impute)** Fit $C_c = (\hat{U}^T\hat{U})^{-1}\hat{U}^T(Y_c^{\text{filled}} - \bar{y})$, fill NaNs with $\text{clamp}(\hat{U} C_c + \bar{y}, 0, 2)$ before centering (§8).
     - Compute $\hat{B}_c$ (effect sizes) via ridge regression on residualised genotypes (§6).
     - Compute $\hat{\Theta}_c$ (full OLS coefficients) and $t_{j,m}$ statistics (§7).
     - Compute $r^2$ variance decomposition.
     - **(If latent impute)** Compute held-out imputation CV on masked positions with the same estimator (§9).
     - Feed $t^2$ values into streaming histogram for GIF estimation (§10).
     - Write per-chunk results to temporary parquet fragments.

6. **Post-processing:**
   - Compute GIF from histogram medians (§10.2).
   - Read back chunk files, calibrate p-values with GIF (§10.3), and write final output.
   - Write summary file with GIF (overall and per covariate), CV metrics, and diagnostics.

### Computational Complexity

| Step | Passes over Y | Memory (RAM) |
|------|--------------|--------------|
| SVD of X | 0 (in-core) | $O(nd + d^2)$ |
| RSVD | $3 + 2 n_{\text{power}}$ passes | $O(nK + p_{\text{est}} K)$ |
| Latent-impute EM step | $3 + 2 n_{\text{power}}$ more passes over $Y_{\text{est}}$ | $O(nK + p_{\text{est}} K)$ |
| GWAS fused | 1 pass | $O(n(1+d+K) + p_c d)$ per chunk |
| GWAS CV | 0 extra passes | $O(nK + p_c K)$ in process_fn |

The latent-imputation EM step costs one extra round of factor estimation over the *estimation subset* only (typically LD-pruned, and small next to the full GWAS scan); the GWAS pass itself costs one extra decode per chunk.

All passes over genotype data are fully streaming: at any point, only one chunk of $\text{chunk\_size}$ SNPs (default 10,000) is held in RAM.

### Imputation Strategies Summary

| Phase | Strategy | Method |
|-------|----------|--------|
| RSVD, first pass (estimation subset) | `Mean` | Column-mean fill; zero after centering |
| RSVD, EM step (estimation subset) | `LatentFill` | $\text{clamp}(\bar{y} + P_{\hat{U}}(Y^{\text{filled}} - \bar{y}), 0, 2)$ with $\hat{U}^{(1)}$ |
| GWAS testing (all SNPs) | `LatentFill` or `Mean` | Same fill with $\hat{U}^{(2)}$, or column mean |
| GWAS CV (all SNPs) | `LatentFill` on a masked copy | Scores the fill against raw truth on held-out positions, paired with the column-mean baseline |
