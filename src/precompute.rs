use anyhow::Result;
use ndarray::{Array1, Array2};
use ndarray_linalg::{InverseInto, SVD};

/// Precomputed quantities from Step 0 of the LFMM2 algorithm.
///
/// All derived from SVD of X and the ridge parameter λ.
/// These are n×n or smaller and fit comfortably in RAM.
pub struct Precomputed {
    /// Q from full SVD of X: n × n orthogonal matrix
    pub q_full: Array2<f64>,
    /// D_λ diagonal: d_λ[j] = λ/(λ+σ_j²) for j<d, 1.0 for j≥d
    pub d_lambda: Array1<f64>,
    /// D_λ^{-1} diagonal
    pub d_lambda_inv: Array1<f64>,
    /// M = D_λ @ Q^T: n × n
    pub m: Array2<f64>,
    /// (X^T X + λI_d)^{-1}: d × d
    pub ridge_inv: Array2<f64>,
}

/// Perform Step 0 precomputations.
///
/// Given X (n × d) and ridge penalty λ:
/// 1. SVD of X: X = Q Σ R^T
/// 2. D_λ = diag([λ/(λ+σ_j²) for j in 0..d] ++ [1.0; n-d])
/// 3. M = D_λ @ Q^T
/// 4. ridge_inv = (X^T X + λI)^{-1}
pub fn precompute(x: &Array2<f64>, lambda: f64) -> Result<Precomputed> {
    let n = x.nrows();
    let d = x.ncols();

    // SVD of X: X = Q Σ R^T
    // ndarray-linalg SVD returns (U, s, Vt) where X = U @ diag(s) @ Vt
    // U is n×n (full) when X is n×d with n > d
    let (u_opt, s, _vt_opt) = x.svd(true, true)?;
    let q_full = u_opt.expect("SVD should return U");
    let sigma = s; // singular values, length min(n, d)

    // Build D_λ diagonal (length n)
    let mut d_lambda = Array1::<f64>::ones(n);
    for j in 0..sigma.len().min(d) {
        let s2 = sigma[j] * sigma[j];
        d_lambda[j] = lambda / (lambda + s2);
    }

    // D_λ^{-1}
    let d_lambda_inv = d_lambda.mapv(|v| 1.0 / v);

    // M = D_λ @ Q^T
    // D_λ is diagonal, so M[i, j] = d_lambda[i] * Q^T[i, j] = d_lambda[i] * Q[j, i]
    let qt = q_full.t();
    let m = {
        let mut m = qt.to_owned();
        for i in 0..n {
            let scale = d_lambda[i];
            m.row_mut(i).mapv_inplace(|v| v * scale);
        }
        m
    };

    // ridge_inv = (X^T X + λ I_d)^{-1}
    let xtx = x.t().dot(x);
    let mut xtx_ridge = xtx;
    for j in 0..d {
        xtx_ridge[(j, j)] += lambda;
    }
    let ridge_inv = xtx_ridge.inv_into()?;

    Ok(Precomputed {
        q_full,
        d_lambda,
        d_lambda_inv,
        m,
        ridge_inv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_precompute_small() {
        // Small 4×2 matrix
        let x = array![
            [1.0, 0.5],
            [0.3, -0.2],
            [-0.5, 1.0],
            [0.8, 0.1],
        ];
        let lambda = 1.0;

        let pre = precompute(&x, lambda).unwrap();

        // Q should be 4×4 orthogonal
        assert_eq!(pre.q_full.shape(), &[4, 4]);
        let qtq = pre.q_full.t().dot(&pre.q_full);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[(i, j)], expected, epsilon = 1e-10);
            }
        }

        // D_lambda: first 2 entries should be lambda/(lambda + sigma_j^2), rest = 1.0
        assert_eq!(pre.d_lambda.len(), 4);
        assert!(pre.d_lambda[0] > 0.0 && pre.d_lambda[0] < 1.0);
        assert!(pre.d_lambda[1] > 0.0 && pre.d_lambda[1] < 1.0);
        assert_abs_diff_eq!(pre.d_lambda[2], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pre.d_lambda[3], 1.0, epsilon = 1e-10);

        // M should be 4×4
        assert_eq!(pre.m.shape(), &[4, 4]);

        // ridge_inv should be 2×2
        assert_eq!(pre.ridge_inv.shape(), &[2, 2]);

        // Verify ridge_inv is inverse of (X^T X + λI)
        let xtx = x.t().dot(&x);
        let mut xtx_ridge = xtx;
        xtx_ridge[(0, 0)] += lambda;
        xtx_ridge[(1, 1)] += lambda;
        let product = xtx_ridge.dot(&pre.ridge_inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(product[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }
}
