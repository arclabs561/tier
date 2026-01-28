#[cfg(test)]
mod tests {
    use crate::hierarchy::HierarchicalConformal;
    use crate::reconciliation::{reconcile, ReconciliationMethod, SummingMatrix};
    use crate::Result;
    use faer::Mat;

    #[test]
    fn test_reconciliation_ols_star() -> Result<()> {
        // Star hierarchy: 1 root, 2 leaves
        // S = [1 1] (root)
        //     [1 0] (leaf 1)
        //     [0 1] (leaf 2)
        let s = SummingMatrix::simple_star(2);

        // Base forecasts: root=3, leaf1=1, leaf2=1
        // (Not coherent: 1+1 = 2 != 3)
        let mut y_hat = Mat::<f64>::zeros(3, 1);
        y_hat[(0, 0)] = 3.0;
        y_hat[(1, 0)] = 1.0;
        y_hat[(2, 0)] = 1.0;

        let y_tilde = reconcile(&s, &y_hat, ReconciliationMethod::Ols)?;

        assert!((y_tilde[(0, 0)] - 2.6666666666666665).abs() < 1e-10);
        assert!((y_tilde[(1, 0)] - 1.3333333333333333).abs() < 1e-10);
        assert!((y_tilde[(2, 0)] - 1.3333333333333333).abs() < 1e-10);

        // Check coherence: root = sum(leaves)
        assert!((y_tilde[(0, 0)] - (y_tilde[(1, 0)] + y_tilde[(2, 0)])).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_conformal_calibration() -> Result<()> {
        let s = SummingMatrix::simple_star(2);
        let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);

        // 10 calibration points
        let y_calib = Mat::<f64>::zeros(3, 10); // True values (all zero for simplicity)

        // Forecasts with known errors
        // Error = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (Euclidean norms)
        let mut y_hat_calib = Mat::<f64>::zeros(3, 10);
        for j in 0..10 {
            y_hat_calib[(1, j)] = (j + 1) as f64; // Error in leaf 1
        }

        // alpha = 0.2 (80% coverage)
        // n=10, (n+1)*(1-alpha) = 11 * 0.8 = 8.8 -> 9th score
        cp.calibrate(&y_calib, &y_hat_calib, 0.2)?;

        // Reconciled scores will be slightly different from base errors,
        // but they should be monotonic with j.
        assert!(cp.quantile() > 0.0);
        Ok(())
    }

    #[test]
    fn test_reconciliation_mint() -> Result<()> {
        let s = SummingMatrix::simple_star(2);
        let m = s.m();

        // Base forecasts: root=3, leaf1=1, leaf2=1
        let mut y_hat = Mat::<f64>::zeros(m, 1);
        y_hat[(0, 0)] = 3.0;
        y_hat[(1, 0)] = 1.0;
        y_hat[(2, 0)] = 1.0;

        // Use identity as covariance (should match OLS)
        let covariance = Mat::<f64>::identity(m, m);

        let y_tilde = reconcile(&s, &y_hat, ReconciliationMethod::MinT { covariance })?;

        assert!((y_tilde[(0, 0)] - 2.6666666666666665).abs() < 1e-10);
        assert!((y_tilde[(0, 0)] - (y_tilde[(1, 0)] + y_tilde[(2, 0)])).abs() < 1e-10);
        Ok(())
    }
}
