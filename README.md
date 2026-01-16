# tier

Clustering and hierarchical structure primitives for the Tekne stack.

## Features

### Clustering
- **IT-Dendrogram**: In-Tree representation for revealing underlying cluster geometry.

### Hierarchical Conformal Prediction
Implementation of **Split Conformal Prediction (SCP)** for hierarchical data, ensuring coherent prediction intervals across multiple scales.

- **Reconciliation**:
  - `Ols`: Ordinary Least Squares
  - `Wls`: Weighted Least Squares
  - `MinT`: Minimum Trace (using covariance estimates)
- **Joint Coverage**: Simultaneous coverage guarantees across all tree nodes.
- **Integrations**: Direct converters for `strata::tree::RaptorTree` and `strata::tree::Dendrogram`.

## Quick Start

```rust
use tier::{HierarchicalConformal, HierarchyTree, ReconciliationMethod};
use faer::Mat;

// 1. Convert an existing tree structure
let h_tree = HierarchyTree::from_raptor(&my_raptor_tree);
let s = h_tree.summing_matrix();

// 2. Calibrate using a calibration set
let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);
cp.calibrate(&y_calib, &y_hat_calib, 0.1)?; // 90% coverage

// 3. Generate coherent intervals
let (lower, upper) = cp.predict_intervals(&y_hat_test)?;
```

## Mathematical Foundation

Hierarchical data follows the constraint $y = S \cdot b$, where $S$ is the summing matrix and $b$ are the base (leaf) values. Reconciliation finds $\tilde{y}$ such that $\|\hat{y} - \tilde{y}\|_W$ is minimized subject to structural coherence.

## References

- Principato et al. (2024). "Conformal Prediction for Hierarchical Data."
- Qiu & Li (2015). "IT-Dendrogram: A new representation for hierarchical clustering."
- Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."
