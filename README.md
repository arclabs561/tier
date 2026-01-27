# tier

Clustering + community detection primitives (feature-gated, domain-agnostic).

This repo is **domain-agnostic** (not Anno-specific). The stable contract is in `CONTRACT.md`.

## What’s in here

- **Clustering** (`cluster`): k-means, DBSCAN, hierarchical clustering, plus utilities.
- **Community detection** (`community`, `knn-graph`): kNN graph construction + Leiden/Louvain/label-prop style methods.
- **Hierarchy + conformal** (`std` + modules): hierarchical reconciliation + split conformal prediction.
- **Metrics / distances** (`metrics`, `wass`): small evaluation helpers used by `flowmatch`’s USGS demos.

## Quick Start (hierarchical conformal)

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

## Quick Start (embedding clustering / kNN graph)

This example is feature-gated because it depends on vector/graph tooling:

```bash
cargo run --example embedding_clustering --features knn-graph
```

## Mathematical Foundation

Hierarchical data follows the constraint $y = S \cdot b$, where $S$ is the summing matrix and $b$ are the base (leaf) values. Reconciliation finds $\tilde{y}$ such that $\|\hat{y} - \tilde{y}\|_W$ is minimized subject to structural coherence.

## References

- Principato et al. (2024). "Conformal Prediction for Hierarchical Data."
- Qiu & Li (2015). "IT-Dendrogram: A new representation for hierarchical clustering."
- Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."
