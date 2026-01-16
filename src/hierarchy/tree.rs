//! Hierarchy tree structures.

use crate::reconciliation::SummingMatrix;
use faer::Mat;
use crate::hierarchy::{RaptorTree, Dendrogram};

/// A rooted tree with edge weights, representing an ultrametric.
#[derive(Debug, Clone)]
pub struct HierarchyTree {
    /// Parent of each node (-1 for root).
    parents: Vec<i32>,
    /// Height of each node (distance from leaves).
    #[allow(dead_code)] // Reserved for future ultrametric operations
    heights: Vec<f64>,
    /// Number of leaf nodes.
    num_leaves: usize,
}

impl HierarchyTree {
    /// Create from a RAPTOR tree.
    pub fn from_raptor<T, S>(tree: &RaptorTree<T, S>) -> Self {
        let m = tree.len();
        let n = tree.leaves().len();
        let mut parents = vec![-1i32; m];
        
        for node in tree.iter() {
            let parent_id = node.id;
            for &child_id in &node.children {
                if child_id < m {
                    parents[child_id] = parent_id as i32;
                }
            }
        }

        Self {
            parents,
            heights: vec![0.0; m], // RAPTOR doesn't track distances by default
            num_leaves: n,
        }
    }

    /// Create from a dendrogram.
    pub fn from_dendrogram(dend: &Dendrogram) -> Self {
        let n_leaves = dend.n_items();
        let merges: Vec<_> = dend
            .merges()
            .map(|m| (m.cluster_a, m.cluster_b, m.distance, m.size))
            .collect();
        Self::from_merges(&merges, n_leaves)
    }

    /// Create from a dendrogram (merge sequence).
    ///
    /// # Arguments
    /// * `merges` - Sequence of (cluster_a, cluster_b, height, size) merges
    /// * `n_leaves` - Number of original items
    pub fn from_merges(merges: &[(usize, usize, f64, usize)], n_leaves: usize) -> Self {
        let n_internal = merges.len();
        let n_total = n_leaves + n_internal;

        let mut parents = vec![-1i32; n_total];
        let mut heights = vec![0.0f64; n_total];

        for (i, &(a, b, h, _)) in merges.iter().enumerate() {
            let internal_id = n_leaves + i;
            parents[a] = internal_id as i32;
            parents[b] = internal_id as i32;
            heights[internal_id] = h;
        }

        Self {
            parents,
            heights,
            num_leaves: n_leaves,
        }
    }

    /// Generate the structural summing matrix S.
    pub fn summing_matrix(&self) -> SummingMatrix {
        let m = self.parents.len();
        let n = self.num_leaves;
        let mut mat = Mat::<f64>::zeros(m, n);

        for leaf_idx in 0..n {
            let mut current = leaf_idx;
            while current < m {
                mat[(current, leaf_idx)] = 1.0;
                let parent = self.parents[current];
                if parent < 0 {
                    break;
                }
                current = parent as usize;
            }
        }

        SummingMatrix::new(mat)
    }

    /// Number of total nodes.
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Number of leaf nodes.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }
}
