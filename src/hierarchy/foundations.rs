//! Mathematical foundations of hierarchical structures.
//!
//! # The Deep Structure
//!
//! Various representations of hierarchy appear throughout computer science:
//! - Dendrograms from agglomerative clustering
//! - RAPTOR trees from recursive summarization
//! - Taxonomies (WordNet, Wikipedia categories)
//! - Hyperbolic embeddings (Poincare, Lorentz)
//!
//! These are all **views of the same mathematical object**: an ultrametric space.
//!
//! # Ultrametric Spaces
//!
//! An ultrametric satisfies a stronger triangle inequality:
//!
//! ```text
//! d(x, z) <= max(d(x, y), d(y, z))   (ultrametric inequality)
//! ```
//!
//! Compare to the standard triangle inequality:
//!
//! ```text
//! d(x, z) <= d(x, y) + d(y, z)       (metric inequality)
//! ```
//!
//! ## Why This Matters
//!
//! **Theorem**: A finite metric space is ultrametric if and only if it can be
//! represented as the leaf-to-leaf distances in a rooted tree with edge weights.
//!
//! This means: **ultrametric = tree structure**.
//!
//! ## The Ultrametric-Dendrogram Correspondence
//!
//! Given points {x1, ..., xn}:
//!
//! ```text
//! Dendrogram                 Ultrametric
//! ----------                 ----------
//!      *  (height=3)         d(a,d) = 3
//!     / \                    d(b,c) = 2
//!    *   d                   d(a,b) = 2
//!   / \                      d(a,c) = 2
//!  *   c                     d(c,d) = 3
//! / \                        d(b,d) = 3
//! a   b
//! ```
//!
//! The ultrametric distance between two leaves = height of their LCA.
//!
//! # Hyperbolic Geometry Connection
//!
//! Hyperbolic space naturally embeds trees with low distortion because:
//!
//! 1. **Volume grows exponentially** with radius (like tree node count with depth)
//! 2. **Geodesics diverge quickly** (capturing the "branching" structure)
//!
//! ```text
//! Euclidean:        Hyperbolic (Poincare disk):
//!                   
//!   . . . . .             .   .
//!  . . . . . .          .       .
//!  . . . . . .         .         .
//!  . . . . . .          .       .
//!   . . . . .             .   .
//!                   
//! Area ~ r²          Area ~ e^r (exponential!)
//! ```
//!
//! A 10-dim hyperbolic space can faithfully embed trees that would need
//! thousands of Euclidean dimensions.
//!
//! # Hierarchy in Embeddings
//!
//! ## Partial Order from Distance
//!
//! Given embeddings with a "centroid" or origin, we can infer hierarchy:
//!
//! ```text
//! x ≤ y  iff  d(origin, x) < d(origin, y) AND x is "between" origin and y
//! ```
//!
//! This is the basis of **hierarchical softmax** in word2vec and the
//! **distributional inclusion hypothesis**: hypernyms (general terms) tend
//! to appear in more contexts, hence closer to the centroid.
//!
//! ## Entailment Cones (Ganea et al. 2018)
//!
//! In hyperbolic space, we can define "cones" of descendants:
//!
//! ```text
//! Cone(x) = {y : d(origin, x) < d(origin, y) AND angle(origin-x, origin-y) < θ}
//! ```
//!
//! If y ∈ Cone(x), then x is an "ancestor" of y in the hierarchy.
//!
//! # Coreference and Abstraction
//!
//! ## Abstract Anaphora
//!
//! When text says "this" or "that", it often refers to an abstract concept
//! (an event, a claim, a process) rather than a concrete noun:
//!
//! ```text
//! "The experiment failed. This was unexpected."
//!                         ^^^^
//!                         Refers to the event "experiment failed"
//! ```
//!
//! Resolving such references is crucial for:
//! - Document summarization (what does "this finding" refer to?)
//! - QA systems (what is "the approach" in the question?)
//! - RAG pipelines (ensuring context coherence across chunks)
//!
//! ## Shell Nouns
//!
//! Abstract nouns that derive meaning from context:
//! - "This **problem**..." (which problem?)
//! - "The **idea** that..." (what idea?)
//! - "Such **behavior**..." (what behavior?)
//!
//! These create hierarchical structure: the shell noun is a summary/abstraction
//! of the content it references.
//!
//! # Mathematical Definitions
//!
//! ## Ultrametric
//!
//! A metric d on set X is **ultrametric** if for all x, y, z ∈ X:
//!
//! ```text
//! d(x, z) ≤ max{d(x, y), d(y, z)}
//! ```
//!
//! ## Subdominant Ultrametric
//!
//! Given any metric d, the **subdominant ultrametric** u* is the largest
//! ultrametric that is pointwise ≤ d:
//!
//! ```text
//! u*(x, y) = min over all paths x = v0, v1, ..., vk = y of max{d(vi, vi+1)}
//! ```
//!
//! This is exactly the single-linkage clustering result.
//!
//! ## Tree Metric
//!
//! A metric is a **tree metric** if it can be realized as path lengths in a
//! weighted tree. Every tree metric is an ultrametric, but not vice versa
//! (tree metrics need not be rooted).
//!
//! # Matryoshka Embeddings
//!
//! Multi-resolution representation via truncatable embeddings (Kusupati et al. 2022):
//!
//! ```text
//! Full embedding:     [x1, x2, x3, ..., x768]
//!                      |   |   |
//! Truncated (256):    [x1, x2, ..., x256]
//!                      |   |
//! Truncated (64):     [x1, ..., x64]
//! ```
//!
//! **Key insight**: Early dimensions capture coarse features, later dimensions
//! capture fine details. This mirrors hierarchical structure:
//!
//! - Low dimensions = high-level clustering (topics, categories)
//! - Full dimensions = fine-grained similarity (specific passages)
//!
//! 2D Matryoshka (Wang et al. 2024) extends this to layer-dimension flexibility,
//! enabling adaptive computation at query time.
//!
//! **Connection to hierarchy**: Matryoshka naturally implements coarse-to-fine
//! search, analogous to descending a RAPTOR tree from summaries to leaves.
//!
//! # References
//!
//! - Carlsson & Mémoli (2010): "Characterization, Stability and Convergence
//!   of Hierarchical Clustering Methods"
//! - Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical
//!   Representations"
//! - Ganea et al. (2018): "Hyperbolic Entailment Cones for Learning
//!   Hierarchical Embeddings"
//! - Chami et al. (2020): "From Trees to Continuous Embeddings and Back:
//!   Hyperbolic Hierarchical Clustering"
//! - Kusupati et al. (2022): "Matryoshka Representation Learning"
//! - Wang et al. (2024): "2D Matryoshka Training for Information Retrieval"
//! - Marasović et al. (2017): "A Mention-Ranking Model for Abstract Anaphora
//!   Resolution"

/// An ultrametric distance function.
///
/// Ultrametric distances satisfy the strong triangle inequality:
/// d(x, z) <= max(d(x, y), d(y, z))
///
/// This is the natural distance function for hierarchical structures.
pub trait Ultrametric<T> {
    /// Compute the ultrametric distance between two points.
    ///
    /// For a dendrogram, this is the height of the lowest common ancestor.
    /// For a tree, this is the path length through the root.
    fn ultrametric_distance(&self, a: &T, b: &T) -> f64;

    /// Find all points within ultrametric ball of radius r around x.
    ///
    /// Due to the ultrametric property, such balls are either disjoint
    /// or one contains the other (no partial overlap).
    fn ultrametric_ball(&self, center: &T, radius: f64) -> Vec<T>
    where
        T: Clone;
}

/// A rooted tree with edge weights, representing an ultrametric.
#[derive(Debug, Clone)]
pub struct UltrametricTree {
    /// Parent of each node (-1 for root).
    parents: Vec<i32>,
    /// Height of each node (distance from leaves).
    heights: Vec<f64>,
    /// Number of leaf nodes.
    num_leaves: usize,
}

impl UltrametricTree {
    /// Create from a dendrogram (merge sequence).
    ///
    /// # Arguments
    ///
    /// * `merges` - Sequence of (cluster_a, cluster_b, height, size) merges
    /// * `n_leaves` - Number of original items
    pub fn from_merges(merges: &[(usize, usize, f64, usize)], n_leaves: usize) -> Self {
        let n_internal = merges.len();
        let n_total = n_leaves + n_internal;

        let mut parents = vec![-1i32; n_total];
        let mut heights = vec![0.0f64; n_total];

        // Leaves have height 0
        // Internal nodes get height from merge

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

    /// Get the lowest common ancestor of two nodes.
    pub fn lca(&self, a: usize, b: usize) -> usize {
        // Collect ancestors of a
        let mut ancestors_a = std::collections::HashSet::new();
        let mut current = a;
        while current < self.parents.len() {
            let _ = ancestors_a.insert(current);
            let parent = self.parents[current];
            if parent < 0 {
                break;
            }
            current = parent as usize;
        }
        let _ = ancestors_a.insert(current); // Include root

        // Walk up from b until we hit an ancestor of a
        current = b;
        while !ancestors_a.contains(&current) {
            let parent = self.parents[current];
            if parent < 0 {
                break;
            }
            current = parent as usize;
        }

        current
    }

    /// Ultrametric distance = height of LCA.
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }
        let lca = self.lca(a, b);
        self.heights[lca]
    }

    /// Height of a node.
    pub fn height(&self, node: usize) -> f64 {
        self.heights.get(node).copied().unwrap_or(0.0)
    }

    /// Number of leaf nodes.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Check the ultrametric inequality holds.
    #[cfg(test)]
    fn verify_ultrametric(&self) -> bool {
        for i in 0..self.num_leaves {
            for j in 0..self.num_leaves {
                for k in 0..self.num_leaves {
                    let d_ik = self.distance(i, k);
                    let d_ij = self.distance(i, j);
                    let d_jk = self.distance(j, k);

                    // Ultrametric: d(i,k) <= max(d(i,j), d(j,k))
                    if d_ik > d_ij.max(d_jk) + 1e-10 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Compute subdominant ultrametric from a distance matrix.
///
/// The subdominant ultrametric is the largest ultrametric that is
/// pointwise <= the original metric. This is equivalent to
/// single-linkage clustering.
///
/// # Algorithm (Minimax path)
///
/// u*(i, j) = min over all paths i -> j of max edge weight
///
/// This can be computed via a variant of Floyd-Warshall.
pub fn subdominant_ultrametric(distances: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = distances.len();
    let mut u = distances.to_vec();

    // Floyd-Warshall variant for minimax paths
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let path_through_k = u[i][k].max(u[k][j]);
                if path_through_k < u[i][j] {
                    u[i][j] = path_through_k;
                }
            }
        }
    }

    u
}

/// Verify a distance matrix satisfies the ultrametric inequality.
pub fn is_ultrametric(distances: &[Vec<f64>], tolerance: f64) -> bool {
    let n = distances.len();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let d_ik = distances[i][k];
                let d_ij = distances[i][j];
                let d_jk = distances[j][k];

                if d_ik > d_ij.max(d_jk) + tolerance {
                    return false;
                }
            }
        }
    }
    true
}

/// Compute distortion when embedding an ultrametric in Euclidean space.
///
/// Distortion = max(expansion, contraction) where:
/// - expansion = max_ij d_embed(i,j) / d_orig(i,j)
/// - contraction = max_ij d_orig(i,j) / d_embed(i,j)
///
/// For ultrametrics on n points, Euclidean embedding requires
/// O(log n) distortion in the worst case.
pub fn embedding_distortion(original: &[Vec<f64>], embedded: &[Vec<f64>]) -> f64 {
    let n = original.len();
    let mut max_expansion = 1.0f64;
    let mut max_contraction = 1.0f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let d_orig = original[i][j];
            let d_embed = embedded[i][j];

            if d_orig > 1e-10 {
                max_expansion = max_expansion.max(d_embed / d_orig);
            }
            if d_embed > 1e-10 {
                max_contraction = max_contraction.max(d_orig / d_embed);
            }
        }
    }

    max_expansion.max(max_contraction)
}

/// Gromov hyperbolicity: measures how "tree-like" a metric space is.
///
/// A metric space is δ-hyperbolic if for all x, y, z, w:
///
/// ```text
/// d(x,y) + d(z,w) <= max(d(x,z) + d(y,w), d(x,w) + d(y,z)) + 2δ
/// ```
///
/// Trees have δ = 0. Lower δ = more tree-like.
///
/// # Returns
///
/// The minimum δ such that the space is δ-hyperbolic.
pub fn gromov_hyperbolicity(distances: &[Vec<f64>]) -> f64 {
    let n = distances.len();
    let mut max_delta = 0.0f64;

    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                for w in 0..n {
                    let sum_xy_zw = distances[x][y] + distances[z][w];
                    let sum_xz_yw = distances[x][z] + distances[y][w];
                    let sum_xw_yz = distances[x][w] + distances[y][z];

                    let max_other = sum_xz_yw.max(sum_xw_yz);
                    let delta = (sum_xy_zw - max_other) / 2.0;

                    max_delta = max_delta.max(delta);
                }
            }
        }
    }

    max_delta
}

// =============================================================================
// Note on Anaphora and Shell Nouns
// =============================================================================
//
// For full abstract anaphora resolution (discourse-level "this", "that", etc.)
// and shell noun classification, see the `anno` crate's discourse module:
//
// - `anno::discourse::ShellNoun` - Full Schmid (2000) taxonomy
// - `anno::discourse::DiscourseReferent` - Abstract antecedents
// - `anno::discourse::centering` - Centering theory for coherence
// - `anno::discourse::uncertain_reference` - Deferred resolution
//
// This module focuses on the *mathematical* foundations (ultrametric spaces,
// hyperbolic geometry) rather than linguistic patterns.

#[cfg(test)]
#[allow(clippy::unwrap_used, unused_results, clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_ultrametric_tree_basic() {
        // Create a simple tree:
        //      4 (h=2)
        //     / \
        //    3   \  (h=1)
        //   / \   \
        //  0   1   2
        let merges = vec![
            (0, 1, 1.0, 2), // Merge 0,1 at height 1 -> node 3
            (3, 2, 2.0, 3), // Merge node 3 with 2 at height 2 -> node 4
        ];

        let tree = UltrametricTree::from_merges(&merges, 3);

        // Check distances
        assert!((tree.distance(0, 1) - 1.0).abs() < 1e-10); // LCA at height 1
        assert!((tree.distance(0, 2) - 2.0).abs() < 1e-10); // LCA at height 2
        assert!((tree.distance(1, 2) - 2.0).abs() < 1e-10); // LCA at height 2

        // Verify ultrametric property
        assert!(tree.verify_ultrametric());
    }

    #[test]
    fn test_subdominant_ultrametric() {
        // A simple 3-point metric that isn't ultrametric
        let distances = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.5],
            vec![2.0, 1.5, 0.0],
        ];

        let u = subdominant_ultrametric(&distances);

        // Check ultrametric inequality
        assert!(is_ultrametric(&u, 1e-10));

        // Check subdominance (u <= original)
        for i in 0..3 {
            for j in 0..3 {
                assert!(u[i][j] <= distances[i][j] + 1e-10);
            }
        }
    }

    #[test]
    fn test_gromov_hyperbolicity_tree() {
        // A tree metric should have hyperbolicity 0
        let tree =
            UltrametricTree::from_merges(&[(0, 1, 1.0, 2), (2, 3, 1.0, 2), (4, 5, 2.0, 4)], 4);

        // Build distance matrix
        let mut distances = vec![vec![0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                distances[i][j] = tree.distance(i, j);
            }
        }

        let delta = gromov_hyperbolicity(&distances);
        assert!(delta < 1e-10, "Tree should be 0-hyperbolic, got {}", delta);
    }

    #[test]
    fn test_is_ultrametric() {
        // Ultrametric example
        let ultra = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 2.0],
            vec![2.0, 2.0, 0.0],
        ];
        assert!(is_ultrametric(&ultra, 1e-10));

        // Non-ultrametric: violates strong triangle inequality
        // d(0,2) = 3 > max(d(0,1), d(1,2)) = max(1, 2) = 2
        let non_ultra = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 2.0],
            vec![3.0, 2.0, 0.0],
        ];
        assert!(!is_ultrametric(&non_ultra, 1e-10));
    }
}
