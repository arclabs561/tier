//! DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
//!
//! # The Algorithm (Ester et al., 1996)
//!
//! DBSCAN is a density-based clustering algorithm that groups points based on
//! neighborhood density. Unlike k-means, it:
//!
//! - Discovers clusters of arbitrary shape
//! - Automatically determines the number of clusters
//! - Identifies noise points (outliers)
//!
//! ## Core Concepts
//!
//! - **Epsilon (ε)**: Maximum distance between two points to be neighbors.
//! - **MinPts**: Minimum neighbors within ε for a point to be "core".
//! - **Core point**: Has at least MinPts neighbors within ε.
//! - **Border point**: Within ε of a core point but not core itself.
//! - **Noise point**: Neither core nor border.
//!
//! ## Algorithm Steps
//!
//! 1. For each unvisited point P:
//!    - Find neighbors within ε
//!    - If |neighbors| < MinPts, mark as noise (may change later)
//!    - Else P is core: start new cluster, expand from neighbors
//!
//! 2. Expansion: For each core point's neighbors:
//!    - Add to cluster
//!    - If core, recursively expand
//!
//! ## Complexity
//!
//! - **Time**: O(n²) naive, O(n log n) with spatial index.
//! - **Space**: O(n) for labels.
//!
//! ## When to Use
//!
//! - Clusters have non-convex shapes
//! - Number of clusters unknown
//! - Data has outliers
//! - Clusters have similar density
//!
//! ## Limitations
//!
//! - Struggles with varying densities (consider OPTICS)
//! - ε parameter is sensitive and dataset-dependent
//!
//! ## References
//!
//! Ester et al. (1996). "A Density-Based Algorithm for Discovering Clusters
//! in Large Spatial Databases with Noise." KDD-96.

use super::traits::Clustering;
use crate::error::{Error, Result};
use std::collections::HashSet;

/// DBSCAN clustering algorithm.
#[derive(Debug, Clone)]
pub struct Dbscan {
    /// Epsilon: maximum distance for neighborhood.
    epsilon: f32,
    /// Minimum points for core point classification.
    min_pts: usize,
}

/// Labels from DBSCAN clustering.
pub const NOISE: usize = usize::MAX;

impl Dbscan {
    /// Create a new DBSCAN clusterer.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Maximum distance between two points to be neighbors.
    /// * `min_pts` - Minimum number of points to form a dense region.
    ///
    /// # Typical Values
    ///
    /// - `epsilon`: Often determined by k-distance plot (k = min_pts - 1).
    /// - `min_pts`: 2 * dimension is a common heuristic. Minimum is 3.
    pub fn new(epsilon: f32, min_pts: usize) -> Self {
        Self { epsilon, min_pts }
    }

    /// Set epsilon (neighborhood radius).
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set minimum points for core classification.
    pub fn with_min_pts(mut self, min_pts: usize) -> Self {
        self.min_pts = min_pts;
        self
    }

    /// Compute Euclidean distance between two points.
    #[inline]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Find all neighbors within epsilon.
    fn region_query(&self, data: &[Vec<f32>], point_idx: usize) -> Vec<usize> {
        let point = &data[point_idx];
        data.iter()
            .enumerate()
            .filter(|(idx, other)| {
                *idx != point_idx && Self::distance(point, other) <= self.epsilon
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Expand cluster from a core point.
    fn expand_cluster(
        &self,
        data: &[Vec<f32>],
        point_idx: usize,
        neighbors: &[usize],
        labels: &mut [usize],
        cluster_id: usize,
        visited: &mut HashSet<usize>,
    ) {
        labels[point_idx] = cluster_id;

        // Use a queue for iterative expansion (avoid deep recursion)
        let mut to_process: Vec<usize> = neighbors.to_vec();

        while let Some(neighbor_idx) = to_process.pop() {
            if visited.contains(&neighbor_idx) {
                continue;
            }
            let _ = visited.insert(neighbor_idx);

            // If was noise, now it's a border point
            if labels[neighbor_idx] == NOISE {
                labels[neighbor_idx] = cluster_id;
            }

            // If not yet assigned to any cluster, add to this one
            if labels[neighbor_idx] == NOISE {
                labels[neighbor_idx] = cluster_id;
            }

            let neighbor_neighbors = self.region_query(data, neighbor_idx);

            // If this neighbor is also a core point, expand from it
            // MinPts includes the point itself
            if neighbor_neighbors.len() + 1 >= self.min_pts {
                labels[neighbor_idx] = cluster_id;
                for nn in neighbor_neighbors {
                    if !visited.contains(&nn) {
                        to_process.push(nn);
                    }
                }
            }
        }
    }
}

impl Default for Dbscan {
    fn default() -> Self {
        Self::new(0.5, 5)
    }
}

impl Clustering for Dbscan {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        let n = data.len();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        if self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                message: "must be positive",
            });
        }

        if self.min_pts == 0 {
            return Err(Error::InvalidParameter {
                name: "min_pts",
                message: "must be at least 1",
            });
        }

        // Initialize: all points as noise
        let mut labels = vec![NOISE; n];
        let mut visited = HashSet::with_capacity(n);
        let mut cluster_id = 0;

        for point_idx in 0..n {
            if visited.contains(&point_idx) {
                continue;
            }
            let _ = visited.insert(point_idx);

            let neighbors = self.region_query(data, point_idx);

            // MinPts includes the point itself, so we need >= min_pts - 1 other neighbors
            if neighbors.len() + 1 < self.min_pts {
                // Not enough neighbors: mark as noise (might be border later)
                continue;
            }

            // Start new cluster
            self.expand_cluster(
                data,
                point_idx,
                &neighbors,
                &mut labels,
                cluster_id,
                &mut visited,
            );
            cluster_id += 1;
        }

        // Convert NOISE to a proper cluster ID for compatibility
        // Note: Some implementations keep NOISE as special value.
        // We use a separate cluster for noise points.
        if labels.contains(&NOISE) {
            for label in &mut labels {
                if *label == NOISE {
                    *label = cluster_id;
                }
            }
        }

        Ok(labels)
    }

    /// DBSCAN discovers clusters dynamically, so this returns 0.
    ///
    /// To get the actual number of clusters, examine the labels after `fit_predict`.
    fn n_clusters(&self) -> usize {
        0 // Unknown until fit
    }
}

/// Extended DBSCAN interface with noise detection.
pub trait DbscanExt {
    /// Fit and predict, returning labels where noise is marked as `None`.
    fn fit_predict_with_noise(&self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>>;

    /// Check if a label represents noise.
    fn is_noise(label: usize) -> bool {
        label == NOISE
    }
}

impl DbscanExt for Dbscan {
    fn fit_predict_with_noise(&self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>> {
        let n = data.len();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        if self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                message: "must be positive",
            });
        }

        // Initialize: all points as noise
        let mut labels = vec![NOISE; n];
        let mut visited = HashSet::with_capacity(n);
        let mut cluster_id = 0;

        for point_idx in 0..n {
            if visited.contains(&point_idx) {
                continue;
            }
            let _ = visited.insert(point_idx);

            let neighbors = self.region_query(data, point_idx);

            // MinPts includes the point itself
            if neighbors.len() + 1 < self.min_pts {
                continue;
            }

            self.expand_cluster(
                data,
                point_idx,
                &neighbors,
                &mut labels,
                cluster_id,
                &mut visited,
            );
            cluster_id += 1;
        }

        // Return with noise as None
        Ok(labels
            .into_iter()
            .map(|l| if l == NOISE { None } else { Some(l) })
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_two_clusters() {
        // Two well-separated clusters
        let data = vec![
            // Cluster 1: around (0, 0)
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Cluster 2: around (5, 5)
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
            vec![5.05, 5.05],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 10);

        // First 5 should be in same cluster
        let cluster1 = labels[0];
        for label in &labels[1..5] {
            assert_eq!(*label, cluster1);
        }

        // Last 5 should be in same cluster
        let cluster2 = labels[5];
        for label in &labels[6..10] {
            assert_eq!(*label, cluster2);
        }

        // Two clusters should be different
        assert_ne!(cluster1, cluster2);
    }

    #[test]
    fn test_dbscan_with_noise() {
        // Two clusters plus an outlier
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            // Outlier
            vec![100.0, 100.0],
            // Cluster 2
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict_with_noise(&data).unwrap();

        assert_eq!(labels.len(), 9);

        // Point 4 (outlier) should be noise
        assert!(labels[4].is_none());

        // Others should have cluster assignments
        for (i, label) in labels.iter().enumerate() {
            if i != 4 {
                assert!(label.is_some());
            }
        }
    }

    #[test]
    fn test_dbscan_all_noise() {
        // Points too far apart
        let data = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
        ];

        let dbscan = Dbscan::new(0.5, 3);
        let labels = dbscan.fit_predict_with_noise(&data).unwrap();

        // All should be noise
        for label in labels {
            assert!(label.is_none());
        }
    }

    #[test]
    fn test_dbscan_all_one_cluster() {
        // All points close together
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
        ];

        let dbscan = Dbscan::new(0.5, 2);
        let labels = dbscan.fit_predict(&data).unwrap();

        // All in same cluster
        let cluster = labels[0];
        for label in labels {
            assert_eq!(label, cluster);
        }
    }

    #[test]
    fn test_dbscan_empty() {
        let data: Vec<Vec<f32>> = vec![];
        let dbscan = Dbscan::new(0.5, 3);
        let result = dbscan.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dbscan_invalid_params() {
        let data = vec![vec![0.0, 0.0]];

        // Invalid epsilon
        let dbscan = Dbscan::new(0.0, 3);
        assert!(dbscan.fit_predict(&data).is_err());

        let dbscan = Dbscan::new(-1.0, 3);
        assert!(dbscan.fit_predict(&data).is_err());

        // Invalid min_pts
        let dbscan = Dbscan::new(0.5, 0);
        assert!(dbscan.fit_predict(&data).is_err());
    }

    #[test]
    fn test_dbscan_chain() {
        // Chain of points - DBSCAN should connect them
        let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.3, 0.0]).collect();

        let dbscan = Dbscan::new(0.5, 2);
        let labels = dbscan.fit_predict(&data).unwrap();

        // All should be in one cluster (chain is connected)
        let cluster = labels[0];
        for label in labels {
            assert_eq!(label, cluster);
        }
    }
}
