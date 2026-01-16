//! K-means clustering.
//!
//! Partitions data into k clusters by minimizing **within-cluster sum of squares**
//! (WCSS). The foundational clustering algorithm, dating to 1957 (Lloyd).
//!
//! # The Objective
//!
//! K-means minimizes:
//!
//! ```text
//! WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
//! ```
//!
//! Sum of squared distances from each point to its cluster centroid.
//!
//! # Lloyd's Algorithm
//!
//! 1. Initialize k centroids (randomly or via k-means++)
//! 2. **Assign**: Each point → nearest centroid
//! 3. **Update**: Each centroid → mean of assigned points
//! 4. Repeat until convergence
//!
//! **Why it converges**: WCSS decreases monotonically. Each step either
//! decreases WCSS or leaves it unchanged. Bounded below by 0 → must converge.
//!
//! # Failure Modes
//!
//! - **Local optima**: NP-hard problem; Lloyd finds local minimum only
//! - **Wrong k**: Must specify k in advance; use elbow method or silhouette
//! - **Non-spherical clusters**: Assumes roughly spherical, equal-sized clusters
//! - **Initialization sensitivity**: Bad initial centroids → bad results
//!
//! ## K-means++ Initialization
//!
//! Addresses initialization by spreading initial centroids:
//! 1. Choose first centroid uniformly at random
//! 2. Choose next centroid with probability proportional to D(x)²
//!    (squared distance to nearest existing centroid)
//!
//! Provides provable O(log k) approximation to optimal WCSS.
//!
//! # Connection to IVF
//!
//! K-means is the foundation of IVF (Inverted File) indexing for ANN search.
//! Partition vectors into k cells, search only nearby cells at query time.

use super::traits::Clustering;
use crate::error::{Error, Result};
use ndarray::Array2;
use rand::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// K-means clustering algorithm.
#[derive(Debug, Clone)]
pub struct Kmeans {
    /// Number of clusters.
    k: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Random seed.
    seed: Option<u64>,
}

impl Kmeans {
    /// Create a new K-means clusterer.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialize centroids using k-means++ algorithm.
    fn init_centroids(&self, data: &Array2<f32>, rng: &mut impl Rng) -> Array2<f32> {
        let n = data.nrows();
        let d = data.ncols();
        let mut centroids = Array2::zeros((self.k, d));

        // First centroid: random point
        let first = rng.random_range(0..n);
        centroids.row_mut(0).assign(&data.row(first));

        // Remaining centroids: k-means++ selection
        for i in 1..self.k {
            let mut distances: Vec<f32> = Vec::with_capacity(n);

            for j in 0..n {
                let point = data.row(j);
                let min_dist = (0..i)
                    .map(|c| {
                        let centroid = centroids.row(c);
                        point
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                    })
                    .fold(f32::MAX, f32::min);
                distances.push(min_dist);
            }

            // Sample proportional to squared distance
            let total: f32 = distances.iter().sum();
            if total == 0.0 {
                let idx = rng.random_range(0..n);
                centroids.row_mut(i).assign(&data.row(idx));
                continue;
            }

            let threshold = rng.random::<f32>() * total;
            let mut cumsum = 0.0;
            let mut selected = 0;

            for (j, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    selected = j;
                    break;
                }
            }

            centroids.row_mut(i).assign(&data.row(selected));
        }

        centroids
    }

    /// Compute squared Euclidean distance.
    fn squared_distance(a: &ndarray::ArrayView1<'_, f32>, b: &ndarray::ArrayView1<'_, f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

impl Clustering for Kmeans {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        let n = data.len();
        let d = data[0].len();

        if self.k > n {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        // Convert to ndarray
        let mut flat: Vec<f32> = Vec::with_capacity(n * d);
        for point in data {
            if point.len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: point.len(),
                });
            }
            flat.extend(point);
        }
        let data_arr =
            Array2::from_shape_vec((n, d), flat).map_err(|e| Error::Other(e.to_string()))?;

        // Initialize RNG
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Initialize centroids
        let mut centroids = self.init_centroids(&data_arr, &mut rng);
        let mut labels = vec![0usize; n];

        for _iter in 0..self.max_iter {
            // Assignment step - parallel when feature enabled
            #[cfg(feature = "parallel")]
            {
                let centroids_ref = &centroids;
                labels.par_iter_mut().enumerate().for_each(|(i, label)| {
                    let point = data_arr.row(i);
                    let mut best_cluster = 0;
                    let mut best_dist = f32::MAX;

                    for k in 0..self.k {
                        let dist = Self::squared_distance(&point, &centroids_ref.row(k));
                        if dist < best_dist {
                            best_dist = dist;
                            best_cluster = k;
                        }
                    }
                    *label = best_cluster;
                });
            }

            #[cfg(not(feature = "parallel"))]
            for (i, label) in labels.iter_mut().enumerate() {
                let point = data_arr.row(i);
                let mut best_cluster = 0;
                let mut best_dist = f32::MAX;

                for k in 0..self.k {
                    let dist = Self::squared_distance(&point, &centroids.row(k));
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = k;
                    }
                }
                *label = best_cluster;
            }

            // Update step
            let mut new_centroids = Array2::zeros((self.k, d));
            let mut counts = vec![0usize; self.k];

            for i in 0..n {
                let k = labels[i];
                for j in 0..d {
                    new_centroids[[k, j]] += data_arr[[i, j]];
                }
                counts[k] += 1;
            }

            for k in 0..self.k {
                if counts[k] > 0 {
                    for j in 0..d {
                        new_centroids[[k, j]] /= counts[k] as f32;
                    }
                } else {
                    // Empty cluster: reinitialize randomly
                    let idx = rng.random_range(0..n);
                    new_centroids.row_mut(k).assign(&data_arr.row(idx));
                }
            }

            // Check convergence
            let shift: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            centroids = new_centroids;

            if shift < self.tol as f32 {
                break;
            }
        }

        Ok(labels)
    }

    fn n_clusters(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans = Kmeans::new(2).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Points 0,1 should be in same cluster, points 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_all_points_assigned() {
        // Property: every point must be assigned to exactly one cluster
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1, (i % 5) as f32])
            .collect();

        let kmeans = Kmeans::new(5).with_seed(123);
        let labels = kmeans.fit_predict(&data).unwrap();

        // All points assigned
        assert_eq!(labels.len(), data.len());

        // All labels in valid range [0, k)
        for &label in &labels {
            assert!(label < 5, "label {} out of range", label);
        }
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        // Edge case: k = n (each point its own cluster)
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let kmeans = Kmeans::new(3).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Each point in different cluster
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_kmeans_deterministic_with_seed() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans1 = Kmeans::new(2).with_seed(42);
        let kmeans2 = Kmeans::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&data).unwrap();

        assert_eq!(labels1, labels2, "same seed should give same result");
    }

    #[test]
    fn test_kmeans_scaling_invariant() {
        // Metamorphic: uniform scaling shouldn't change cluster assignments
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let scaled: Vec<Vec<f32>> = data
            .iter()
            .map(|v| v.iter().map(|x| x * 100.0).collect())
            .collect();

        let kmeans1 = Kmeans::new(2).with_seed(42);
        let kmeans2 = Kmeans::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&scaled).unwrap();

        // Same structure (labels may be permuted)
        assert_eq!(labels1[0], labels1[1]);
        assert_eq!(labels2[0], labels2[1]);
        assert_eq!(labels1[2], labels1[3]);
        assert_eq!(labels2[2], labels2[3]);
        assert_ne!(labels1[0], labels1[2]);
        assert_ne!(labels2[0], labels2[2]);
    }

    #[test]
    fn test_kmeans_empty_input_error() {
        let data: Vec<Vec<f32>> = vec![];
        let kmeans = Kmeans::new(2);
        let result = kmeans.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_larger_than_n_error() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let kmeans = Kmeans::new(5); // k > n
        let result = kmeans.fit_predict(&data);
        assert!(result.is_err());
    }
}
