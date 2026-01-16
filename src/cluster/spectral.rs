//! Spectral clustering via graph Laplacian eigenvectors.
//!
//! Spectral clustering works by:
//! 1. Build similarity graph from points
//! 2. Compute normalized Laplacian
//! 3. Find k smallest eigenvectors
//! 4. Run k-means on the embedded points
//!
//! # When to Use Spectral Clustering
//!
//! Spectral clustering excels at finding non-convex clusters that k-means misses.
//! It's particularly good for:
//!
//! - Image segmentation (groups of similar pixels)
//! - Community detection in graphs
//! - Clusters with complex shapes
//!
//! # Trade-offs
//!
//! | Aspect | Spectral | K-means |
//! |--------|----------|---------|
//! | Shape | Any | Convex |
//! | Complexity | O(n³) eigendecomp | O(nkd × iter) |
//! | Memory | O(n²) similarity | O(nd) |
//! | Scalability | < 10k points | Millions |
//!
//! # Algorithm
//!
//! ```text
//! 1. Compute affinity matrix A (Gaussian kernel or kNN)
//! 2. Compute normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
//! 3. Find k smallest eigenvectors of L_sym (excluding trivial)
//! 4. Form matrix U ∈ R^{n×k} from eigenvectors as columns
//! 5. Normalize rows of U to unit length
//! 6. Run k-means on rows of U
//! ```
//!
//! # Example
//!
//! ```ignore
//! use strata::cluster::spectral::SpectralClustering;
//! use ndarray::array;
//!
//! let points = array![
//!     [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],  // Cluster 1
//!     [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],  // Cluster 2
//! ];
//!
//! let labels = SpectralClustering::new(2)
//!     .sigma(1.0)
//!     .fit(&points)?;
//! ```
//!
//! # References
//!
//! - Ng, Jordan, Weiss (2001). "On Spectral Clustering"
//! - von Luxburg (2007). "A Tutorial on Spectral Clustering"

use lapl::{gaussian_similarity, knn_graph, normalized_laplacian};
use ndarray::Array2;

use crate::{Error, Result};

/// Spectral clustering configuration and runner.
///
/// Uses graph Laplacian eigenvectors to embed points, then k-means to cluster.
#[derive(Debug, Clone)]
pub struct SpectralClustering {
    /// Number of clusters
    k: usize,
    /// Affinity type
    affinity: AffinityType,
    /// Sigma for Gaussian kernel (if using rbf affinity)
    sigma: f64,
    /// Number of neighbors for kNN affinity
    n_neighbors: usize,
    /// K-means iterations
    kmeans_iter: usize,
}

/// Type of affinity matrix to build.
#[derive(Debug, Clone, Copy)]
pub enum AffinityType {
    /// Gaussian (RBF) kernel: exp(-||x-y||² / 2σ²)
    Rbf,
    /// k-nearest neighbors graph
    Knn,
    /// Precomputed affinity matrix
    Precomputed,
}

impl SpectralClustering {
    /// Create new spectral clustering with k clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            affinity: AffinityType::Rbf,
            sigma: 1.0,
            n_neighbors: 10,
            kmeans_iter: 100,
        }
    }

    /// Set affinity type.
    pub fn affinity(mut self, affinity: AffinityType) -> Self {
        self.affinity = affinity;
        self
    }

    /// Set sigma for Gaussian kernel.
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set number of neighbors for kNN affinity.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set number of k-means iterations.
    pub fn kmeans_iter(mut self, iter: usize) -> Self {
        self.kmeans_iter = iter;
        self
    }

    /// Fit spectral clustering to points.
    ///
    /// # Arguments
    ///
    /// * `points` - n × d matrix of n points in d dimensions
    ///
    /// # Returns
    ///
    /// Cluster assignments for each point.
    pub fn fit(&self, points: &Array2<f64>) -> Result<Vec<usize>> {
        let n = points.nrows();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if n < self.k {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        // Build affinity matrix
        let affinity = self.build_affinity(points);

        // Compute normalized Laplacian
        let laplacian = normalized_laplacian(&affinity);

        // Get embedding from Laplacian eigenvectors
        // Since we don't have a proper eigensolver, we use power iteration
        // to approximate the smallest eigenvectors
        let embedding = self.laplacian_embedding(&laplacian)?;

        // Run k-means on the embedding
        self.kmeans_on_embedding(&embedding)
    }

    /// Fit using precomputed affinity matrix.
    pub fn fit_affinity(&self, affinity: &Array2<f64>) -> Result<Vec<usize>> {
        let n = affinity.nrows();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if affinity.ncols() != n {
            return Err(Error::DimensionMismatch {
                expected: n,
                found: affinity.ncols(),
            });
        }

        let laplacian = normalized_laplacian(affinity);
        let embedding = self.laplacian_embedding(&laplacian)?;
        self.kmeans_on_embedding(&embedding)
    }

    fn build_affinity(&self, points: &Array2<f64>) -> Array2<f64> {
        match self.affinity {
            AffinityType::Rbf => gaussian_similarity(points, self.sigma),
            AffinityType::Knn => {
                // Build distance matrix first
                let n = points.nrows();
                let mut distances = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let mut dist_sq = 0.0;
                        for d in 0..points.ncols() {
                            let diff = points[[i, d]] - points[[j, d]];
                            dist_sq += diff * diff;
                        }
                        distances[[i, j]] = dist_sq.sqrt();
                    }
                }
                knn_graph(&distances, self.n_neighbors)
            }
            AffinityType::Precomputed => {
                // This shouldn't happen via fit()
                panic!("use fit_affinity for precomputed affinity")
            }
        }
    }

    fn laplacian_embedding(&self, laplacian: &Array2<f64>) -> Result<Array2<f64>> {
        let n = laplacian.nrows();
        let k = self.k;

        // Simple approach: use power iteration to find smallest eigenvectors
        // of (I - L) which correspond to largest of (I - L), i.e., smallest of L
        //
        // For a proper implementation, use ndarray-linalg or similar.
        // This is a simplified version for demonstration.

        let mut embedding = Array2::zeros((n, k));

        // Identity minus Laplacian (shifts spectrum)
        let mut shifted = Array2::eye(n) - laplacian;

        // Make it positive definite by shifting further
        // The normalized Laplacian has eigenvalues in [0, 2]
        // So (I - L) has eigenvalues in [-1, 1]
        // We want the k largest eigenvalues of (I - L) = k smallest of L
        shifted = &shifted + Array2::eye(n) * 1.0; // Now in [0, 2]

        // Power iteration to find dominant eigenvectors
        let mut vs: Vec<ndarray::Array1<f64>> = Vec::with_capacity(k);

        for idx in 0..k {
            // Random starting vector
            let mut v: ndarray::Array1<f64> = (0..n)
                .map(|i| ((i * 7 + idx * 13) % 97) as f64 / 97.0 - 0.5)
                .collect();

            // Power iteration
            for _ in 0..50 {
                // v = A @ v
                v = shifted.dot(&v);

                // Orthogonalize against previous eigenvectors
                for prev in &vs {
                    let proj = v.dot(prev);
                    v = &v - &(prev * proj);
                }

                // Normalize
                let norm = v.dot(&v).sqrt();
                if norm > 1e-10 {
                    v /= norm;
                }
            }

            // Store eigenvector
            for i in 0..n {
                embedding[[i, idx]] = v[i];
            }
            vs.push(v);
        }

        // Row-normalize the embedding
        for i in 0..n {
            let mut row_norm_sq = 0.0;
            for j in 0..k {
                row_norm_sq += embedding[[i, j]] * embedding[[i, j]];
            }
            let row_norm = row_norm_sq.sqrt();
            if row_norm > 1e-10 {
                for j in 0..k {
                    embedding[[i, j]] /= row_norm;
                }
            }
        }

        Ok(embedding)
    }

    fn kmeans_on_embedding(&self, embedding: &Array2<f64>) -> Result<Vec<usize>> {
        let n = embedding.nrows();
        let d = embedding.ncols();
        let k = self.k;

        // Initialize centroids (simple: first k points)
        let mut centroids = Array2::zeros((k, d));
        for i in 0..k.min(n) {
            for j in 0..d {
                centroids[[i, j]] = embedding[[i, j]];
            }
        }

        let mut labels = vec![0usize; n];

        for _ in 0..self.kmeans_iter {
            // Assign points to nearest centroid
            for i in 0..n {
                let mut best_dist = f64::INFINITY;
                let mut best_k = 0;
                for ki in 0..k {
                    let mut dist = 0.0;
                    for j in 0..d {
                        let diff = embedding[[i, j]] - centroids[[ki, j]];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = ki;
                    }
                }
                labels[i] = best_k;
            }

            // Update centroids
            let mut counts = vec![0usize; k];
            centroids.fill(0.0);

            for i in 0..n {
                let ki = labels[i];
                counts[ki] += 1;
                for j in 0..d {
                    centroids[[ki, j]] += embedding[[i, j]];
                }
            }

            for ki in 0..k {
                if counts[ki] > 0 {
                    for j in 0..d {
                        centroids[[ki, j]] /= counts[ki] as f64;
                    }
                }
            }
        }

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_spectral_two_clusters() {
        // Two clearly separated clusters
        let points = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ];

        let labels = SpectralClustering::new(2)
            .sigma(1.0)
            .kmeans_iter(50)
            .fit(&points)
            .expect("fit should succeed");

        // First 3 should be in same cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_spectral_with_knn() {
        let points = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ];

        let labels = SpectralClustering::new(2)
            .affinity(AffinityType::Knn)
            .n_neighbors(2)
            .fit(&points)
            .expect("fit should succeed");

        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_spectral_empty_error() {
        let points: Array2<f64> = Array2::zeros((0, 2));
        let result = SpectralClustering::new(2).fit(&points);
        assert!(result.is_err());
    }
}
