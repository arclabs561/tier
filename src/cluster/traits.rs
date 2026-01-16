//! Clustering traits.

use crate::error::Result;

/// Trait for clustering algorithms.
pub trait Clustering {
    /// Fit the model to data and return cluster assignments.
    ///
    /// Returns a vector of cluster labels, one per input point.
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>>;

    /// Get the number of clusters.
    fn n_clusters(&self) -> usize;
}

/// Trait for soft clustering algorithms that return probabilities.
pub trait SoftClustering: Clustering {
    /// Fit and return soft cluster assignments (probabilities).
    ///
    /// Returns a matrix where entry \[i\]\[k\] is the probability that
    /// point i belongs to cluster k.
    fn fit_predict_proba(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f64>>>;
}
