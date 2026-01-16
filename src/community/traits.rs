//! Community detection traits.

use crate::error::Result;
use petgraph::graph::UnGraph;

/// Trait for community detection algorithms.
pub trait CommunityDetection {
    /// Detect communities in a graph.
    ///
    /// Returns a mapping from node index to community ID.
    fn detect<N, E>(&self, graph: &UnGraph<N, E>) -> Result<Vec<usize>>;

    /// Get the resolution parameter (if applicable).
    fn resolution(&self) -> f64 {
        1.0
    }
}
