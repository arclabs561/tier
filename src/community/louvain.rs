//! Louvain algorithm for community detection.
//!
//! Fast modularity optimization through local node moves and graph aggregation.
//!
//! ## The Algorithm (Blondel et al. 2008)
//!
//! Louvain is a multi-level, greedy modularity optimization algorithm:
//!
//! 1. **Phase 1 (Local Moving)**: Start with each node in its own community.
//!    Repeatedly move nodes to neighboring community with highest modularity
//!    gain until no improvement.
//!
//! 2. **Phase 2 (Aggregation)**: Build a meta-graph where communities become
//!    single nodes. Edge weights are sums of edges between communities.
//!    Self-loops represent internal community edges.
//!
//! 3. **Iterate**: Repeat phases 1-2 on the meta-graph until modularity
//!    stops improving.
//!
//! ## Multi-Level Benefits
//!
//! - Finds hierarchical community structure at different resolutions
//! - Often achieves higher modularity than single-level
//! - Faster convergence due to coarsening
//!
//! ## References
//!
//! Blondel et al. (2008). "Fast unfolding of communities in large networks."
//! Journal of Statistical Mechanics: Theory and Experiment, P10008.

use super::traits::CommunityDetection;
use crate::error::{Error, Result};
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Louvain community detection algorithm.
#[derive(Debug, Clone)]
pub struct Louvain {
    /// Resolution parameter (gamma).
    resolution: f64,
    /// Maximum iterations per level.
    max_iter: usize,
    /// Maximum levels of aggregation.
    max_levels: usize,
    /// Minimum modularity improvement to continue.
    min_modularity_gain: f64,
}

impl Louvain {
    /// Create a new Louvain detector with default settings.
    pub fn new() -> Self {
        Self {
            resolution: 1.0,
            max_iter: 100,
            max_levels: 10,
            min_modularity_gain: 1e-7,
        }
    }

    /// Set resolution parameter.
    ///
    /// Higher values produce smaller communities.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set maximum iterations per level.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set maximum aggregation levels.
    pub fn with_max_levels(mut self, levels: usize) -> Self {
        self.max_levels = levels;
        self
    }

    /// Compute modularity of a weighted graph partition.
    fn modularity_weighted(
        &self,
        n: usize,
        edges: &[(usize, usize, f64)],
        self_loops: &[f64],
        communities: &[usize],
    ) -> f64 {
        // Total edge weight (counting each edge once, plus self-loops)
        let m: f64 = edges.iter().map(|(_, _, w)| w).sum::<f64>() + self_loops.iter().sum::<f64>();
        if m == 0.0 {
            return 0.0;
        }

        // Compute weighted degrees
        let mut degrees = vec![0.0; n];
        for &(i, j, w) in edges {
            degrees[i] += w;
            degrees[j] += w;
        }
        for (i, &sl) in self_loops.iter().enumerate() {
            degrees[i] += 2.0 * sl; // self-loops counted twice for degree
        }

        let mut q = 0.0;

        // Intra-community edges
        for &(i, j, w) in edges {
            if communities[i] == communities[j] {
                let expected = degrees[i] * degrees[j] / (2.0 * m);
                q += w - self.resolution * expected;
            }
        }

        // Self-loops (always within community)
        for (i, &sl) in self_loops.iter().enumerate() {
            if sl > 0.0 {
                let expected = degrees[i] * degrees[i] / (2.0 * m);
                q += sl - self.resolution * expected / 2.0;
            }
        }

        q / m
    }

    /// Phase 1: Local moving on weighted graph.
    /// Returns (communities, improved).
    fn local_moving(
        &self,
        n: usize,
        edges: &[(usize, usize, f64)],
        self_loops: &[f64],
    ) -> (Vec<usize>, bool) {
        // Build adjacency with weights
        let mut adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
        for &(i, j, w) in edges {
            *adj[i].entry(j).or_insert(0.0) += w;
            *adj[j].entry(i).or_insert(0.0) += w;
        }

        // Total weight (for modularity calculation)
        let m: f64 = edges.iter().map(|(_, _, w)| w).sum::<f64>() + self_loops.iter().sum::<f64>();
        if m == 0.0 {
            return ((0..n).collect(), false);
        }

        // Weighted degrees
        let mut degrees = vec![0.0; n];
        for &(i, j, w) in edges {
            degrees[i] += w;
            degrees[j] += w;
        }
        for (i, &sl) in self_loops.iter().enumerate() {
            degrees[i] += 2.0 * sl;
        }

        // Initialize communities
        let mut communities: Vec<usize> = (0..n).collect();
        let mut community_degrees = degrees.clone();
        let mut any_improved = false;

        for _iter in 0..self.max_iter {
            let mut improved = false;

            for node in 0..n {
                let current_community = communities[node];
                let ki = degrees[node];

                // Temporarily remove node from community
                community_degrees[current_community] -= ki;

                // Find neighboring communities and their edge weights
                let mut community_weights: HashMap<usize, f64> = HashMap::new();
                for (&neighbor, &w) in &adj[node] {
                    let nc = communities[neighbor];
                    *community_weights.entry(nc).or_insert(0.0) += w;
                }

                // Find best community
                let mut best_community = current_community;
                let mut best_gain = 0.0;

                for (&target_comm, &ki_in) in &community_weights {
                    let sigma_tot = community_degrees[target_comm];
                    let gain = ki_in / m - self.resolution * sigma_tot * ki / (2.0 * m * m);
                    if gain > best_gain {
                        best_gain = gain;
                        best_community = target_comm;
                    }
                }

                // Also check staying in current (now empty) community
                // gain = 0 for staying alone

                if best_community != current_community {
                    communities[node] = best_community;
                    community_degrees[best_community] += ki;
                    improved = true;
                    any_improved = true;
                } else {
                    community_degrees[current_community] += ki;
                }
            }

            if !improved {
                break;
            }
        }

        (communities, any_improved)
    }

    /// Phase 2: Aggregate graph based on communities.
    /// Returns (new_edges, new_self_loops, node_to_original_mapping).
    fn aggregate(
        &self,
        _n: usize, // Original node count (unused, kept for API symmetry)
        edges: &[(usize, usize, f64)],
        self_loops: &[f64],
        communities: &[usize],
    ) -> (Vec<(usize, usize, f64)>, Vec<f64>, Vec<Vec<usize>>) {
        // Find unique communities and create mapping
        let mut unique_comms: Vec<usize> = communities.to_vec();
        unique_comms.sort_unstable();
        unique_comms.dedup();
        let n_new = unique_comms.len();

        let comm_to_new: HashMap<usize, usize> = unique_comms
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        // Track which original nodes map to each new node
        let mut new_to_old: Vec<Vec<usize>> = vec![Vec::new(); n_new];
        for (node, &comm) in communities.iter().enumerate() {
            new_to_old[comm_to_new[&comm]].push(node);
        }

        // Aggregate edges
        let mut new_edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
        for &(i, j, w) in edges {
            let ci = comm_to_new[&communities[i]];
            let cj = comm_to_new[&communities[j]];
            if ci == cj {
                // Will become self-loop
                continue;
            }
            let key = if ci < cj { (ci, cj) } else { (cj, ci) };
            *new_edge_weights.entry(key).or_insert(0.0) += w;
        }

        let new_edges: Vec<(usize, usize, f64)> = new_edge_weights
            .into_iter()
            .map(|((i, j), w)| (i, j, w))
            .collect();

        // Aggregate self-loops (including edges within communities)
        let mut new_self_loops = vec![0.0; n_new];
        for (i, &sl) in self_loops.iter().enumerate() {
            let ci = comm_to_new[&communities[i]];
            new_self_loops[ci] += sl;
        }
        for &(i, j, w) in edges {
            let ci = comm_to_new[&communities[i]];
            let cj = comm_to_new[&communities[j]];
            if ci == cj {
                new_self_loops[ci] += w;
            }
        }

        (new_edges, new_self_loops, new_to_old)
    }

    /// Expand partition from aggregated level to original nodes.
    fn expand_partition(partition: &[usize], node_mapping: &[Vec<usize>]) -> Vec<usize> {
        // Find max original node index
        let max_node = node_mapping.iter().flatten().copied().max().unwrap_or(0);
        let mut result = vec![0; max_node + 1];

        for (agg_node, original_nodes) in node_mapping.iter().enumerate() {
            let comm = partition[agg_node];
            for &orig in original_nodes {
                result[orig] = comm;
            }
        }
        result
    }
}

impl Default for Louvain {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunityDetection for Louvain {
    fn detect<N, E>(&self, graph: &UnGraph<N, E>) -> Result<Vec<usize>> {
        let n = graph.node_count();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        if graph.edge_count() == 0 {
            // No edges: each node is its own community
            return Ok((0..n).collect());
        }

        // Convert graph to weighted edge list (unit weights)
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for edge in graph.edge_references() {
            let i = edge.source().index();
            let j = edge.target().index();
            if i < j {
                edges.push((i, j, 1.0));
            }
        }
        let self_loops = vec![0.0; n];

        // Multi-level Louvain
        let mut current_n = n;
        let mut current_edges = edges;
        let mut current_self_loops = self_loops;

        // Stack of node mappings for expanding final partition
        let mut mapping_stack: Vec<Vec<Vec<usize>>> = Vec::new();

        // Initial mapping: each node maps to itself
        // TODO: Use this for proper multi-level expansion
        let _initial_mapping: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        let mut prev_modularity = f64::NEG_INFINITY;

        for _level in 0..self.max_levels {
            // Phase 1: Local moving
            let (partition, improved) =
                self.local_moving(current_n, &current_edges, &current_self_loops);

            if !improved {
                break;
            }

            // Check modularity improvement
            let mod_now = self.modularity_weighted(
                current_n,
                &current_edges,
                &current_self_loops,
                &partition,
            );

            if mod_now - prev_modularity < self.min_modularity_gain {
                break;
            }
            prev_modularity = mod_now;

            // Phase 2: Aggregate
            let (new_edges, new_self_loops, node_mapping) =
                self.aggregate(current_n, &current_edges, &current_self_loops, &partition);

            // If no aggregation happened (each node is its own community), stop
            if node_mapping.len() == current_n {
                break;
            }

            mapping_stack.push(node_mapping.clone());
            current_n = node_mapping.len();
            current_edges = new_edges;
            current_self_loops = new_self_loops;
        }

        // Expand partition back to original nodes
        // Start with identity partition at current level
        let mut result: Vec<usize> = (0..current_n).collect();

        // Expand through all aggregation levels
        while let Some(mapping) = mapping_stack.pop() {
            result = Self::expand_partition(&result, &mapping);
        }

        // Ensure result has correct length
        if result.len() < n {
            result.resize(n, 0);
        }
        result.truncate(n);

        // Renumber to consecutive integers
        let mut unique: Vec<usize> = result.to_vec();
        unique.sort_unstable();
        unique.dedup();

        Ok(result
            .iter()
            .map(|&c| unique.iter().position(|&u| u == c).unwrap_or(0))
            .collect())
    }

    fn resolution(&self) -> f64 {
        self.resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_louvain_triangle() {
        // Simple triangle - should be one community
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        let _ = graph.add_edge(n0, n1, ());
        let _ = graph.add_edge(n1, n2, ());
        let _ = graph.add_edge(n0, n2, ());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 3);
        // Triangle is well-connected, should be one community
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);
    }

    #[test]
    fn test_louvain_two_cliques() {
        // Two triangles connected by a single edge
        let mut graph = UnGraph::<(), ()>::new_undirected();

        // First clique
        let a0 = graph.add_node(());
        let a1 = graph.add_node(());
        let a2 = graph.add_node(());
        let _ = graph.add_edge(a0, a1, ());
        let _ = graph.add_edge(a1, a2, ());
        let _ = graph.add_edge(a0, a2, ());

        // Second clique
        let b0 = graph.add_node(());
        let b1 = graph.add_node(());
        let b2 = graph.add_node(());
        let _ = graph.add_edge(b0, b1, ());
        let _ = graph.add_edge(b1, b2, ());
        let _ = graph.add_edge(b0, b2, ());

        // Bridge
        let _ = graph.add_edge(a2, b0, ());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 6);

        // First clique should be in same community
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);

        // Second clique should be in same community
        assert_eq!(communities[3], communities[4]);
        assert_eq!(communities[4], communities[5]);

        // Two cliques should be in different communities
        assert_ne!(communities[0], communities[3]);
    }

    #[test]
    fn test_louvain_empty_graph() {
        let graph = UnGraph::<(), ()>::new_undirected();
        let louvain = Louvain::new();
        let result = louvain.detect(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_louvain_single_node() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let _ = graph.add_node(());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0], 0);
    }

    #[test]
    fn test_louvain_disconnected() {
        // Two isolated nodes
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let _ = graph.add_node(());
        let _ = graph.add_node(());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 2);
        // Disconnected nodes should be in separate communities
        assert_ne!(communities[0], communities[1]);
    }
}
