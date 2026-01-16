//! # tier
//!
//! Hierarchical abstraction: clustering, summarization, and tree structures for multi-resolution views.
//!
//! This crate consolidates hierarchical primitives for the Tekne stack.

pub mod error;
pub mod cluster;
pub mod reconciliation;
pub mod hierarchy;
pub mod community;
pub mod summarize;
pub mod metrics;

#[cfg(test)]
mod reconciliation_tests;

pub use crate::cluster::ItDendrogram;
pub use crate::reconciliation::{reconcile, ReconciliationMethod, SummingMatrix};
pub use crate::hierarchy::{HierarchicalConformal, HierarchyTree, ReconciliationScore};

// Re-exports from Strata merge
pub use error::{Error, Result};
pub use metrics::{ari, completeness, fowlkes_mallows, homogeneity, nmi, purity, v_measure};

#[cfg(feature = "cluster")]
pub use cluster::{Clustering, Gmm, HierarchicalClustering, Kmeans, Linkage, SoftClustering};

#[cfg(feature = "community")]
pub use community::{CommunityDetection, LabelPropagation, Leiden, Louvain};

#[cfg(feature = "knn-graph")]
pub use community::{
    knn_graph_from_embeddings, knn_graph_with_config, KnnGraphConfig, WeightFunction,
};

#[cfg(feature = "summarize")]
pub use summarize::Summarizer;

pub use hierarchy::{Dendrogram, RaptorTree, TreeConfig};

