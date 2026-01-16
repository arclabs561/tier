//! Generic tree node.

use core::fmt;

/// A node in a hierarchical tree.
///
/// Nodes can be either leaf nodes (containing original items) or internal
/// nodes (containing summaries of their children).
#[derive(Debug, Clone)]
pub struct Node<T, S = T> {
    /// Unique identifier for this node.
    pub id: usize,
    /// The content at this node (item for leaves, summary for internal).
    pub content: NodeContent<T, S>,
    /// Depth in tree (0 = leaves).
    pub level: usize,
    /// Child node IDs (empty for leaves).
    pub children: Vec<usize>,
    /// Optional metadata.
    pub metadata: Option<NodeMetadata>,
}

/// Content of a tree node.
#[derive(Debug, Clone)]
pub enum NodeContent<T, S = T> {
    /// Leaf node containing an original item.
    Leaf(T),
    /// Internal node containing a summary of children.
    Summary(S),
}

/// Metadata about a node's construction.
#[derive(Debug, Clone, Default)]
pub struct NodeMetadata {
    /// Number of leaf descendants.
    pub leaf_count: usize,
    /// Clustering method used to group children.
    pub cluster_method: Option<String>,
    /// Summarization method used.
    pub summary_method: Option<String>,
}

impl<T, S> Node<T, S> {
    /// Create a new leaf node.
    pub fn leaf(id: usize, item: T) -> Self {
        Self {
            id,
            content: NodeContent::Leaf(item),
            level: 0,
            children: Vec::new(),
            metadata: None,
        }
    }

    /// Create a new internal (summary) node.
    pub fn internal(id: usize, summary: S, level: usize, children: Vec<usize>) -> Self {
        Self {
            id,
            content: NodeContent::Summary(summary),
            level,
            children,
            metadata: None,
        }
    }

    /// Check if this is a leaf node.
    pub fn is_leaf(&self) -> bool {
        matches!(self.content, NodeContent::Leaf(_))
    }

    /// Get the item if this is a leaf node.
    pub fn as_leaf(&self) -> Option<&T> {
        match &self.content {
            NodeContent::Leaf(item) => Some(item),
            _ => None,
        }
    }

    /// Get the summary if this is an internal node.
    pub fn as_summary(&self) -> Option<&S> {
        match &self.content {
            NodeContent::Summary(summary) => Some(summary),
            _ => None,
        }
    }

    /// Add metadata to this node.
    pub fn with_metadata(mut self, metadata: NodeMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl<T: fmt::Display, S: fmt::Display> fmt::Display for Node<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.content {
            NodeContent::Leaf(item) => write!(f, "Leaf[{}]: {}", self.id, item),
            NodeContent::Summary(summary) => {
                write!(f, "Node[{}] L{}: {}", self.id, self.level, summary)
            }
        }
    }
}
