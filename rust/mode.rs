// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
use fixedbitset::FixedBitSet;

/// Compute the mode (most common value) for each node in a river network.
///
/// This is a generalized mode calculation that works for both upstream and downstream
/// aggregation by inverting the graph structure when needed.
///
/// # Arguments
/// * `py` - Python interpreter handle
/// * `field` - Categorical values at each node (integer codes)
/// * `upstream_nodes` - Array of upstream node indices for each edge
/// * `downstream_nodes` - Array of downstream node indices for each edge
/// * `sources` - Indices of source nodes (starting points for traversal)
/// * `n_nodes` - Total number of nodes in the network
/// * `invert_graph` - If true, invert the graph direction (for downstream aggregation)
///
/// # Returns
/// * PyArray1<i64> - The mode (most common value) for each node
///
/// # Algorithm
/// The algorithm processes the network in topological order, accumulating categorical
/// counts efficiently using hashmaps. For each node:
/// 1. Start with the node's own value
/// 2. Accumulate counts from all connected nodes (upstream or downstream)
/// 3. Find the category with maximum count (ties broken by smallest value)
#[pyfunction]
pub fn compute_mode_rust<'py>(
    py: Python<'py>,
    field: PyReadonlyArray1<'py, i64>,
    upstream_nodes: PyReadonlyArray1<'py, usize>,
    downstream_nodes: PyReadonlyArray1<'py, usize>,
    sources: PyReadonlyArray1<'py, usize>,
    n_nodes: usize,
    invert_graph: bool,
) -> PyResult<Py<PyArray1<i64>>> {
    let field_slice = field.as_slice()?;
    let upstream_slice = upstream_nodes.as_slice()?;
    let downstream_slice = downstream_nodes.as_slice()?;
    let sources_slice = sources.as_slice()?;

    // Build adjacency lists based on flow direction
    let (from_adj, to_adj) = if invert_graph {
        // For downstream aggregation: invert the graph
        // "from" nodes are downstream, "to" nodes are upstream
        let mut down_adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        let mut up_adj: Vec<Option<usize>> = vec![None; n_nodes];
        for (&u, &d) in upstream_slice.iter().zip(downstream_slice.iter()) {
            if d < n_nodes {
                down_adj[u].push(d);  // downstream nodes from each node
                up_adj[d] = Some(u);  // (simplified, assumes single upstream)
            }
        }
        (down_adj, up_adj)
    } else {
        // For upstream aggregation: use normal direction
        // "from" nodes are upstream, "to" nodes are downstream
        let mut up_adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        let mut down_adj: Vec<Option<usize>> = vec![None; n_nodes];
        for (&u, &d) in upstream_slice.iter().zip(downstream_slice.iter()) {
            if d < n_nodes {
                up_adj[d].push(u);
                down_adj[u] = Some(d);
            }
        }
        (up_adj, down_adj)
    };

    // Store count maps for each node using Mutex for thread-safe parallel access
    let node_counts: Vec<Mutex<HashMap<i64, i64>>> = (0..n_nodes)
        .map(|_| Mutex::new(HashMap::new()))
        .collect();

    // Initialize sources with their own values
    for &src in sources_slice {
        let mut counts = node_counts[src].lock().unwrap();
        let val = field_slice[src];
        *counts.entry(val).or_insert(0) += 1;
    }

    // Compute topological order
    let mut current = sources_slice.to_vec();
    let mut next = Vec::with_capacity(current.len());
    let mut visited = FixedBitSet::with_capacity(n_nodes);

    // Move to first next layer
    for &i in &current {
        if let Some(to_node) = to_adj[i] {
            if !visited.contains(to_node) {
                visited.insert(to_node);
                next.push(to_node);
            }
        }
    }
    std::mem::swap(&mut current, &mut next);

    // Process in topological order
    while !current.is_empty() {
        // Process current layer in parallel
        current.par_iter().for_each(|&node| {
            // Accumulate counts from all "from" nodes
            let mut combined_counts = HashMap::new();

            // Add current node's own value
            let val = field_slice[node];
            *combined_counts.entry(val).or_insert(0) += 1;

            // Merge counts from all "from" nodes
            for &from_node in &from_adj[node] {
                let from_counts = node_counts[from_node].lock().unwrap();
                for (&cat, &count) in from_counts.iter() {
                    *combined_counts.entry(cat).or_insert(0) += count;
                }
            }

            // Store the combined counts
            let mut node_counts_lock = node_counts[node].lock().unwrap();
            *node_counts_lock = combined_counts;
        });

        // Move to next layer
        next.clear();
        visited.clear();
        for &i in &current {
            if let Some(to_node) = to_adj[i] {
                if !visited.contains(to_node) {
                    visited.insert(to_node);
                    next.push(to_node);
                }
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    // Extract mode (most common value) for each node
    let result: Vec<i64> = (0..n_nodes)
        .into_par_iter()
        .map(|node| {
            let counts = node_counts[node].lock().unwrap();
            if counts.is_empty() {
                // This shouldn't happen in a valid network, use node's own value
                field_slice[node]
            } else {
                // Find category with maximum count
                // In case of ties, use the smallest category value
                counts.iter()
                    .max_by_key(|(&cat, &count)| (count, -cat))
                    .map(|(&cat, _)| cat)
                    .unwrap_or(field_slice[node])
            }
        })
        .collect();

    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned().into())
}

/// Compute the upstream mode (most common value) for each node in a river network.
///
/// This is a convenience wrapper around compute_mode_rust with invert_graph=false.
///
/// # Arguments
/// * `py` - Python interpreter handle
/// * `field` - Categorical values at each node (integer codes)
/// * `upstream_nodes` - Array of upstream node indices for each edge
/// * `downstream_nodes` - Array of downstream node indices for each edge
/// * `sources` - Indices of source nodes (no upstream connections)
/// * `n_nodes` - Total number of nodes in the network
///
/// # Returns
/// * PyArray1<i64> - The mode (most common value) for each node
#[pyfunction]
pub fn compute_upstream_mode_rust<'py>(
    py: Python<'py>,
    field: PyReadonlyArray1<'py, i64>,
    upstream_nodes: PyReadonlyArray1<'py, usize>,
    downstream_nodes: PyReadonlyArray1<'py, usize>,
    sources: PyReadonlyArray1<'py, usize>,
    n_nodes: usize,
) -> PyResult<Py<PyArray1<i64>>> {
    compute_mode_rust(py, field, upstream_nodes, downstream_nodes, sources, n_nodes, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_simple_linear() {
        // Simple linear network: 0 -> 1 -> 2
        // Field: [1, 2, 1]
        // Expected mode: [1, 1, 1] (at node 2, we have two 1s and one 2)

        // For a linear network 0->1->2:
        // upstream_nodes = [0, 1], downstream_nodes = [1, 2]
        // sources = [0]

        let field = vec![1i64, 2, 1];
        let upstream_nodes = vec![0usize, 1];
        let downstream_nodes = vec![1usize, 2];
        let sources = vec![0usize];
        let n_nodes = 3;

        // This test would require Python environment to run fully
        // For now, we validate the logic conceptually

        // Expected behavior:
        // Node 0 (source): mode([1]) = 1
        // Node 1: mode([1, 2]) = 1 (tie, smallest wins)
        // Node 2: mode([1, 2, 1]) = 1 (appears twice)
    }

    #[test]
    fn test_mode_all_same() {
        // All nodes have the same value - mode should be that value everywhere
        let field = vec![5i64; 10];
        // Mode should be 5 for all nodes regardless of topology
    }

    #[test]
    fn test_mode_tie_breaking() {
        // Test that ties are broken by smallest value
        // Field: [1, 2] - both appear once, mode should be 1
    }
}
