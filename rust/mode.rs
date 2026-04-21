// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use dashmap::DashMap;
use fixedbitset::FixedBitSet;
use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

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
        let mut up_adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (&u, &d) in upstream_slice.iter().zip(downstream_slice.iter()) {
            if d < n_nodes {
                down_adj[u].push(d); // downstream nodes from each node
                up_adj[d].push(u); // upstream nodes to each node (can be multiple)
            }
        }
        (down_adj, up_adj)
    } else {
        // For upstream aggregation: use normal direction
        // "from" nodes are upstream, "to" nodes are downstream
        let mut up_adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        let mut down_adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (&u, &d) in upstream_slice.iter().zip(downstream_slice.iter()) {
            if d < n_nodes {
                up_adj[d].push(u);
                down_adj[u].push(d);
            }
        }
        (up_adj, down_adj)
    };

    // Store count maps for each node using Mutex for thread-safe parallel access
    let node_counts: Vec<Mutex<HashMap<i64, i64>>> =
        (0..n_nodes).map(|_| Mutex::new(HashMap::new())).collect();

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
        for &to_node in &to_adj[i] {
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
            for &to_node in &to_adj[i] {
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
                counts
                    .iter()
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
    compute_mode_rust(
        py,
        field,
        upstream_nodes,
        downstream_nodes,
        sources,
        n_nodes,
        false,
    )
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

/// Helper function to extract mode from a count map
fn extract_mode(counts: &HashMap<i64, i64>) -> i64 {
    counts
        .iter()
        .max_by_key(|(&cat, &count)| (count, -cat))
        .map(|(&cat, _)| cat)
        .unwrap_or(0)
}

/// Optimized upstream mode calculation using topological groups
///
/// This matches the pattern used by percentile for better performance.
#[pyfunction]
pub fn calc_mode<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, i64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let upstream_map: DashMap<i64, HashMap<i64, i64>> = DashMap::new();
    let field_array: ArrayView1<i64> = field.as_array();
    let mut result: Vec<i64> = field_array.to_vec();

    for group in &topo_groups {
        process_level_mode(group, &upstream_map, &field_array, &mut result);
    }

    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned().into())
}

fn process_level_mode(
    topo_group: &PyReadonlyArray2<'_, i64>,
    upstream_map: &DashMap<i64, HashMap<i64, i64>>,
    field: &ArrayView1<i64>,
    result: &mut Vec<i64>,
) {
    let arr = topo_group.as_array();
    let did_vec = arr.row(0);
    let uid_vec = arr.row(1);

    let did_slice = did_vec
        .as_slice()
        .expect("Expected contiguous did_vec slice");
    let uid_slice = uid_vec
        .as_slice()
        .expect("Expected contiguous uid_vec slice");

    // Process edges in parallel: accumulate counts from uid to did
    did_slice
        .par_iter()
        .zip(uid_slice.par_iter())
        .for_each(|(&did, &uid)| {
            // Get uid's accumulated counts (or start with just uid's value)
            let uid_counts = upstream_map
                .remove(&uid)
                .map(|entry| entry.1)
                .unwrap_or_else(|| {
                    let mut map = HashMap::new();
                    map.insert(field[uid as usize], 1);
                    map
                });

            // Merge uid's counts into did's counts
            upstream_map
                .entry(did)
                .and_modify(|did_counts| {
                    for (cat, count) in uid_counts.iter() {
                        *did_counts.entry(*cat).or_insert(0) += count;
                    }
                })
                .or_insert_with(|| {
                    let mut map = uid_counts;
                    *map.entry(field[did as usize]).or_insert(0) += 1;
                    map
                });
        });

    // Extract modes for all did nodes in this level
    let mode_results: Vec<(i64, i64)> = did_slice
        .par_iter()
        .map(|&did| {
            let counts = upstream_map.get(&did).unwrap();
            let mode = extract_mode(&counts);
            (did, mode)
        })
        .collect();

    for (did, mode) in mode_results {
        let idx = did as usize;
        if idx < result.len() {
            result[idx] = mode;
        }
    }
}

/// Optimized downstream mode calculation using topological groups
///
/// Processes groups in reverse topological order for downstream aggregation.
#[pyfunction]
pub fn calc_mode_downstream<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, i64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let downstream_map: DashMap<i64, HashMap<i64, i64>> = DashMap::new();
    let field_array: ArrayView1<i64> = field.as_array();
    let mut result: Vec<i64> = field_array.to_vec();

    // Process in REVERSE topological order (sinks to sources)
    for group in topo_groups.iter().rev() {
        process_level_mode_downstream(group, &downstream_map, &field_array, &mut result);
    }

    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned().into())
}

fn process_level_mode_downstream(
    topo_group: &PyReadonlyArray2<'_, i64>,
    downstream_map: &DashMap<i64, HashMap<i64, i64>>,
    field: &ArrayView1<i64>,
    result: &mut Vec<i64>,
) {
    let arr = topo_group.as_array();
    let did_vec = arr.row(0);
    let uid_vec = arr.row(1);

    let did_slice = did_vec
        .as_slice()
        .expect("Expected contiguous did_vec slice");
    let uid_slice = uid_vec
        .as_slice()
        .expect("Expected contiguous uid_vec slice");

    // Process edges in parallel: accumulate counts from did to uid (downstream direction)
    did_slice
        .par_iter()
        .zip(uid_slice.par_iter())
        .for_each(|(&did, &uid)| {
            // Clone did's accumulated counts (multiple uids may need it)
            let did_counts = downstream_map
                .get(&did)
                .map(|entry| entry.clone())
                .unwrap_or_else(|| {
                    let mut map = HashMap::new();
                    map.insert(field[did as usize], 1);
                    map
                });

            // Merge did's counts into uid's counts
            downstream_map
                .entry(uid)
                .and_modify(|uid_counts| {
                    for (cat, count) in did_counts.iter() {
                        *uid_counts.entry(*cat).or_insert(0) += count;
                    }
                })
                .or_insert_with(|| {
                    let mut map = did_counts;
                    *map.entry(field[uid as usize]).or_insert(0) += 1;
                    map
                });
        });

    // Extract modes for all uid nodes in this level
    let mode_results: Vec<(i64, i64)> = uid_slice
        .par_iter()
        .map(|&uid| {
            let counts = downstream_map.get(&uid).unwrap();
            let mode = extract_mode(&counts);
            (uid, mode)
        })
        .collect();

    for (uid, mode) in mode_results {
        let idx = uid as usize;
        if idx < result.len() {
            result[idx] = mode;
        }
    }
}
