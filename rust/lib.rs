// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
use fixedbitset::FixedBitSet;
use std::collections::HashMap;

#[pyfunction]
fn compute_topological_labels_rust<'py>(
    py: Python<'py>,
    sources: PyReadonlyArray1<'py, usize>,
    sinks: PyReadonlyArray1<'py, usize>,
    downstream_nodes: PyReadonlyArray1<'py, usize>,
    n_nodes: usize,
) -> PyResult<Py<PyArray1<i64>>> {

    let labels: Vec<AtomicI64> = (0..n_nodes)
        .map(|_| AtomicI64::new(0))
        .collect();

    let mut current = sources.as_slice()?.to_vec();
    let sinks = sinks.as_slice()?;
    let downstream = downstream_nodes.as_slice()?;

    let mut next = Vec::with_capacity(current.len());
    let mut visited = FixedBitSet::with_capacity(n_nodes);

    for &i in &current {
        let d = downstream[i];
        if d != n_nodes {
            next.push(d);
        }
    }
    std::mem::swap(&mut current, &mut next);

    for n in 1..=n_nodes {
        if current.is_empty() {
            sinks.par_iter().for_each(|&i| {
                labels[i].store((n as i64) - 1, Ordering::Relaxed);
            });
            break;
        }

        current.par_iter().for_each(|&i| {
            labels[i].store(n as i64, Ordering::Relaxed);
        });

        next.clear();
        visited.clear();
        for &i in &current {
            let d = downstream[i];
            if d != n_nodes && !visited.contains(d) {
                visited.insert(d);
                next.push(d);
            }
        }

        std::mem::swap(&mut current, &mut next);
    }

    if !current.is_empty() {
        return Err(PyErr::new::<PyValueError, _>("River Network contains a cycle."));
    }

    let result: Vec<i64> = labels.iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();
    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned().into())
}

/// Compute the upstream mode (most common value) for each node in a river network.
///
/// This function computes the categorical mode (most frequent value) of all upstream
/// nodes for each node in the river network. For categorical data, this is the spatial
/// majority aggregation.
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
///
/// # Algorithm
/// The algorithm processes the network in topological order from sources downstream,
/// accumulating categorical counts efficiently using hashmaps. For each node:
/// 1. Start with the node's own value
/// 2. Accumulate counts from all upstream nodes
/// 3. Find the category with maximum count (ties broken by smallest value)
#[pyfunction]
fn compute_upstream_mode_rust<'py>(
    py: Python<'py>,
    field: PyReadonlyArray1<'py, i64>,
    upstream_nodes: PyReadonlyArray1<'py, usize>,
    downstream_nodes: PyReadonlyArray1<'py, usize>,
    sources: PyReadonlyArray1<'py, usize>,
    n_nodes: usize,
) -> PyResult<Py<PyArray1<i64>>> {
    let field_slice = field.as_slice()?;
    let upstream_slice = upstream_nodes.as_slice()?;
    let downstream_slice = downstream_nodes.as_slice()?;
    let sources_slice = sources.as_slice()?;

    // Build adjacency list: for each node, list of upstream nodes
    let mut upstream_adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (&u, &d) in upstream_slice.iter().zip(downstream_slice.iter()) {
        if d < n_nodes {
            upstream_adj[d].push(u);
        }
    }

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

    // Compute topological order (similar to compute_topological_labels_rust)
    let mut current = sources_slice.to_vec();
    let mut next = Vec::with_capacity(current.len());
    let mut visited = FixedBitSet::with_capacity(n_nodes);

    // Build downstream adjacency
    let mut downstream_adj: Vec<Option<usize>> = vec![None; n_nodes];
    for (&u, &d) in upstream_slice.iter().zip(downstream_slice.iter()) {
        if d < n_nodes {
            downstream_adj[u] = Some(d);
        }
    }

    // Move to first downstream layer
    for &i in &current {
        if let Some(d) = downstream_adj[i] {
            if !visited.contains(d) {
                visited.insert(d);
                next.push(d);
            }
        }
    }
    std::mem::swap(&mut current, &mut next);

    // Process in topological order
    while !current.is_empty() {
        // Process current layer in parallel
        current.par_iter().for_each(|&node| {
            // Accumulate counts from all upstream nodes
            let mut combined_counts = HashMap::new();

            // Add current node's own value
            let val = field_slice[node];
            *combined_counts.entry(val).or_insert(0) += 1;

            // Merge counts from all upstream nodes
            for &upstream_node in &upstream_adj[node] {
                let upstream_counts = node_counts[upstream_node].lock().unwrap();
                for (&cat, &count) in upstream_counts.iter() {
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
            if let Some(d) = downstream_adj[i] {
                if !visited.contains(d) {
                    visited.insert(d);
                    next.push(d);
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

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_topological_labels_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_upstream_mode_rust, m)?)?;
    Ok(())
}
