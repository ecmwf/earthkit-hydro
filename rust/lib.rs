// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use pyo3::prelude::*;
use rayon::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArrayDyn, PyArray2, PyArray1};
use ndarray::{Axis, Zip};
use pyo3::exceptions::PyValueError;
use std::sync::atomic::{AtomicI64, Ordering};
use fixedbitset::FixedBitSet;

#[pyfunction]
fn _flow_rust<'py>(
    py: Python<'py>,
    field: &'py PyArrayDyn<f64>,  // shape (..., F)
    groups: &PyList, // shape (3, N)
    invert_graph: bool,
    node_modifier_use_upstream: bool,
    node_multiplicative_weight: Option<&'py PyArrayDyn<f64>>, // shape (..., G)
    edge_multiplicative_weight: Option<&'py PyArrayDyn<f64>>, // shape (..., E)
    node_additive_weight: Option<&'py PyArrayDyn<f64>>,       // shape (..., G)
    edge_additive_weight: Option<&'py PyArrayDyn<f64>>,       // shape (..., E)
) -> PyResult<()> {
    let mut field = unsafe { field.as_array_mut() };
    let ndim = field.ndim();
    let last_axis = ndim - 1;

    for group_any in groups.iter() {
        let group = group_any.downcast::<PyArray2<usize>>()?;
        let group = unsafe { group.as_array() }; // shape (3, K)
        if group.shape()[0] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("Group must have shape (3, K)"));
        }

        let k = group.shape()[1];

        // Extract slices from group
        let dids = group.row(0);
        let uids = group.row(1);
        let eids = group.row(2);

        // Determine srcs and dsts according to invert_graph
        let (srcs, dsts) = if invert_graph { (dids, uids) } else { (uids, dids) };
        // Determine modifier_group based on flag
        let modifier_group = if node_modifier_use_upstream { srcs } else { dsts };

        // Gather modifier fields: field[..., srcs]
        // We'll create an Array with shape: field.shape[0..-1] + (k,)
        let mut modifier_shape = field.shape().to_vec();
        modifier_shape[last_axis] = k;
        let mut modifier = ndarray::ArrayD::<f64>::zeros(modifier_shape);

        // Gather slices for each src index
        for (i, &src) in srcs.iter().enumerate() {
            let src_slice = field.index_axis(Axis(last_axis), src);
            let mut mod_slice = modifier.index_axis_mut(Axis(last_axis), i);
            mod_slice.assign(&src_slice);
        }

        // Apply node multiplicative weights
        if let Some(node_mul) = node_multiplicative_weight {
            let node_mul = unsafe { node_mul.as_array() };
            for (i, &mod_gid) in modifier_group.iter().enumerate() {
                let weight = node_mul.index_axis(Axis(node_mul.ndim() - 1), mod_gid);
                let weight_b = weight.broadcast(modifier.index_axis(Axis(last_axis), i).shape())
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Node multiplicative weight not broadcastable"))?;
                let mut slice = modifier.index_axis_mut(Axis(last_axis), i);
                slice *= &weight_b;
            }
        }

        // Apply edge multiplicative weights
        if let Some(edge_mul) = edge_multiplicative_weight {
            let edge_mul = unsafe { edge_mul.as_array() };
            for (i, &eid) in eids.iter().enumerate() {
                let weight = edge_mul.index_axis(Axis(edge_mul.ndim() - 1), eid);
                let weight_b = weight.broadcast(modifier.index_axis(Axis(last_axis), i).shape())
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Edge multiplicative weight not broadcastable"))?;
                let mut slice = modifier.index_axis_mut(Axis(last_axis), i);
                slice *= &weight_b;
            }
        }

        // Apply node additive weights
        if let Some(node_add) = node_additive_weight {
            let node_add = unsafe { node_add.as_array() };
            for (i, &mod_gid) in modifier_group.iter().enumerate() {
                let add = node_add.index_axis(Axis(node_add.ndim() - 1), mod_gid);
                let add_b = add.broadcast(modifier.index_axis(Axis(last_axis), i).shape())
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Node additive weight not broadcastable"))?;
                let mut slice = modifier.index_axis_mut(Axis(last_axis), i);
                slice += &add_b;
            }
        }

        // Apply edge additive weights
        if let Some(edge_add) = edge_additive_weight {
            let edge_add = unsafe { edge_add.as_array() };
            for (i, &eid) in eids.iter().enumerate() {
                let add = edge_add.index_axis(Axis(edge_add.ndim() - 1), eid);
                let add_b = add.broadcast(modifier.index_axis(Axis(last_axis), i).shape())
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Edge additive weight not broadcastable"))?;
                let mut slice = modifier.index_axis_mut(Axis(last_axis), i);
                slice += &add_b;
            }
        }

        // Scatter-add modifier[..., i] into field[..., dsts[i]]
        for (i, &dst) in dsts.iter().enumerate() {
            let mut dest = field.index_axis_mut(Axis(last_axis), dst);
            let mod_slice = modifier.index_axis(Axis(last_axis), i);
            Zip::from(&mut dest).and(&mod_slice).for_each(|a, &b| *a += b);
        }
    }

    Ok(())
}

#[pyfunction]
fn compute_topological_labels_rust<'py>(
    py: Python<'py>,
    sources: &PyArray1<usize>,
    sinks: &PyArray1<usize>,
    downstream_nodes: &PyArray1<usize>,
    n_nodes: usize,
) -> PyResult<&'py PyArray1<i64>> {

    let labels: Vec<AtomicI64> = (0..n_nodes)
        .map(|_| AtomicI64::new(0))
        .collect();

    let downstream = unsafe { downstream_nodes
        .as_slice()
        .expect("Failed to get downstream_nodes slice")};

    let mut current = unsafe { sources
        .as_slice()
        .expect("Failed to get sources slice")
        .to_vec() };

    let sinks = unsafe { sinks
            .as_slice()
            .expect("Failed to get sinks slice")
            .to_vec() };

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

    Ok(PyArray1::from_vec(py, result))
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_flow_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_topological_labels_rust, m)?)?;
    Ok(())
}
