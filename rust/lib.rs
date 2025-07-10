// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1};
use pyo3::exceptions::PyValueError;
use std::sync::atomic::{AtomicI64, Ordering};
use fixedbitset::FixedBitSet;

use std::collections::HashMap;

#[pyfunction]
fn flow_downstream_par(
    _py: Python,
    topo_groups: &PyAny,           // List of (up, down) np arrays
    field: &PyArray1<f64>,         // Mutable target array
) -> PyResult<()> {
    let field_slice = unsafe { field.as_slice_mut()? };

    for group in topo_groups.iter()? {
        let (up, down, eid): (&PyArray1<usize>, &PyArray1<usize>, &PyArray1<usize>) = group?.extract()?;
        let up_inds = unsafe { up.as_slice()? };
        let down_inds = unsafe { down.as_slice()? };

        // Parallel accumulation: final output is a single merged HashMap
        let contribs: HashMap<usize, f64> = up_inds
            .par_iter()
            .zip(down_inds.par_iter())
            .fold(
                || HashMap::new(),
                |mut local_map, (&u, &d)| {
                    *local_map.entry(d).or_insert(0.0) += field_slice[u];
                    local_map
                },
            )
            .reduce(
                || HashMap::new(),
                |mut acc, map| {
                    for (k, v) in map {
                        *acc.entry(k).or_insert(0.0) += v;
                    }
                    acc
                },
            );

        // Serial application of results to field
        for (d, val) in contribs {
            field_slice[d] += val;
        }
    }

    Ok(())
}

// Wrapper to mark raw pointer as Send + Sync safely
#[derive(Copy, Clone)]
struct FieldPtr(*mut f64);

unsafe impl Send for FieldPtr {}
unsafe impl Sync for FieldPtr {}

#[pyfunction]
fn move_upstream_par(
    _py: Python,
    topo_groups: Vec<(&PyArray1<usize>, &PyArray1<usize>, &PyArray1<usize>)>,
    field: &PyArray1<f64>,
) -> PyResult<()> {
    // Get mutable slice safely
    let field_slice = unsafe { field.as_slice_mut()? };
    let field_ptr = FieldPtr(field_slice.as_mut_ptr());

    for (down, up, _edge) in topo_groups {
        let up_inds = unsafe { up.as_slice()? };
        let down_inds = unsafe { down.as_slice()? };

        // Move the raw pointer inside the closure to satisfy the compiler
        up_inds.par_iter()
            .zip(down_inds.par_iter())
            .for_each(|(&u, &d)| {
                // Copy the pointer here inside the closure, so no lifetime issues
                let ptr_for_closure = field_ptr;
                unsafe {
                    let val = *ptr_for_closure.0.add(u);
                    *ptr_for_closure.0.add(d) += val;
                }
            });
    }

    Ok(())
}

#[pyfunction]
fn move_upstream_par_2(
    _py: Python,
    topo_groups: &PyAny,
    field: &PyArray1<f64>,
) -> PyResult<()> {
    // Get mutable slice safely
    let field_slice = unsafe { field.as_slice_mut()? };
    let field_ptr = FieldPtr(field_slice.as_mut_ptr());

    for group in topo_groups.iter()? {
        let (down, up): (&PyArray1<usize>, &PyArray1<usize>) = group?.extract()?;
        let up_inds = unsafe { up.as_slice()? };
        let down_inds = unsafe { down.as_slice()? };

        // Move the raw pointer inside the closure to satisfy the compiler
        up_inds.par_iter()
            .zip(down_inds.par_iter())
            .for_each(|(&u, &d)| {
                // Copy the pointer here inside the closure, so no lifetime issues
                let ptr_for_closure = field_ptr;
                unsafe {
                    let val = *ptr_for_closure.0.add(u);
                    *ptr_for_closure.0.add(d) += val;
                }
            });
    }

    Ok(())
}

#[pyfunction]
fn move_upstream_seq(
    _py: Python,
    topo_groups: Vec<(&PyArray1<usize>, &PyArray1<usize>, &PyArray1<usize>)>,
    field: &PyArray1<f64>,
) -> PyResult<()> {
    // Get mutable slice safely
    let field_slice = unsafe { field.as_slice_mut()? };

    for (down, up, _edge) in topo_groups {
        let up_inds = unsafe { up.as_slice()? };
        let down_inds = unsafe { down.as_slice()? };

        for (&u, &d) in up_inds.iter().zip(down_inds.iter()) {
            field_slice[d] += field_slice[u];
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

    // let labels = unsafe { labels
    // .as_slice_mut()
    // .expect("Failed to get labels slice")};
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

    // Return as PyArray1
    Ok(PyArray1::from_vec(py, result))

    // Ok(labels.to_pyarray(py))
}

#[pyfunction]
fn get_rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {

    rayon::ThreadPoolBuilder::new().build_global().ok();

    // Optional: pre-run a dummy parallel op to ensure threads are spun up
    rayon::scope(|s| {
        for _ in 0..rayon::current_num_threads() {
            s.spawn(|_| {
                std::hint::black_box(42); // prevent optimization
            });
        }
    });

    m.add_function(wrap_pyfunction!(move_upstream_par, m)?)?;
    m.add_function(wrap_pyfunction!(move_upstream_par_2, m)?)?;
    m.add_function(wrap_pyfunction!(move_upstream_seq, m)?)?;
    m.add_function(wrap_pyfunction!(compute_topological_labels_rust, m)?)?;
    m.add_function(wrap_pyfunction!(flow_downstream_par, m)?)?;
    m.add_function(wrap_pyfunction!(get_rayon_num_threads, m)?)?;
    Ok(())
}
