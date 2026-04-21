// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use dashmap::DashMap;
use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

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
