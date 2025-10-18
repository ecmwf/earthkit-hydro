// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use std::sync::atomic::{AtomicI64, Ordering};
use fixedbitset::FixedBitSet;

// use pyo3::prelude::*;
// use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use dashmap::DashMap;
// use rayon::prelude::*;
use std::collections::HashMap;
use ndarray::ArrayView1;

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

fn insert_sorted_unique(dest: &mut Vec<i64>, src: &[i64]) {
    let mut i = 0; // index for dest
    let mut j = 0; // index for src

    let mut new_vec = Vec::with_capacity(dest.len() + src.len());

    while i < dest.len() && j < src.len() {
        if dest[i] < src[j] {
            new_vec.push(dest[i]);
            i += 1;
        } else if dest[i] > src[j] {
            new_vec.push(src[j]);
            j += 1;
        } else {
            // equal, push once and advance both
            new_vec.push(dest[i]);
            i += 1;
            j += 1;
        }
    }

    if i < dest.len() {
        new_vec.extend_from_slice(&dest[i..]);
    }
    if j < src.len() {
        new_vec.extend_from_slice(&src[j..]);
    }

    *dest = new_vec;
}

#[pyfunction]
fn test_rust<'py>(
    _py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    _field: PyReadonlyArray1<'py, f64>,
) -> PyResult<HashMap<i64, Vec<i64>>> {

    let mut upstream_map: HashMap<i64, Vec<i64>> = HashMap::new();

    for group in topo_groups.iter() {
        let arr = group.as_array();
        let did_vec = arr.row(0);
        let uid_vec = arr.row(1);
        // let eid_vec = arr.row(2);

        for (&did, &uid) in did_vec.iter().zip(uid_vec.iter()) {
            // Get uid upstream additions (or fallback to [uid])
            let uid_upstream_additions = upstream_map.get(&uid).cloned().unwrap_or_else(|| vec![uid]);

            // Insert did upstream if missing and extend it with uid upstream additions
            upstream_map.entry(did).or_insert_with(|| vec![did])
                .extend(uid_upstream_additions.into_iter());

            // Remove uid upstream set now that we merged it
            upstream_map.remove(&uid);
        }

        // for (&did, &uid) in did_vec.iter().zip(uid_vec.iter()) {
        //     // Make sure uid has an upstream set including itself
        //     upstream_map.entry(uid).or_insert_with(|| vec![uid]);

        //     // Immutable borrow scope: get uid's upstream (which includes uid itself now)
        //     let uid_upstream_additions = {
        //         if let Some(uid_upstream) = upstream_map.get(&uid) {
        //             // Clone additions to avoid borrow checker issues
        //             let mut additions = uid_upstream.clone(); // includes uid itself
        //             additions.sort_unstable();
        //             // additions.dedup();
        //             additions
        //         } else {
        //             vec![uid] // fallback, should never happen after entry above
        //         }
        //     }; // immutable borrow dropped here

        //     // Ensure did also has its upstream initialized (with itself)
        //     upstream_map.entry(did).or_insert_with(|| vec![did]);

        //     // Now mutable borrow to did_upstream safe
        //     let did_upstream = upstream_map.entry(did).or_default();
        //     insert_sorted_unique(did_upstream, &uid_upstream_additions);
        // }

}
    Ok(upstream_map)
}

// // Utility: insert elements while keeping dest sorted & unique
// fn insert_sorted_unique(dest: &mut Vec<i64>, additions: &[i64]) {
//     dest.extend_from_slice(additions);
//     dest.sort_unstable();
//     dest.dedup();
// }

// #[pyfunction]
// fn build_upstream<'py>(
//     _py: Python<'py>,
//     topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
// ) -> PyResult<()> {
//     use rayon::prelude::*;
//     use dashmap::DashMap;

//     // Step 1: Convert all Python arrays into a flat Vec of edges: (did, uid)
//     let all_edges: Vec<(i64, i64)> = topo_groups
//         .iter()
//         .flat_map(|group| {
//             let array = group.as_array(); // shape: (3, N)
//             let n_edges = array.shape()[1];
//             let dids = array.row(0);
//             let uids = array.row(1);
//             (0..n_edges).map(move |i| (dids[i], uids[i])).collect::<Vec<_>>()
//         })
//         .collect();

//     // Step 2: Collect all node IDs
//     let all_nodes: Vec<i64> = all_edges
//         .iter()
//         .flat_map(|(did, uid)| vec![*did, *uid])
//         .collect();

//     let upstream_map = DashMap::<i64, Vec<i64>>::new();

//     // Step 3: Pre-fill with self-refs
//     all_nodes.par_iter().for_each(|&node| {
//         upstream_map.entry(node).or_insert_with(|| vec![node]);
//     });

//     // Step 4: Process all edges in parallel
//     all_edges.par_iter().for_each(|&(did, uid)| {
//         let uid_upstream = upstream_map
//             .get(&uid)
//             .map(|v| v.clone())
//             .unwrap_or_else(|| vec![uid]);

//         let mut did_upstream = upstream_map.entry(did).or_insert_with(|| vec![did]);
//         insert_sorted_unique(&mut did_upstream, &uid_upstream);

//         upstream_map.remove(&uid);
//     });

//     // Done!
//     Ok(())
// }

#[pyfunction]
fn process_nodes<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    p: f64
) -> PyResult<Py<PyArray1<f64>>> {

    let upstream_map: DashMap<i64, Vec<f64>> = DashMap::new();

    let field_array: ArrayView1<f64> = field.as_array();

    let mut result: Vec<f64> = field_array.to_vec(); //vec![0.0; field_array.len()];

    for group in &topo_groups {
        process_level_and_cleanup(group, &upstream_map, &field_array, &mut result, p);
    }

    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned().into())
}

fn process_level_and_cleanup(
    topo_group: &PyReadonlyArray2<'_, i64>,
    upstream_map: &DashMap<i64, Vec<f64>>,
    field : &ArrayView1<f64>,
    result : &mut Vec<f64>,
    p : f64
) {
    let arr = topo_group.as_array();
    let did_vec = arr.row(0);
    let uid_vec = arr.row(1);

    let did_slice = did_vec.as_slice().expect("Expected contiguous did_vec slice");
    let uid_slice = uid_vec.as_slice().expect("Expected contiguous uid_vec slice");

    did_slice.par_iter().zip(uid_slice.par_iter()).for_each(|(&did, &uid)| {
        let uid_upstream = {
            // Get uid upstream vector by removing it from the map, so you can move it without cloning
            // If it doesn't exist, fallback to vec![uid]
            upstream_map.remove(&uid).map(|entry| entry.1).unwrap_or_else(|| vec![field[uid as usize]])
        };

        // Insert or extend did's upstream vector
        upstream_map.entry(did)
            .and_modify(|did_upstream| {
                merge_sorted_unique_f64(did_upstream, &uid_upstream);
            })
            .or_insert_with(|| {
                    let mut v = uid_upstream;
                    match binary_search_f64(&v, field[did as usize]) {
                        Ok(_) => {}
                        Err(pos) => v.insert(pos, field[did as usize]),
                    }

                    v
            });
        });

    // for entry in upstream_map.iter() {
    //     let key = *entry.key() as usize;
    //     let values = entry.value();

    //     let pct = percentile(values.as_slice(), 50.0);  // Or any metric you want

    //     result[key] = pct;  // Safe, no concurrency worries
    // }

    let pct_results: Vec<(i64, f64)> = did_slice.par_iter()
    .map(|&did| {
        let values = upstream_map.get(&did).unwrap();
        let pct = percentile(values.as_slice(), p);
        (did, pct)
    })
    .collect();

    for (did, pct) in pct_results {
        let idx = did as usize;
        if idx < result.len() {
            result[idx] = pct;
        }
    }

}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {

    let n = sorted_values.len();
    let rank = percentile * (n as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = rank - lower as f64;
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    }
}

fn binary_search_f64(slice: &[f64], target: f64) -> Result<usize, usize> {
    let mut size = slice.len();
    if size == 0 {
        return Err(0);
    }
    let mut base = 0usize;

    while size > 0 {
        let half = size / 2;
        let mid = base + half;

        match slice[mid].partial_cmp(&target).unwrap() {
            std::cmp::Ordering::Less => {
                base = mid + 1;
                size -= half + 1;
            }
            std::cmp::Ordering::Equal => return Ok(mid),
            std::cmp::Ordering::Greater => size = half,
        }
    }
    Err(base)
}



fn merge_sorted_unique_f64(a: &mut Vec<f64>, b: &[f64]) {
    let mut i = 0;
    let mut j = 0;
    let mut result = Vec::with_capacity(a.len() + b.len());

    while i < a.len() && j < b.len() {
        match a[i].partial_cmp(&b[j]).unwrap() {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    *a = result;
}


fn merge_sorted_unique(a: &mut Vec<i64>, b: &[i64]) {
    let mut i = 0;
    let mut j = 0;
    let mut result = Vec::with_capacity(a.len() + b.len());

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    *a = result;
}


#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_topological_labels_rust, m)?)?;
    m.add_function(wrap_pyfunction!(test_rust, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes, m)?)?;
    Ok(())
}
