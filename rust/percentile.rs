// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use dashmap::DashMap;
use ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn calc_perc<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    p: f64,
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
    field: &ArrayView1<f64>,
    result: &mut Vec<f64>,
    p: f64,
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

    did_slice
        .par_iter()
        .zip(uid_slice.par_iter())
        .for_each(|(&did, &uid)| {
            let uid_upstream = {
                // Get uid upstream vector by removing it from the map, so you can move it without cloning
                // If it doesn't exist, fallback to vec![uid]
                upstream_map
                    .remove(&uid)
                    .map(|entry| entry.1)
                    .unwrap_or_else(|| vec![field[uid as usize]])
            };

            // Insert or extend did's upstream vector
            upstream_map
                .entry(did)
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

    let pct_results: Vec<(i64, f64)> = did_slice
        .par_iter()
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
