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

#[pyfunction]
pub fn calc_weighted_perc<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    p: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    // Single DashMap storing (sorted_values, corresponding_weights) per node
    let upstream_map: DashMap<i64, (Vec<f64>, Vec<f64>)> = DashMap::new();

    let field_array: ArrayView1<f64> = field.as_array();
    let weights_array: ArrayView1<f64> = weights.as_array();

    let mut result: Vec<f64> = field_array.to_vec();

    for group in &topo_groups {
        process_level_and_cleanup(
            group,
            &upstream_map,
            &field_array,
            &weights_array,
            &mut result,
            p,
        );
    }

    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned().into())
}

fn process_level_and_cleanup(
    topo_group: &PyReadonlyArray2<'_, i64>,
    upstream_map: &DashMap<i64, (Vec<f64>, Vec<f64>)>,
    field: &ArrayView1<f64>,
    weights: &ArrayView1<f64>,
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
            // Remove uid's accumulated upstream data (move, not clone)
            let (uid_vals, uid_wts) = upstream_map
                .remove(&uid)
                .map(|entry| entry.1)
                .unwrap_or_else(|| (vec![field[uid as usize]], vec![weights[uid as usize]]));

            // Insert or merge into did's upstream entry
            upstream_map
                .entry(did)
                .and_modify(|(did_vals, did_wts)| {
                    merge_sorted_weighted(did_vals, &uid_vals, did_wts, &uid_wts);
                })
                .or_insert_with(|| {
                    let did_val = field[did as usize];
                    let did_wt = weights[did as usize];
                    let mut vals = uid_vals;
                    let mut wts = uid_wts;
                    let pos = match binary_search_f64(&vals, did_val) {
                        Ok(pos) | Err(pos) => pos,
                    };
                    vals.insert(pos, did_val);
                    wts.insert(pos, did_wt);
                    (vals, wts)
                });
        });

    let pct_results: Vec<(i64, f64)> = did_slice
        .par_iter()
        .map(|&did| {
            let entry = upstream_map.get(&did).unwrap();
            let (ref vals, ref wts) = *entry;
            let pct = weighted_percentile(vals, wts, p);
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

/// Weighted percentile with linear interpolation.
///
/// Each sorted value x_i is assigned quantile position:
///     q_i = C_i^- / D
/// where C_i^- = Σ_{j<i} w_j (exclusive cumulative weight) and D = W - w_{n-1}.
/// When all weights are equal this reduces to q_i = i/(n-1), matching numpy.
fn weighted_percentile(sorted_values: &[f64], weights: &[f64], p: f64) -> f64 {
    let n = sorted_values.len();
    if n == 1 {
        return sorted_values[0];
    }
    if p <= 0.0 {
        return sorted_values[0];
    }
    if p >= 1.0 {
        return sorted_values[n - 1];
    }

    // D = W - w_{n-1} so that q_{n-1} = C_{n-1}^- / D = 1
    let total_w: f64 = weights.iter().sum();
    let d = total_w - weights[n - 1];

    if d <= 0.0 {
        return sorted_values[0];
    }

    // target in cumulative-weight-before space: C_k^- ≤ target < C_{k+1}^-
    let target = p * d;

    let mut cum_before: f64 = 0.0; // C_0^- = 0
    for i in 0..n - 1 {
        let next_cum = cum_before + weights[i]; // C_{i+1}^-
        if target <= next_cum {
            // Interpolation fraction within [q_i, q_{i+1}]
            let frac = if weights[i] > 0.0 {
                (target - cum_before) / weights[i]
            } else {
                0.0
            };
            return sorted_values[i] + frac * (sorted_values[i + 1] - sorted_values[i]);
        }
        cum_before = next_cum;
    }

    sorted_values[n - 1]
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

/// Merge two sorted-by-value arrays, keeping ALL entries (duplicates preserved).
/// Each entry retains its individual weight.
fn merge_sorted_weighted(
    a_vals: &mut Vec<f64>,
    b_vals: &[f64],
    a_wts: &mut Vec<f64>,
    b_wts: &[f64],
) {
    let mut i = 0;
    let mut j = 0;
    let mut result_vals = Vec::with_capacity(a_vals.len() + b_vals.len());
    let mut result_wts = Vec::with_capacity(a_wts.len() + b_wts.len());

    while i < a_vals.len() && j < b_vals.len() {
        if a_vals[i] <= b_vals[j] {
            result_vals.push(a_vals[i]);
            result_wts.push(a_wts[i]);
            i += 1;
        } else {
            result_vals.push(b_vals[j]);
            result_wts.push(b_wts[j]);
            j += 1;
        }
    }

    result_vals.extend_from_slice(&a_vals[i..]);
    result_wts.extend_from_slice(&a_wts[i..]);
    result_vals.extend_from_slice(&b_vals[j..]);
    result_wts.extend_from_slice(&b_wts[j..]);

    *a_vals = result_vals;
    *a_wts = result_wts;
}
