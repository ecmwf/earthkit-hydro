// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::metric::{run, Metric};

struct Percentile<'a> {
    field: ArrayView1<'a, f64>,
    p: f64,
}

impl Metric for Percentile<'_> {
    type Acc = Vec<f64>;
    type Out = f64;

    fn singleton(&self, node: usize) -> Vec<f64> {
        vec![self.field[node]]
    }

    fn merge(&self, dst: &mut Vec<f64>, src: &Vec<f64>) {
        merge_sorted(dst, src);
    }

    fn finalize(&self, acc: &Vec<f64>) -> f64 {
        percentile(acc, self.p)
    }
}

struct WeightedPercentile<'a> {
    field: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    p: f64,
}

impl Metric for WeightedPercentile<'_> {
    type Acc = (Vec<f64>, Vec<f64>);
    type Out = f64;

    fn singleton(&self, node: usize) -> (Vec<f64>, Vec<f64>) {
        (vec![self.field[node]], vec![self.weights[node]])
    }

    fn merge(&self, dst: &mut (Vec<f64>, Vec<f64>), src: &(Vec<f64>, Vec<f64>)) {
        merge_sorted_weighted(&mut dst.0, &src.0, &mut dst.1, &src.1);
    }

    fn finalize(&self, acc: &(Vec<f64>, Vec<f64>)) -> f64 {
        weighted_percentile(&acc.0, &acc.1, self.p)
    }
}

#[pyfunction]
pub fn calc_perc<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    p: f64,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let field_array = field.as_array();
    let mut result: Vec<f64> = field_array.to_vec();
    let metric = Percentile {
        field: field_array,
        p,
    };
    run(&metric, &topo_groups, false, bifurcates, &mut result);
    Ok(PyArray1::from_vec(py, result).to_owned().into())
}

#[pyfunction]
pub fn calc_perc_downstream<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    p: f64,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let field_array = field.as_array();
    let mut result: Vec<f64> = field_array.to_vec();
    let metric = Percentile {
        field: field_array,
        p,
    };
    run(&metric, &topo_groups, true, bifurcates, &mut result);
    Ok(PyArray1::from_vec(py, result).to_owned().into())
}

#[pyfunction]
pub fn calc_weighted_perc<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    p: f64,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let field_array = field.as_array();
    let weights_array = weights.as_array();
    let mut result: Vec<f64> = field_array.to_vec();
    let metric = WeightedPercentile {
        field: field_array,
        weights: weights_array,
        p,
    };
    run(&metric, &topo_groups, false, bifurcates, &mut result);
    Ok(PyArray1::from_vec(py, result).to_owned().into())
}

#[pyfunction]
pub fn calc_weighted_perc_downstream<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    p: f64,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let field_array = field.as_array();
    let weights_array = weights.as_array();
    let mut result: Vec<f64> = field_array.to_vec();
    let metric = WeightedPercentile {
        field: field_array,
        weights: weights_array,
        p,
    };
    run(&metric, &topo_groups, true, bifurcates, &mut result);
    Ok(PyArray1::from_vec(py, result).to_owned().into())
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

fn merge_sorted(a: &mut Vec<f64>, b: &[f64]) {
    let mut i = 0;
    let mut j = 0;
    let mut result = Vec::with_capacity(a.len() + b.len());

    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            result.push(a[i]);
            i += 1;
        } else {
            result.push(b[j]);
            j += 1;
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    *a = result;
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
