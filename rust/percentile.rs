// SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::metric::Metric;

struct Percentile<'a> {
    field: ArrayView1<'a, f64>,
    p: f64,
}

impl Metric for Percentile<'_> {
    type Acc = Vec<f64>;
    type Out = f64;

    fn initial(&self) -> Vec<f64> {
        self.field.to_vec()
    }

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

    fn initial(&self) -> Vec<f64> {
        self.field.to_vec()
    }

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
    let metric = Percentile {
        field: field.as_array(),
        p,
    };
    Ok(metric.compute(py, &topo_groups, false, bifurcates))
}

#[pyfunction]
pub fn calc_perc_downstream<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, f64>,
    p: f64,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let metric = Percentile {
        field: field.as_array(),
        p,
    };
    Ok(metric.compute(py, &topo_groups, true, bifurcates))
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
    let metric = WeightedPercentile {
        field: field.as_array(),
        weights: weights.as_array(),
        p,
    };
    Ok(metric.compute(py, &topo_groups, false, bifurcates))
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
    let metric = WeightedPercentile {
        field: field.as_array(),
        weights: weights.as_array(),
        p,
    };
    Ok(metric.compute(py, &topo_groups, true, bifurcates))
}

/// Unweighted percentile using the inverted-CDF (step) method.
///
/// Matches NumPy's ``method="inverted_cdf"``: each of the `n` sorted values holds
/// probability mass `1/n`, and the p-th percentile is the smallest value whose
/// inclusive cumulative probability reaches `p`, i.e. `x_i` for the smallest `i`
/// with `(i + 1) >= p * n`. That rank is computed directly as
/// `ceil(p * n) - 1` (clamped to a valid index), avoiding a scan of the
/// accumulator. The result is always one of the input values (no interpolation);
/// `p = 0` gives the minimum and `p = 1` the maximum.
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    let n = sorted_values.len();
    // Smallest count `m = i + 1` with `m >= p * n`, i.e. `ceil(p * n)`, at least 1.
    let count = (p * n as f64).ceil().max(1.0);
    let index = (count as usize - 1).min(n - 1);
    sorted_values[index]
}

/// Weighted percentile using the inverted-CDF (step) method.
///
/// Matches NumPy's ``method="inverted_cdf"`` with ``weights``: each sorted value
/// carries probability mass proportional to its weight, and the p-th percentile
/// is the smallest value whose inclusive cumulative weight reaches `p * W` (where
/// `W` is the total weight). The result is always one of the input values.
///
/// Uniform weights reduce this exactly to the unweighted [`percentile`] above, so
/// the weighted and unweighted definitions are consistent. Weights genuinely shift
/// the result, including for two values.
fn weighted_percentile(sorted_values: &[f64], weights: &[f64], p: f64) -> f64 {
    let total: f64 = weights.iter().sum();
    let target = p * total;
    let mut cumulative = 0.0;
    for (&value, &weight) in sorted_values.iter().zip(weights) {
        cumulative += weight;
        if cumulative >= target {
            return value;
        }
    }
    sorted_values[sorted_values.len() - 1]
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

/// Merge two value/weight arrays that are each already sorted by value, keeping
/// every entry (duplicates included) with its own weight.
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
