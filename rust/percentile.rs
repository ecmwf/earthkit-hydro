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

/// Symmetric "edge length" between two neighbouring weights.
///
/// This is the design knob that controls how strongly a node's weight stretches
/// the percentile space of its adjacent intervals. The arithmetic mean makes the
/// total axis length equal the summed weight (each interior node contributes its
/// own weight, endpoints half), i.e. percentile width == probability mass.
#[inline]
fn edge_length(a: f64, b: f64) -> f64 {
    0.5 * (a + b)
}

/// Weighted percentile that redistributes percentile space by weight.
///
/// The sorted values are placed at knots `P_i = (Σ_{k<i} L_k) / Σ L`, where the
/// interval width `L_i = edge_length(w_i, w_{i+1})`. Heavier nodes widen their
/// adjacent intervals, so every weight influences the position of all later
/// knots (a value's weight affects both intervals it borders). Within the
/// bracketing interval the two values are interpolated linearly.
///
/// Equal weights make every `L_i` equal, giving `P_i = i/(n-1)` and reproducing
/// NumPy's `linear` (type-7) percentile exactly. The minimum is returned at
/// `p = 0` and the maximum at `p = 1`.
///
/// Note: for `n = 2` there is a single interval, so `P = [0, 1]` regardless of
/// weights and the result is the midpoint at `p = 0.5` — weights only redistribute
/// space when there are at least three values.
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

    let total: f64 = (0..n - 1)
        .map(|i| edge_length(weights[i], weights[i + 1]))
        .sum();
    if total <= 0.0 {
        // All weights zero: fall back to an unweighted (type-7) position.
        let rank = p * (n as f64 - 1.0);
        let i = rank.floor() as usize;
        let t = rank - i as f64;
        return sorted_values[i] + t * (sorted_values[i + 1] - sorted_values[i]);
    }

    let target = p * total;
    let mut cum = 0.0;
    for i in 0..n - 1 {
        let len = edge_length(weights[i], weights[i + 1]);
        if target <= cum + len {
            let t = if len > 0.0 { (target - cum) / len } else { 0.0 };
            return sorted_values[i] + t * (sorted_values[i + 1] - sorted_values[i]);
        }
        cum += len;
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
