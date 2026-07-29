// SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::metric::Metric;

struct Mode<'a> {
    field: ArrayView1<'a, i64>,
}

impl Metric for Mode<'_> {
    type Acc = HashMap<i64, i64>;
    type Out = i64;

    fn initial(&self) -> Vec<i64> {
        self.field.to_vec()
    }

    fn singleton(&self, node: usize) -> HashMap<i64, i64> {
        let mut counts = HashMap::new();
        counts.insert(self.field[node], 1);
        counts
    }

    fn merge(&self, dst: &mut HashMap<i64, i64>, src: &HashMap<i64, i64>) {
        for (&cat, &count) in src.iter() {
            *dst.entry(cat).or_insert(0) += count;
        }
    }

    fn finalize(&self, counts: &HashMap<i64, i64>) -> i64 {
        counts
            .iter()
            .max_by_key(|(&cat, &count)| (count, -cat))
            .map(|(&cat, _)| cat)
            .unwrap_or(0)
    }
}

#[pyfunction]
pub fn calc_mode<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, i64>,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<i64>>> {
    let metric = Mode {
        field: field.as_array(),
    };
    Ok(metric.compute(py, &topo_groups, false, bifurcates))
}

#[pyfunction]
pub fn calc_mode_downstream<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, i64>,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<i64>>> {
    let metric = Mode {
        field: field.as_array(),
    };
    Ok(metric.compute(py, &topo_groups, true, bifurcates))
}
