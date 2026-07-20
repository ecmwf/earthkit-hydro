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
use std::collections::HashMap;

use crate::metric::{run, Metric};

struct Mode<'a> {
    field: ArrayView1<'a, i64>,
}

impl Metric for Mode<'_> {
    type Acc = HashMap<i64, i64>;
    type Out = i64;

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
    let field_array = field.as_array();
    let mut result: Vec<i64> = field_array.to_vec();
    let metric = Mode { field: field_array };
    run(&metric, &topo_groups, false, bifurcates, &mut result);
    Ok(PyArray1::from_vec(py, result).to_owned().into())
}

#[pyfunction]
pub fn calc_mode_downstream<'py>(
    py: Python<'py>,
    topo_groups: Vec<PyReadonlyArray2<'py, i64>>,
    field: PyReadonlyArray1<'py, i64>,
    bifurcates: bool,
) -> PyResult<Py<PyArray1<i64>>> {
    let field_array = field.as_array();
    let mut result: Vec<i64> = field_array.to_vec();
    let metric = Mode { field: field_array };
    run(&metric, &topo_groups, true, bifurcates, &mut result);
    Ok(PyArray1::from_vec(py, result).to_owned().into())
}
