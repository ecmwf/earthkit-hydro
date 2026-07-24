// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use dashmap::DashMap;
use numpy::{Element, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// A metric that accumulates per-node state across the river network.
///
/// An implementor only describes *what* to accumulate. The traversal itself
/// (topological ordering, parallelism, memory cleanup) lives in [`Metric::compute`]:
///
/// * `initial`   – each node's result before any accumulation (its own value).
/// * `singleton` – seed an accumulator from a single node.
/// * `merge`     – fold one accumulator into another.
/// * `finalize`  – turn a finished accumulator into a result value.
///
/// `reverse == false` accumulates upstream (data flows uid -> did); `reverse ==
/// true` accumulates downstream (data flows did -> uid).
pub trait Metric: Sync {
    type Acc: Clone + Send + Sync;
    type Out: Element + Send;

    fn initial(&self) -> Vec<Self::Out>;
    fn singleton(&self, node: usize) -> Self::Acc;
    fn merge(&self, dst: &mut Self::Acc, src: &Self::Acc);
    fn finalize(&self, acc: &Self::Acc) -> Self::Out;

    /// Accumulate over the whole network and return the result as a NumPy array.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        topo_groups: &[PyReadonlyArray2<'py, i64>],
        reverse: bool,
        bifurcates: bool,
    ) -> Py<PyArray1<Self::Out>>
    where
        Self: Sized,
    {
        let result = run(self, topo_groups, reverse, bifurcates);
        PyArray1::from_vec(py, result).to_owned().into()
    }
}

/// Accumulate a metric over the topological groups and return the per-node result.
///
/// Accumulators are moved on their final use. When a network bifurcates a source
/// feeds several edges (and levels), so edges are grouped by source to reuse one
/// accumulator across all its targets instead of cloning it per edge.
///
/// Accumulation is edge-based: if bifurcating paths reconverge, a shared node is
/// represented once per path. Exact unique-node semantics would require carrying
/// reachability sets, with potentially quadratic time and memory costs.
fn run<M: Metric>(
    metric: &M,
    topo_groups: &[PyReadonlyArray2<'_, i64>],
    reverse: bool,
    bifurcates: bool,
) -> Vec<M::Out> {
    let mut result = metric.initial();
    let map: DashMap<i64, M::Acc> = DashMap::new();

    // Upstream traversal walks levels sources -> sinks; downstream reverses that.
    let order: Vec<usize> = if reverse {
        (0..topo_groups.len()).rev().collect()
    } else {
        (0..topo_groups.len()).collect()
    };

    // Bifurcating networks reuse a source across several edges and levels, so we
    // record the last level each source feeds and only drop it once it is done.
    let last_use = bifurcates.then(|| last_use_by_source(topo_groups, &order, reverse));

    for (level, &g) in order.iter().enumerate() {
        let arr = topo_groups[g].as_array();
        let did_row = arr.row(0);
        let uid_row = arr.row(1);
        let did = did_row.as_slice().expect("Expected contiguous did slice");
        let uid = uid_row.as_slice().expect("Expected contiguous uid slice");
        let (source, target): (&[i64], &[i64]) = if reverse { (did, uid) } else { (uid, did) };

        match &last_use {
            Some(last_use) => accumulate_grouped(metric, source, target, &map, last_use, level),
            None => accumulate_simple(metric, source, target, &map, reverse),
        }
        write_results(metric, target, &map, &mut result);
    }

    result
}

/// Record, for every source node, the highest traversal level at which it is used.
fn last_use_by_source(
    topo_groups: &[PyReadonlyArray2<'_, i64>],
    order: &[usize],
    reverse: bool,
) -> HashMap<i64, usize> {
    let mut last_use = HashMap::new();
    for (level, &g) in order.iter().enumerate() {
        let arr = topo_groups[g].as_array();
        for &s in arr.row(if reverse { 0 } else { 1 }).iter() {
            last_use.insert(s, level);
        }
    }
    last_use
}

/// Accumulate a level whose sources each feed a single edge (non-bifurcating).
fn accumulate_simple<M: Metric>(
    metric: &M,
    source: &[i64],
    target: &[i64],
    map: &DashMap<i64, M::Acc>,
    reverse: bool,
) {
    source
        .par_iter()
        .zip(target.par_iter())
        .for_each(|(&s, &t)| {
            // Upstream sources are used once and can be moved out; downstream
            // sources may feed several targets so they are cloned instead.
            let s_acc = if reverse {
                map.get(&s)
                    .map(|entry| entry.clone())
                    .unwrap_or_else(|| metric.singleton(s as usize))
            } else {
                map.remove(&s)
                    .map(|entry| entry.1)
                    .unwrap_or_else(|| metric.singleton(s as usize))
            };
            merge_into(metric, map, t, &s_acc);
        });
}

/// Finalize this level's target accumulators and write them into `result`.
fn write_results<M: Metric>(
    metric: &M,
    target: &[i64],
    map: &DashMap<i64, M::Acc>,
    result: &mut [M::Out],
) {
    let finalized: Vec<(i64, M::Out)> = target
        .par_iter()
        .map(|&t| {
            let acc = map.get(&t).unwrap();
            (t, metric.finalize(&acc))
        })
        .collect();

    for (t, value) in finalized {
        let idx = t as usize;
        if idx < result.len() {
            result[idx] = value;
        }
    }
}

/// Accumulate a level where sources may feed several edges. Edges are grouped by
/// source so each accumulator is moved once and shared across its targets rather
/// than cloned per edge; a source is only removed on the level of its last use.
fn accumulate_grouped<M: Metric>(
    metric: &M,
    source: &[i64],
    target: &[i64],
    map: &DashMap<i64, M::Acc>,
    last_use: &HashMap<i64, usize>,
    level: usize,
) {
    let mut edges_by_source: HashMap<i64, Vec<i64>> = HashMap::new();
    for (&s, &t) in source.iter().zip(target) {
        edges_by_source.entry(s).or_default().push(t);
    }

    let retained: Vec<(i64, M::Acc)> = edges_by_source
        .into_par_iter()
        .filter_map(|(s, targets)| {
            let s_acc = map
                .remove(&s)
                .map(|entry| entry.1)
                .unwrap_or_else(|| metric.singleton(s as usize));

            for t in targets {
                merge_into(metric, map, t, &s_acc);
            }

            (last_use[&s] != level).then_some((s, s_acc))
        })
        .collect();

    for (s, acc) in retained {
        map.insert(s, acc);
    }
}

#[inline]
fn merge_into<M: Metric>(metric: &M, map: &DashMap<i64, M::Acc>, target: i64, source: &M::Acc) {
    map.entry(target)
        .and_modify(|acc| metric.merge(acc, source))
        .or_insert_with(|| {
            let mut acc = metric.singleton(target as usize);
            metric.merge(&mut acc, source);
            acc
        });
}
