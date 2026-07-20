// (C) Copyright 2025- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

use dashmap::DashMap;
use numpy::PyReadonlyArray2;
use rayon::prelude::*;
use std::collections::HashMap;

/// A metric that accumulates per-node state across the river network.
///
/// Implementors only describe *what* to accumulate (`Acc`), how to seed it for a
/// single node (`singleton`), how to combine two accumulations (`merge`) and how
/// to turn an accumulation into a result value (`finalize`). The traversal itself
/// (topological ordering, parallelism, memory cleanup) is handled by [`run`].
pub trait Metric: Sync {
    type Acc: Clone + Send + Sync;
    type Out: Copy + Send;

    fn singleton(&self, node: usize) -> Self::Acc;
    fn merge(&self, dst: &mut Self::Acc, src: &Self::Acc);
    fn finalize(&self, acc: &Self::Acc) -> Self::Out;
}

/// Run a metric over the topological groups.
///
/// `reverse == false` accumulates upstream (data flows uid -> did) and
/// `reverse == true` accumulates downstream (data flows did -> uid). Accumulators
/// are moved on their final use. Levels with repeated sources are grouped by
/// source so one accumulator can feed every outgoing edge without being cloned.
///
/// Accumulation is edge-based: if bifurcating paths reconverge, a shared node is
/// represented once per path. Exact unique-node semantics would require carrying
/// reachability sets, with potentially quadratic time and memory costs.
pub fn run<M: Metric>(
    metric: &M,
    topo_groups: &[PyReadonlyArray2<'_, i64>],
    reverse: bool,
    bifurcates: bool,
    result: &mut [M::Out],
) {
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
        finalize(metric, target, &map, result);
    }
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

fn finalize<M: Metric>(
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
