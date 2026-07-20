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
    if !bifurcates {
        if reverse {
            for group in topo_groups.iter().rev() {
                process_level_fast(metric, group, &map, reverse, result);
            }
        } else {
            for group in topo_groups {
                process_level_fast(metric, group, &map, reverse, result);
            }
        }
        return;
    }

    let mut last_use = HashMap::new();
    if reverse {
        for (level, group) in topo_groups.iter().rev().enumerate() {
            for &s in group.as_array().row(0).iter() {
                last_use.insert(s, level);
            }
        }
    } else {
        for (level, group) in topo_groups.iter().enumerate() {
            for &s in group.as_array().row(1).iter() {
                last_use.insert(s, level);
            }
        }
    }

    if reverse {
        for (level, group) in topo_groups.iter().rev().enumerate() {
            process_level(metric, group, &map, &last_use, level, reverse, result);
        }
    } else {
        for (level, group) in topo_groups.iter().enumerate() {
            process_level(metric, group, &map, &last_use, level, reverse, result);
        }
    }
}

fn process_level_fast<M: Metric>(
    metric: &M,
    topo_group: &PyReadonlyArray2<'_, i64>,
    map: &DashMap<i64, M::Acc>,
    reverse: bool,
    result: &mut [M::Out],
) {
    let arr = topo_group.as_array();
    let did_vec = arr.row(0);
    let uid_vec = arr.row(1);
    let did = did_vec.as_slice().expect("Expected contiguous did slice");
    let uid = uid_vec.as_slice().expect("Expected contiguous uid slice");
    let (source, target): (&[i64], &[i64]) = if reverse { (did, uid) } else { (uid, did) };

    source
        .par_iter()
        .zip(target.par_iter())
        .for_each(|(&s, &t)| {
            let s_acc = if reverse {
                map.get(&s)
                    .map(|entry| entry.clone())
                    .unwrap_or_else(|| metric.singleton(s as usize))
            } else {
                map.remove(&s)
                    .map(|entry| entry.1)
                    .unwrap_or_else(|| metric.singleton(s as usize))
            };

            map.entry(t)
                .and_modify(|acc| metric.merge(acc, &s_acc))
                .or_insert_with(|| {
                    let mut acc = metric.singleton(t as usize);
                    metric.merge(&mut acc, &s_acc);
                    acc
                });
        });

    finalize(metric, target, map, result);
}

fn process_level<M: Metric>(
    metric: &M,
    topo_group: &PyReadonlyArray2<'_, i64>,
    map: &DashMap<i64, M::Acc>,
    last_use: &HashMap<i64, usize>,
    level: usize,
    reverse: bool,
    result: &mut [M::Out],
) {
    let arr = topo_group.as_array();
    let did_vec = arr.row(0);
    let uid_vec = arr.row(1);
    let did = did_vec.as_slice().expect("Expected contiguous did slice");
    let uid = uid_vec.as_slice().expect("Expected contiguous uid slice");

    let (source, target): (&[i64], &[i64]) = if reverse { (did, uid) } else { (uid, did) };

    process_grouped(metric, source, target, map, last_use, level);

    finalize(metric, target, map, result);
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

fn process_grouped<M: Metric>(
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
                map.entry(t)
                    .and_modify(|acc| metric.merge(acc, &s_acc))
                    .or_insert_with(|| {
                        let mut acc = metric.singleton(t as usize);
                        metric.merge(&mut acc, &s_acc);
                        acc
                    });
            }

            (last_use[&s] != level).then_some((s, s_acc))
        })
        .collect();

    for (s, acc) in retained {
        map.insert(s, acc);
    }
}
