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
/// `reverse == false` accumulates upstream (data flows uid -> did, source
/// accumulations are consumed as they are no longer needed). `reverse == true`
/// accumulates downstream (data flows did -> uid, source accumulations are shared
/// with multiple targets so they are cloned). In both cases a source's entry is
/// only stored while nodes above/below it are still being processed, keeping the
/// live set as small as possible.
pub fn run<M: Metric>(
    metric: &M,
    topo_groups: &[PyReadonlyArray2<'_, i64>],
    reverse: bool,
    result: &mut [M::Out],
) {
    let map: DashMap<i64, M::Acc> = DashMap::new();
    if reverse {
        for group in topo_groups.iter().rev() {
            process_level(metric, group, &map, reverse, result);
        }
    } else {
        for group in topo_groups.iter() {
            process_level(metric, group, &map, reverse, result);
        }
    }
}

fn process_level<M: Metric>(
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

    // Upstream (forward): each source uid is unique in the level and never needed
    // again, so it is removed (moved). Downstream (reverse): a source did may feed
    // several uids, so it is cloned. `target` is the node whose result this level
    // computes.
    let (source, target): (&[i64], &[i64]) = if reverse { (did, uid) } else { (uid, did) };

    source
        .par_iter()
        .zip(target.par_iter())
        .for_each(|(&s, &t)| {
            let s_acc = if reverse {
                map.get(&s)
                    .map(|e| e.clone())
                    .unwrap_or_else(|| metric.singleton(s as usize))
            } else {
                map.remove(&s)
                    .map(|e| e.1)
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
