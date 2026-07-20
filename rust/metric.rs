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
            merge_into(metric, map, t, &s_acc);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct TestAcc {
        value: usize,
        clones: Arc<AtomicUsize>,
    }

    impl Clone for TestAcc {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::Relaxed);
            Self {
                value: self.value,
                clones: Arc::clone(&self.clones),
            }
        }
    }

    struct SumMetric {
        clones: Arc<AtomicUsize>,
    }

    impl Metric for SumMetric {
        type Acc = TestAcc;
        type Out = usize;

        fn singleton(&self, _node: usize) -> TestAcc {
            TestAcc {
                value: 1,
                clones: Arc::clone(&self.clones),
            }
        }

        fn merge(&self, dst: &mut TestAcc, src: &TestAcc) {
            dst.value += src.value;
        }

        fn finalize(&self, acc: &TestAcc) -> usize {
            acc.value
        }
    }

    #[test]
    fn grouped_fanout_does_not_clone_accumulators() {
        const EDGES: i64 = 10_000;
        let clones = Arc::new(AtomicUsize::new(0));
        let metric = SumMetric {
            clones: Arc::clone(&clones),
        };
        let map = DashMap::new();
        let source = vec![0; EDGES as usize];
        let target: Vec<i64> = (1..=EDGES).collect();
        let last_use = HashMap::from([(0, 0)]);

        process_grouped(&metric, &source, &target, &map, &last_use, 0);

        assert_eq!(clones.load(Ordering::Relaxed), 0);
        assert!(!map.contains_key(&0));
        assert_eq!(map.len(), EDGES as usize);
        assert!(target
            .iter()
            .all(|target| map.get(target).unwrap().value == 2));
    }

    #[test]
    fn grouped_source_is_retained_until_its_last_level() {
        let clones = Arc::new(AtomicUsize::new(0));
        let metric = SumMetric {
            clones: Arc::clone(&clones),
        };
        let map = DashMap::new();
        let last_use = HashMap::from([(0, 1)]);

        process_grouped(&metric, &[0], &[1], &map, &last_use, 0);
        assert_eq!(map.get(&0).unwrap().value, 1);
        assert_eq!(map.get(&1).unwrap().value, 2);

        process_grouped(&metric, &[0], &[2], &map, &last_use, 1);
        assert!(!map.contains_key(&0));
        assert_eq!(map.get(&2).unwrap().value, 2);
        assert_eq!(clones.load(Ordering::Relaxed), 0);
    }
}
