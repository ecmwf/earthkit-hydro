use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};

#[pyfunction]
fn propagate_labels<'py>(
    py: Python<'py>,
    labels: &PyArray1<i32>,
    inlets: &PyArray1<usize>,
    downstream_nodes: &PyArray1<usize>,
    n_nodes: usize,
) -> &'py PyArray1<i32> {

    let labels = unsafe { labels
    .as_slice_mut()
    .expect("Could not get mutable slice for labels")};

    let downstream = unsafe { downstream_nodes
        .as_slice()
        .expect("Failed to get downstream_nodes slice")};

    let mut current = unsafe { inlets
        .as_slice()
        .expect("Failed to get inlets slice")
        .to_vec()};

    let mut next = Vec::with_capacity(current.len());

    for n in 1..=n_nodes {
        current.retain(|&i| i != n_nodes);

        if current.is_empty() {
            break;
        }

        for &i in &current {
            labels[i] = n as i32;
        }

        next.clear();
        for &i in &current {
            let d = downstream[i];
            if d != n_nodes {
                next.push(d);
            }
        }

        std::mem::swap(&mut current, &mut next);
    }

    labels.to_pyarray(py)
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(propagate_labels, m)?)?;
    Ok(())
}
