use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;

#[pyfunction]
fn propagate_labels<'py>(
    py: Python<'py>,
    labels: &PyArray1<i64>,
    sources: &PyArray1<usize>,
    sinks: &PyArray1<usize>,
    downstream_nodes: &PyArray1<usize>,
    n_nodes: usize,
) -> PyResult<&'py PyArray1<i64>> {

    let labels = unsafe { labels
    .as_slice_mut()
    .expect("Failed to get labels slice")};

    let downstream = unsafe { downstream_nodes
        .as_slice()
        .expect("Failed to get downstream_nodes slice")};

    let mut current = unsafe { sources
        .as_slice()
        .expect("Failed to get sources slice")
        .to_vec() };

    let sinks = unsafe { sinks
            .as_slice()
            .expect("Failed to get sinks slice")
            .to_vec() };

    let mut next = Vec::with_capacity(current.len());

    next.clear();
        for &i in &current {
            let d = downstream[i];
            if d != n_nodes {
                next.push(d);
            }
        }
    std::mem::swap(&mut current, &mut next);


    for n in 1..=n_nodes {
        if current.is_empty() {
            for &i in &sinks {
                labels[i] = (n as i64) - 1;
            }
            break;
        }

        for &i in &current {
            labels[i] = n as i64;
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

    if !current.is_empty() {
        return Err(PyErr::new::<PyValueError, _>("River Network contains a cycle."));
    }

    Ok(labels.to_pyarray(py))
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(propagate_labels, m)?)?;
    Ok(())
}
