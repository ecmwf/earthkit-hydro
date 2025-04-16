use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn compute_topological_labels_rust(py: Python, sources: &PyArray1<i64>, sinks: &PyArray1<i64>, downstream_nodes: &PyArray1<i64>) -> Py<PyArray1<i64>> {
    let sources_slice = unsafe { sources.as_slice().unwrap() };
    let sinks_slice = unsafe { sinks.as_slice().unwrap() };
    let downstream_nodes = unsafe { downstream_nodes.as_slice().unwrap() };

    let n_nodes = downstream_nodes.len();

    let mut labels = vec![0; n_nodes];

    let mut current_inlets: Vec<i64> = sources_slice.iter().map(|x| downstream_nodes[*x as usize]).collect();
    let mut loop_size = 0;

    for n in 1..=n_nodes {
        current_inlets = current_inlets.iter().map(|v| *v).filter(|x| *x != n_nodes as i64).collect();
        let inlet_size = current_inlets.len();
        if inlet_size == 0 {
            loop_size = n;
            break;
        }
        for &v in current_inlets.iter(){
            labels[v as usize] = n;
        }
        current_inlets = current_inlets.iter().map(|x| downstream_nodes[*x as usize]).collect();
    };

    for &v in sinks_slice.iter(){
        labels[v as usize] = loop_size-1;
    }

    let labels_int: Vec<i64> = labels.iter().map(|&x| x as i64).collect();
    labels_int.into_pyarray(py).to_owned()
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_topological_labels_rust, m)?)?;
    Ok(())
}
