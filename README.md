# earthkit-hydro

<p align="center">
  <picture>
    <source srcset="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-hydro-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-hydro-light.svg" height="120">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg" alt="ECMWF Software EnginE">
  </a>
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/emerging_badge.svg" alt="Maturity Level">
  </a>
  <a href="https://codecov.io/gh/ecmwf/earthkit-hydro">
    <img src="https://codecov.io/gh/ecmwf/earthkit-hydro/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/ecmwf/earthkit-hydro/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-hydro?color=blue&label=Release&style=flat-square" alt="Latest Release">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

**earthkit-hydro** is a Python library for common hydrological functions.

**DISCLAIMER**

> This project is in the **BETA** stage of development. Please be aware that interfaces and functionality may change as the project develops. If this software is to be used in operational systems you are **strongly advised to use a released tag in your system configuration**, and you should be willing to accept incoming changes and bug fixes that require adaptations on your part. ECMWF **does use** this software in operations and abides by the same caveats.

## Main Features

- Support for PCRaster, CaMa-Flood and HydroSHEDS river networks
- Computing statistics over catchments and subcatchments
- Finding catchments and subcatchments
- Calculation of upstream or downstream fields
- Handle arbitrary missing values
- Handle N-dimensional fields

## Installation
Clone source code repository

```
git clone https://github.com/ecmwf/earthkit-hydro.git
cd earthkit-hydro
```

Create and activate conda environment

```
conda create -n hydro python=3.10
conda activate hydro
```

For default installation, run

```
pip install .
```

For a developer installation (includes linting and test libraries), run

```
pip install -e .[dev]
pre-commit install
```

If you only plan to run the tests, instead run

```
pip install -e .[test]
```

## Documentation
Earthkit-hydro can be imported as following:
```
import earthkit.hydro as ekh
```

The package contains different ways of constructing or loading a `RiverNetwork` object. A `RiverNetwork` object is a representation of a river network on a grid.
It can be used to compute basic hydrological functions, such as propagating a scalar field along the river network or extract a catchment from the river network.

### Mathematical Details
Given a discretisation of a domain i.e. a set of points $\mathcal{D}=\{ (x_i, y_i)\}_{i=1}^N$, a river network is a directed acyclic graph $\mathcal{R}=(V,E)$ where the vertices $V \subseteq \mathcal{D}$. The out-degree of each vertex is at most 1 i.e. each point in the river network points to at most one downstream location.

For ease of notation, if an edge exists from $(x_i, y_i)$ to $(x_j, y_j)$, we write $i \rightarrow j$.

### Readers

```
ekh.load_river_network(domain="efas", version="5")
```
Loads a precomputed `RiverNetwork`. Current options are
- domain: "efas", version: "5"
- domain: "glofas", version: "4"


```
ekh.create_river_network(path, river_network_format, source)
```
Creates a `RiverNetwork`. Current options are
- river_network_format: "esri_d8", "pcr_d8", "cama" or "precomputed"
- source: An earthkit-data compatable source. See [list](https://earthkit-data.readthedocs.io/en/latest/guide/sources.html)

### Computing Metrics Over River Networks

```
ekh.calculate_catchment_metric(river_network, field, stations, metric, weights)
```
Calculates the metric over each catchment defined by stations. Current options are
- metric: "sum", "max", "min", "mean"

```
ekh.calculate_subcatchment_metric(river_network, field, stations, metric, weights)
```
Calculates the metric over each subcatchment defined by stations. Current options are
- metric: "sum", "max", "min", "mean"

```
ekh.calculate_upstream_metric(river_network, field, metric, weights)
```
Calculates a metric over all upstream nodes for a river network. If weights is provided, it is used to weight the field in the calculation. Options are
- metric: "sum", "mean", "max", "min"

```
ekh.flow_downstream(river_network, field)
```
_(for advanced users)_ Calculates the total accumulated flux down a river network.\
$$v_i^{\prime}=v_i+\sum_{j \rightarrow i}~v_j^{\prime}$$

<img src="docs/images/accuflux.gif" width="200px" height="160px" />

### Finding Catchments and Subcatchments

```
ekh.find_catchments(river_network, field)
```
Finds the catchments (all upstream nodes of specified nodes, with overwriting).\
$$v_i^{\prime} = v_j^{\prime}  ~ \text{if} ~  v_j^{\prime} \neq 0 ~ \text{else} ~ v_i, ~j ~ \text{s.t.} ~ i \rightarrow j$$

<img src="docs/images/catchment.gif" width="200px" height="160px" />

```
ekh.find_subcatchments(river_network, field)
```
Finds the subcatchments (all upstream nodes of specified nodes, without overwriting).\
$$v_i^{\prime} = v_j^{\prime}  ~ \text{if} ~  (v_j^{\prime} \neq 0 ~ \text{and} ~ v_j = 0) ~ \text{else} ~ v_i, ~j ~ \text{s.t.} ~ i \rightarrow j$$

<img src="docs/images/subcatchment.gif" width="200px" height="160px" />

### Calculating Upstream or Downstream Fields

```
ekh.move_downstream(river_network, field)
```
Updates each node with the sum of its upstream nodes.\
$$v_i^{\prime}=\sum_{j \rightarrow i}~v_j$$

```
ekh.move_upstream(river_network, field)
```
Updates each node with its downstream node.\
$$v_i^{\prime} = v_j, ~j ~ \text{s.t.} ~ i \rightarrow j$$

### Exporting or Masking a River Network

```
river_network.create_subnetwork(field)
```
Computes the river subnetwork defined by a field mask of the domain.

```
river_network.export(filename)
```
Exports the `RiverNetwork` as a joblib pickle.

## Migrating from PCRaster

earthkit-hydro provides many functons with pcraster equivalents, summarised below:

| PCRaster | earthkit-hydro |  Note |
|---|---|---|
| accuflux | calculate_upstream_metric | metric="sum" |
| catchmenttotal | calculate_upstream_metric | metric="sum"  |
| areatotal | calculate_metric_for_labels | metric="sum". Returns dictionary, not full field |
| areaaverage | calculate_metric_for_labels | metric="mean". Returns dictionary, not full field |
| areamaximum | calculate_metric_for_labels | metric="max". Returns dictionary, not full field |
| areaminimum | calculate_metric_for_labels | metric="min". Returns dictionary, not full field |
| downstream | move_upstream | |
| upstream | move_downstream | |
| catchment | find_catchments | |
| subcatchment | find_subcatchments | |
| abs, sin, cos, tan, ...  | np.abs, np.sin, np.cos, np.tan, ... | any numpy operations can be directly used |

_Points of difference_
- earthkit-hydro treats missing values as np.nans i.e. any arithmetic involving a missing value will return a missing value. PCRaster does not always handle missing values exactly the same.
- earthkit-hydro can handle vector fields and fields of integers, floats, bools. PCRaster supports a restricted subset of this.

## License

```
Copyright 2024, European Centre for Medium Range Weather Forecasts.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
