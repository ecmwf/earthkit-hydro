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
  <!-- <a href="https://codecov.io/gh/ecmwf/earthkit-hydro">
    <img src="https://codecov.io/gh/ecmwf/earthkit-hydro/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a> -->
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/ecmwf/earthkit-hydro/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-hydro?color=purple&label=Release" alt="Latest Release">
  </a>
</p>

<p align="center">
  <!-- <a href="#quick-start">Quick Start</a>
  • -->
  <a href="#installation">Installation</a>
  •
  <a href="https://earthkit-hydro.readthedocs.io">Documentation</a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

**earthkit-hydro** is a Python library for common hydrological functions. It is the hydrological component of [earthkit](https://github.com/ecmwf/earthkit).

## Main Features

<table>
  <tr>
    <td><img src="./docs/images/glofas.png" height="300" alt="Adapted from: doc_figure" /></td>
    <td><img src="./docs/images/array_backends_with_xr.png" height="300" alt="Array backends with xr" /></td>
  </tr>
</table>

- Catchment delineation
- Catchment-based statistics
- Directional flow-based accumulations
- River network distance calculations
- Upstream/downstream field propagation
- Bifurcation handling
- Custom weighting and decay support
- Support for PCRaster, CaMa-Flood, HydroSHEDS and MERIT-Hydro river network formats
- Compatible with major array-backends: xarray, numpy, cupy, torch, jax and tensorflow
- GPU support
- Differentiable operations suitable for machine learning


## Installation
For a default installation, run

```
pip install earthkit-hydro
```

*Developer instructions:*

For a developer setup (includes linting and test libraries), run

```
conda create -n hydro python=3.12
conda activate hydro
conda install -c conda-forge rust
git clone https://github.com/ecmwf/earthkit-hydro.git
cd earthkit-hydro
pip install -e .[dev]
pre-commit install
```
Note: this project is a mixed Rust-Python project with a pure Python fallback. To handle this, the behaviour of the install is based on an environmental variable `USE_RUST`, with the following behaviour
- `Not set or any other value (default behaviour)`:
Attempts to build with Rust and if failure, skips and falls back to pure Python implementation.
- `USE_RUST=0`:
Builds pure Python implementation.
- `USE_RUST=1`:
Builds with Rust and fails if something goes wrong.


## Licence

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

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
