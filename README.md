# earthkit-hydro

**DISCLAIMER**

> This project is in the **BETA** stage of development. Please be aware that interfaces and functionality may change as the project develops. If this software is to be used in operational systems you are **strongly advised to use a released tag in your system configuration**, and you should be willing to accept incoming changes and bug fixes that require adaptations on your part. ECMWF **does use** this software in operations and abides by the same caveats.

**earthkit-hydro** is a computationally efficient library for common hydrological functions

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

```
import earthkit.hydro as ekh
```

- ### Readers

```
ekh.load_river_network(domain="efas",version="5",...)
```
Loads a precomputed `RiverNetwork`. Current options are
- domain: "efas", version: "5"
- domain: "glofas", version: "4"


```
ekh.from_netcdf_d8(filename)
```
Creates a `RiverNetwork` from a d8 netcdf format.

```
ekh.from_netcdf_cama(filename, type)
```
Creates a `RiverNetwork` from a CAMA netcdf format of type "downxy" or "nextxy".

```
ekh.from_bin_cama(filename, type)
```
Creates a `RiverNetwork` from a CAMA bin format of type "downxy" or "nextxy".

- ### RiverNetwork methods

```
network.accuflux(field)
```
Calculates the total accumulated flux down a river network.

<img src="docs/accuflux.gif" width="200px" height="160px" />

```
network.upstream(field)
```
Updates each node with the sum of its upstream nodes.

```
network.downstream(field)
```
Updates each node with its downstream node.

```
network.catchment(field)
```
Finds the catchments (all upstream nodes of specified nodes, with overwriting).

<img src="docs/catchment.gif" width="200px" height="160px" />

```
network.subcatchment(field)
```
Finds the subcatchments (all upstream nodes of specified nodes, without overwriting).

<img src="docs/subcatchment.gif" width="200px" height="160px" />

```
network.export(filename)
```
Exports the `RiverNetwork` as a joblib pickle.

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
