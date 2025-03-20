<a href="https://github.com/ecmwf/earthkit-hydro">
  <p align="center">
    <picture>
      <source srcset="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-hydro-dark.svg" media="(prefers-color-scheme: dark)">
      <img src="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-hydro-light.svg" height="120">
    </picture>
  </p>
</a>

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
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/ecmwf/earthkit-hydro/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-hydro?color=blue&label=Release&style=flat-square" alt="Latest Release">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

**earthkit-hydro** is a Python library for common hydrological functions.

## Main Features

- Support for PCRaster, CaMa-Flood and HydroSHEDS river networks
- Computing statistics over catchments and subcatchments
- Finding catchments and subcatchments
- Calculation of upstream or downstream fields
- Handle arbitrary missing values
- Handle N-dimensional fields

## Installation
For default installation, run

```
pip install earthkit-hydro
```

For a developer installation (includes linting and test libraries), run

```
git clone https://github.com/ecmwf/earthkit-hydro.git
cd earthkit-hydro
pip install -e .[dev]
pre-commit install
```

## Documentation
**An [example notebook](docs/notebooks/example.ipynb) showing how to use the earthkit-hydro is provided in addition to the documentation below.**

earthkit-hydro can be imported as following:
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
ekh.river_network.load(domain, version)
```
Loads a precomputed `RiverNetwork`. Current options can be listed with `ekh.river_network.available()` and are:
| `domain` | `version` | Details | Note | Attribution |
| --- | --- | --- | --- | --- |
| "efas" | "5" | 1arcmin European |  | [<sup>1</sup>](#attrib1) |
| "efas" | "4" | 5km European | Smaller domain than v5 | [<sup>1</sup>](#attrib1) |
| "glofas" | "4" | 3arcmin global |  | [<sup>2</sup>](#attrib2) |
| "glofas" | "3" | 6arcmin global |  | [<sup>2</sup>](#attrib2) |
| "cama_03min" | "4" | 3arcmin global |  | [<sup>3</sup>](#attrib3) |
| "cama_05min" | "4" | 5arcmin global |  | [<sup>3</sup>](#attrib3) |
| "cama_06min" | "4" | 6arcmin global |  | [<sup>3</sup>](#attrib3) |
| "cama_15min" | "4" | 15arcmin global |  | [<sup>3</sup>](#attrib3) |
| "hydrosheds_05min" | "1" | 5arcmin global | 56° South to 84° North | [<sup>4</sup>](#attrib4) |
| "hydrosheds_06min" | "1" | 6arcmin global | 56° South to 84° North | [<sup>4</sup>](#attrib4) |


```
ekh.river_network.create(path, river_network_format, source="file")
```
Creates a `RiverNetwork`. Current options are
- river_network_format: "esri_d8", "pcr_d8", "cama" or "precomputed"
- source: An earthkit-data compatable source. See [list](https://earthkit-data.readthedocs.io/en/latest/guide/sources.html).

### Computing Metrics Over River Networks
There are four high-level ways to compute metrics depending on the use-case.

#### Metrics Over Upstream Nodes
```
ekh.upstream.sum(river_network, field, weights=None)
ekh.upstream.max(river_network, field, weights=None)
ekh.upstream.min(river_network, field, weights=None)
ekh.upstream.mean(river_network, field, weights=None)
ekh.upstream.product(river_network, field, weights=None)
```
Given an input field, returns as output a new field with the upstream metric calculated for each cell.

#### Metrics Over Catchments
```
ekh.catchments.sum(river_network, field, stations, weights=None)
ekh.catchments.max(river_network, field, stations, weights=None)
ekh.catchments.min(river_network, field, stations, weights=None)
ekh.catchments.mean(river_network, field, stations, weights=None)
ekh.catchments.product(river_network, field, stations, weights=None)
```
Given a field and a list of points defining stations, calculates the metric over all upstream nodes for each of the stations.

#### Metrics Over Subcatchments
```
ekh.subcatchments.sum(river_network, field, stations, weights=None)
ekh.subcatchments.max(river_network, field, stations, weights=None)
ekh.subcatchments.min(river_network, field, stations, weights=None)
ekh.subcatchments.mean(river_network, field, stations, weights=None)
ekh.subcatchments.product(river_network, field, stations, weights=None)
```
Given a field and a list of points defining stations, finds the subcatchments defined by the stations and computes the metric for each subcatchment.

#### Metrics Over Arbitrary Zones
```
ekh.zonal.sum(field, labels, weights=None, return_field=False)
ekh.zonal.max(field, labels, weights=None, return_field=False)
ekh.zonal.min(field, labels, weights=None, return_field=False)
ekh.zonal.mean(field, labels, weights=None, return_field=False)
ekh.zonal.product(field, labels, weights=None, return_field=False)
```
Calculates a metric over the input field for each zone defined by the labels field. If return_field is True, returns a field otherwise returns a dictionary of {label: metric} pairs.

_(for advanced users)_
Similarly, one can also use a low-level API via
```
ekh.calculate_upstream_metric(river_network, field, metric, weights=None)
ekh.calculate_catchment_metric(river_network, field, stations, metric, weights=None)
ekh.calculate_subcatchment_metric(river_network, field, stations, metric, weights=None)
ekh.calculate_zonal_metric(field, labels, metric, weights=None)

# applies the ufunc on the field starting from the sources all the way down to the sinks
ekh.flow_downstream(river_network, field, ufunc=np.add)
# applies the ufunc on the field starting from the sinks all the way up to the sources
ekh.flow_upstream(river_network, field, ufunc=np.add)
```
These are analagous to above.

### Finding Catchments and Subcatchments

```
ekh.catchments.find(river_network, field)
```
Finds the catchments (all upstream nodes of specified nodes, with overwriting).\
$$v_i^{\prime} = v_j^{\prime}  ~ \text{if} ~  v_j^{\prime} \neq 0 ~ \text{else} ~ v_i, ~j ~ \text{s.t.} ~ i \rightarrow j$$

<img src="docs/images/catchment.gif" width="200px" height="160px" />

```
ekh.subcatchments.find(river_network, field)
```
Finds the subcatchments (all upstream nodes of specified nodes, without overwriting).\
$$v_i^{\prime} = v_j^{\prime}  ~ \text{if} ~  (v_j^{\prime} \neq 0 ~ \text{and} ~ v_j = 0) ~ \text{else} ~ v_i, ~j ~ \text{s.t.} ~ i \rightarrow j$$

<img src="docs/images/subcatchment.gif" width="200px" height="160px" />

### Calculating Upstream or Downstream Fields

```
ekh.move_downstream(river_network, field, ufunc=np.add)
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

earthkit-hydro provides many functions with PCRaster equivalents, summarised below:

| PCRaster | earthkit-hydro |  Note |
|---|---|---|
| accuflux | upstream.sum | |
| catchmenttotal | upstream.sum |  |
| areatotal | zonal.sum | return_field=True |
| areaaverage | zonal.mean | return_field=True |
| areamaximum | zonal.max | return_field=True |
| areaminimum | zonal.min | return_field=True |
| downstream | move_upstream | |
| upstream | move_downstream | |
| catchment | catchments.find | |
| subcatchment | subcatchments.find | |
| abs, sin, cos, tan, ...  | np.abs, np.sin, np.cos, np.tan, ... | any numpy operations can be directly used |

_Points of difference_
- earthkit-hydro treats missing values as np.nans i.e. any arithmetic involving a missing value will return a missing value. PCRaster does not always handle missing values exactly the same.
- earthkit-hydro can handle vector fields and fields of integers, floats, bools. PCRaster supports a restricted subset of this.

## Attributions
<a id="attrib1"><sup>1</sup></a>
The EFAS river network is available under the conditions set out in the [European Commission Reuse and Copyright Notice](https://data.jrc.ec.europa.eu/licence/com_reuse) and is available at [https://data.jrc.ec.europa.eu/dataset/f572c443-7466-4adf-87aa-c0847a169f23](https://data.jrc.ec.europa.eu/dataset/f572c443-7466-4adf-87aa-c0847a169f23).

    Choulga, Margarita; Moschini, Francesca; Mazzetti, Cinzia; Grimaldi, Stefania; Disperati, Juliana; Beck, Hylke; Salamon, Peter; Prudhomme, Christel (2023): LISFLOOD static and parameter maps for Europe. European Commission, Joint Research Centre (JRC) [Dataset] PID: http://data.europa.eu/89h/f572c443-7466-4adf-87aa-c0847a169f23

<a id="attrib2"><sup>2</sup></a>
The GloFAS river network is available under the conditions set out in the [European Commission Reuse and Copyright Notice](https://data.jrc.ec.europa.eu/licence/com_reuse) and is available at [https://data.jrc.ec.europa.eu/dataset/68050d73-9c06-499c-a441-dc5053cb0c86](https://data.jrc.ec.europa.eu/dataset/68050d73-9c06-499c-a441-dc5053cb0c86).

    Choulga, Margarita; Moschini, Francesca; Mazzetti, Cinzia; Disperati, Juliana; Grimaldi, Stefania; Beck, Hylke; Salamon, Peter; Prudhomme, Christel (2023): LISFLOOD static and parameter maps for GloFAS. European Commission, Joint Research Centre (JRC) [Dataset] PID: http://data.europa.eu/89h/68050d73-9c06-499c-a441-dc5053cb0c86

<a id="attrib3"><sup>3</sup></a>
The CaMa river networks are available under [CC-BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/) and are available at [http://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/](http://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/).

    Yamazaki, Dai; Ikeshima, Daiki; Sosa, Jeison; Bates, Paul D.; Allen, George H.; Pavelsky, Tamlin M. (2019): MERIT Hydro: A high-resolution global hydrography map based on latest topography datasets. Water Resources Research, vol.55, pp.5053-5073, 2019, DOI: 10.1029/2019WR024873

<a id="attrib4"><sup>4</sup></a>
The HydroSHEDS river networks are available under the conditions set out in the [HydroSHEDS Version One Licence Agreement](https://data.hydrosheds.org/file/technical-documentation/HydroSHEDS_TechDoc_v1_4.pdf) and are available at [https://www.hydrosheds.org](https://www.hydrosheds.org).

    Lehner, Bernhard; Verdin, Kristine; Jarvis, Andy (2008): New global hydrography derived from spaceborne elevation data. Eos, Transactions, 89(10): 93-94. Data available at https://www.hydrosheds.org.

## Licence

[Apache Licence 2.0](LICENCE)

In applying this license, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
