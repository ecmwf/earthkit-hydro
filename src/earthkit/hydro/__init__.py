import earthkit.hydro.catchments  # for dynamic function creation
import earthkit.hydro.readers  # for tests
import earthkit.hydro.river_network
import earthkit.hydro.subcatchments  # for dynamic function creation
import earthkit.hydro.upstream  # for dynamic function creation
import earthkit.hydro.zonal  # for dynamic function creation
from earthkit.hydro.accumulation import flow_downstream, flow_upstream
from earthkit.hydro.catchments import calculate_catchment_metric
from earthkit.hydro.catchments import find as find_catchments
from earthkit.hydro.movement import move_downstream, move_upstream
from earthkit.hydro.river_network import create as create_river_network
from earthkit.hydro.river_network import load as load_river_network
from earthkit.hydro.subcatchments import calculate_subcatchment_metric
from earthkit.hydro.subcatchments import find as find_subcatchments
from earthkit.hydro.upstream import calculate_upstream_metric
from earthkit.hydro.zonal import calculate_zonal_metric

from ._version import __version__
