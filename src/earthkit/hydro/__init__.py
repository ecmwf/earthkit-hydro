# imports to achieve the dynamic function creation
import earthkit.hydro.catchments
import earthkit.hydro.readers  # for tests
import earthkit.hydro.subcatchments
import earthkit.hydro.upstream
from earthkit.hydro.accumulation import flow_downstream
from earthkit.hydro.catchments import calculate_catchment_metric
from earthkit.hydro.catchments import find as find_catchments
from earthkit.hydro.label import calculate_metric_for_labels
from earthkit.hydro.movement import move_downstream, move_upstream
from earthkit.hydro.subcatchments import calculate_subcatchment_metric
from earthkit.hydro.subcatchments import find as find_subcatchments
from earthkit.hydro.upstream import calculate_upstream_metric
