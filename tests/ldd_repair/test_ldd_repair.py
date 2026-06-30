import os
from pathlib import Path

from earthkit.hydro._readers import find_main_var, import_earthkit_or_prompt_install
from earthkit.hydro._readers.ldd_repair import LddRepair


def run_lddrepair_test(test_name: str = "ldd", river_network_format: str = "pcr_d8", save_repaired: bool = False):
    current_folder = os.path.dirname(__file__)
    data_folder = Path(current_folder, "data")
    ldd_path = data_folder / f"{test_name}.nc"
    ldd_repaired_path = data_folder / f"{test_name}_repaired.nc"

    ekd = import_earthkit_or_prompt_install(river_network_format=river_network_format, source="file")

    ldd_ds = ekd.from_source("file", ldd_path).to_xarray(mask_and_scale=False)
    var_name = find_main_var(ldd_ds)
    ldd_array = ldd_ds[var_name].values.astype(int)
    ldd_array = LddRepair(ldd_array, river_network_format=river_network_format).repair() # type: ignore

    if save_repaired:
        ldd_repaired_path_temp = data_folder / f"{test_name}_repaired_temp.nc"
        ldd_ds_final = ldd_ds.copy()
        ldd_ds_final[var_name].values = ldd_array
        ldd_ds_final.to_netcdf(ldd_repaired_path_temp)

    ldd_repaired_ds = ekd.from_source("file", ldd_repaired_path).to_xarray(
        mask_and_scale=False
    )
    var_name = find_main_var(ldd_repaired_ds)
    ldd_repaired_array = ldd_repaired_ds[var_name].values.astype(int)

    ldd_ds.close()
    ldd_repaired_ds.close()

    assert (
        ldd_array == ldd_repaired_array
    ).all(), "LDD repair did not produce the expected output. Please check the generated repaired LDD and compare it with the expected repaired LDD."


def test_lddrepair():
    run_lddrepair_test(test_name="ldd", river_network_format="pcr_d8")


def test_lddrepair_esri():
    save = False
    run_lddrepair_test(test_name="ldd_esri", river_network_format="esri_d8", save_repaired=save)

