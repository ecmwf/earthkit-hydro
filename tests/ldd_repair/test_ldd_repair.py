import os
from pathlib import Path

from earthkit.hydro._readers import find_main_var, import_earthkit_or_prompt_install
from earthkit.hydro._readers.ldd_repair import LddRepair


def test_lddrepair():
    current_folder = os.path.dirname(__file__)
    data_folder = Path(current_folder, "data")
    ldd_path = data_folder / "ldd.nc"
    ldd_repaired_path = data_folder / "ldd_repaired.nc"

    ekd = import_earthkit_or_prompt_install("pcr_d8", "file")

    ldd_ds = ekd.from_source("file", ldd_path).to_xarray(mask_and_scale=False)
    var_name = find_main_var(ldd_ds)
    ldd_array = ldd_ds[var_name].values.astype(int)
    ldd_array = LddRepair(ldd_array).repair()

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


def test_lddrepair_esri():
    current_folder = os.path.dirname(__file__)
    data_folder = Path(current_folder, "data")
    ldd_path = data_folder / f"ldd_esri.nc"
    ldd_repaired_path = data_folder / f"ldd_esri_repaired.nc"

    ekd = import_earthkit_or_prompt_install("pcr_d8", "file")

    ldd_ds = ekd.from_source("file", ldd_path).to_xarray(mask_and_scale=False)
    var_name = find_main_var(ldd_ds)
    ldd_array = ldd_ds[var_name].values.astype(int)
    ldd_array = LddRepair(ldd_array, river_network_format="esri_d8").repair()

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


# def test_lddrepair2():
#     current_folder = os.path.dirname(__file__)
#     data_folder = Path("/mnt/nahaUsers/dispeju/PerGoncalo/2026testlddrepair")
#     name='paki'
#     ldd_path = data_folder / f"{name}.tif.nc"
#     ldd_repaired_path = data_folder / f"{name}_repaired.tif.nc"
#     ldd_repaired_earthkit_path = data_folder / f"{name}_repaired_earthkit.tif.nc"

#     ekd = import_earthkit_or_prompt_install("pcr_d8", "file")

#     ldd_ds = ekd.from_source("file", ldd_path).to_xarray(mask_and_scale=False)
#     var_name = find_main_var(ldd_ds)
#     ldd_array = ldd_ds[var_name].values.astype(int)
#     ldd_array = LddRepair(ldd_array).repair()

#     ldd_ds_final = ldd_ds.copy()
#     ldd_ds_final[var_name].values = ldd_array
#     ldd_ds_final.to_netcdf(ldd_repaired_earthkit_path)

#     ldd_repaired_ds = ekd.from_source("file", ldd_repaired_path).to_xarray(
#         mask_and_scale=False
#     )
#     var_name = find_main_var(ldd_repaired_ds)
#     ldd_repaired_array = ldd_repaired_ds[var_name].values.astype(int)

#     ldd_ds.close()
#     ldd_repaired_ds.close()

#     assert (
#         ldd_array == ldd_repaired_array
#     ).all(), "LDD repair did not produce the expected output. Please check the generated repaired LDD and compare it with the expected repaired LDD."


