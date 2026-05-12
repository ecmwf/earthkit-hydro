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
