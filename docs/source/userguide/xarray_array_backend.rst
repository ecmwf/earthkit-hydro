Handling xarray and multiple array backends
===========================================

earthkit-hydro is designed to work seamlessly with xarray and multiple array backends, including numpy, cupy, torch, jax, and tensorflow. This flexibility allows users to choose the backend that best suits their computational needs, whether for CPU or GPU operations.

Changing the river network array backend
----------------------------------------

By default, a river network is loaded for the numpy backend. However, it can be easily converted to other backends via the `to_device` method.

.. code-block:: python

    import earthkit.hydro as ekh

    network = ekh.river_network.load("efas", "5").to_device(array_backend="torch")

The network can also be transferred to a specific device such as a GPU

.. code-block:: python

    network = ekh.river_network.load("efas", "5").to_device("cuda", "torch")

xarray and array-backend agnostic operations
--------------------------------------------

earthkit-hydro is created with array-backend agnostic operations in mind. It is structured such that each operation has two versions: a top-level xarray-oriented version and an array version.

.. code-block:: python

    # xarray-oriented operation (returns xarray)
    ekh.submodule.operation(...)

    # array-oriented operations (returns arrays)
    ekh.submodule.array.operation(...)

This design allows users to work primarily with xarray objects, while still having access to lower-level array operations when needed.

The philosophy for the xarray-oriented operations is to return the same type of object as the input where possible, ensuring consistency across operations. For example,

.. code-block:: python

    import earthkit.data as ekd
    network = ekh.river_network.load("efas", "5")

    # xarray dataset inputted, xarray dataset returned
    ds = ekd.from_source("file", "data.nc").to_xarray()
    output = ekh.upstream.sum(network, ds)
    assert isinstance(output, xr.Dataset)

    # xarray dataarray inputted, xarray dataarray returned
    da = ds['main_variable']
    output = ekh.upstream.sum(network, da)
    assert isinstance(output, xr.DataArray)

    # If no xarray is provided, dataarray is returned
    # array inputted, xarray dataarray returned
    arr = np.ones(network.shape)
    output = ekh.upstream.sum(network, da)
    assert isinstance(output, xr.DataArray)

Array backends are automatically detected from the river network i.e.

.. code-block:: python

    network = ekh.river_network.load("efas", "5")
    input_array = np.ones(network.shape)
    output = ekh.upstream.array.sum(network, input_array) # numpy array returned
    assert isinstance(output, numpy.ndarray)

    network = ekh.river_network.load("efas", "5").to_device(array_backend="torch")
    input_array = torch.ones(network.shape)
    output = ekh.upstream.array.sum(network, input_array) # torch tensor returned
    assert isinstance(output, torch.Tensor)

    # Note: trying to use a numpy array with a torch-backed river network will raise

This means that users can switch between array backends without changing their code, as long as the input and output types are consistent. It also allows seamless support for xarray objects with a cupy backend.
