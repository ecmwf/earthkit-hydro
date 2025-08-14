def get_array_backend(x):

    if type(x) is str:
        mod = x
    else:
        mod = type(x).__module__

    if "torch" in mod:
        from .torch_backend import TorchBackend

        return TorchBackend()
    elif "tensorflow" in mod:
        from .tensorflow_backend import TFBackend

        return TFBackend()
    elif "jax" in mod:
        from .jax_backend import JAXBackend

        return JAXBackend()
    elif "cupy" in mod:
        from .cupy_backend import CuPyBackend

        return CuPyBackend()
    elif "numpy" in mod:
        from .numpy_backend import NumPyBackend

        return NumPyBackend()
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")
