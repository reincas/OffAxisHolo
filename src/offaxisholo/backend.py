"""
Array backend selection (CPU / GPU) for all numerical calculations.

The library is written against the Python array API standard
(https://data-apis.org/array-api/) via ``array-api-compat``. The concrete array
library is selected at runtime:

    "numpy" (default): calculations on the CPU.
    "cupy":            calculations on an NVIDIA GPU (CUDA). Requires the optional
                       dependency ``pip install offaxisholo[gpu]``.

Usage
-----
    import offaxisholo as oah

    oah.set_backend("cupy")              # global
    oah.set_precision("single")          # complex64 / float32 (default "double")

    with oah.use_backend("cupy", precision="single"):   # temporary
        processor = reconstruct_hologram(holo, background)
    phase = oah.to_numpy(processor.phase_map)

Conventions
-----------
* Entry points (Hologram, angularSpectrum, calculate_efield, simulation field
  generators) move their input to the active backend with ``asarray``.
* All further operations stay on the backend (device) of their input arrays.
* Plotting, saving and file IO always work on NumPy arrays (``to_numpy``).

Precision
---------
"double" (float64 / complex128) is the default and reproduces the former NumPy
results. "single" (float32 / complex64) is considerably faster on consumer GPUs,
whose float64 throughput is only 1/32 to 1/64 of their float32 throughput. Its
accuracy for a given application has to be validated (see ToDos.txt).

Operations outside of the array API standard (padding, ndimage filters, DCT) are
dispatched by the helper functions of this module.
"""

from contextlib import contextmanager

import array_api_compat
import numpy as np

BACKENDS = ("numpy", "cupy")
PRECISIONS = ("double", "single")

_state = {"backend": "numpy", "precision": "double"}


# ----------------------------------------------------------------- selection
def _import_cupy():
    """Import CuPy and make sure that a CUDA device is available."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            "Backend 'cupy' requires CuPy. Install it with "
            "'pip install offaxisholo[gpu]' (or the cupy-cudaXXx wheel matching "
            "your CUDA driver)."
        ) from err
    try:
        n_devices = cupy.cuda.runtime.getDeviceCount()
    except Exception as err:  # noqa: BLE001 - CUDA raises various error types
        raise RuntimeError(f"No usable CUDA device found: {err}") from err
    if n_devices < 1:
        raise RuntimeError("No CUDA device found for backend 'cupy'.")
    return cupy


def set_backend(name: str) -> None:
    """Select the array backend ('numpy' or 'cupy') for all calculations."""
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Possible backends: {BACKENDS}.")
    if name == "cupy":
        _import_cupy()
    _state["backend"] = name


def get_backend() -> str:
    """Return the name of the active backend."""
    return _state["backend"]


def set_precision(precision: str) -> None:
    """Select the floating point precision: 'double' (default) or 'single'."""
    precision = precision.lower()
    if precision not in PRECISIONS:
        raise ValueError(
            f"Unknown precision '{precision}'. Possible values: {PRECISIONS}."
        )
    _state["precision"] = precision


def get_precision() -> str:
    """Return the active precision ('double' or 'single')."""
    return _state["precision"]


@contextmanager
def use_backend(name: str = None, precision: str = None):
    """
    Temporarily select backend and/or precision. None keeps the current value.

        with use_backend("cupy", precision="single"):
            ...
    """
    previous = dict(_state)
    try:
        if name is not None:
            set_backend(name)
        if precision is not None:
            set_precision(precision)
        yield
    finally:
        _state.update(previous)


# ------------------------------------------------------------------ namespace
def _active_namespace():
    if _state["backend"] == "cupy":  # availability checked in set_backend
        import array_api_compat.cupy as xp
    else:
        import array_api_compat.numpy as xp
    return xp


def get_namespace(*arrays):
    """
    Return the array API namespace of the given arrays (None entries are
    ignored). Without arrays, the namespace of the active backend is returned.
    """
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        return _active_namespace()
    return array_api_compat.array_namespace(*arrays)


def real_dtype(xp=None):
    """Real floating point dtype of the active precision."""
    xp = xp if xp is not None else _active_namespace()
    return xp.float64 if _state["precision"] == "double" else xp.float32


def complex_dtype(xp=None):
    """Complex floating point dtype of the active precision."""
    xp = xp if xp is not None else _active_namespace()
    return xp.complex128 if _state["precision"] == "double" else xp.complex64


# ------------------------------------------------------------------ transfer
def is_array(x) -> bool:
    """True for arrays of any supported backend."""
    return array_api_compat.is_array_api_obj(x)


def is_gpu(x) -> bool:
    """True if x is a GPU (CuPy) array."""
    return array_api_compat.is_cupy_array(x)


def to_numpy(x):
    """
    Return x as NumPy array (copy from the GPU if necessary). Values that are not
    arrays (None, scalars, ...) are returned unchanged.
    """
    if x is None:
        return None
    if array_api_compat.is_cupy_array(x):
        return x.get()
    if is_array(x):
        return np.asarray(x)
    return x


def asarray(x, dtype=None, *, like=None):
    """
    Convert x to an array of the active backend (or of the backend of the array
    ``like``, if given), with optional dtype conversion. Data is copied between
    host and device only if necessary.
    """
    use_cupy = is_gpu(like) if like is not None else _state["backend"] == "cupy"
    if use_cupy:
        cupy = _import_cupy()
        return cupy.asarray(x, dtype=dtype)
    return np.asarray(to_numpy(x), dtype=dtype)


# --------------------------------------------- operations outside the standard
def _native(x):
    """Return the native array module (numpy or cupy) of x."""
    if is_gpu(x):
        import cupy

        return cupy
    return np


def _ndimage(x):
    if is_gpu(x):
        import cupyx.scipy.ndimage as ndimage
    else:
        import scipy.ndimage as ndimage
    return ndimage


def _scipy_fft(x):
    if is_gpu(x):
        import cupyx.scipy.fft as sfft
    else:
        import scipy.fft as sfft
    return sfft


def angle(x):
    """Wrapped phase arg(x) = atan2(Im x, Re x) (identical to numpy.angle)."""
    xp = get_namespace(x)
    return xp.atan2(xp.imag(x), xp.real(x))


def pad(x, pad_width, **kwargs):
    """numpy.pad / cupy.pad."""
    return _native(x).pad(x, pad_width, **kwargs)


def gaussian_filter(x, sigma, **kwargs):
    """scipy.ndimage.gaussian_filter / cupyx.scipy.ndimage.gaussian_filter."""
    return _ndimage(x).gaussian_filter(x, sigma, **kwargs)


def maximum_filter(x, size, **kwargs):
    """scipy.ndimage.maximum_filter / cupyx.scipy.ndimage.maximum_filter."""
    return _ndimage(x).maximum_filter(x, size=size, **kwargs)


def dctn(x, type=2, norm="ortho"):
    """N-dimensional discrete cosine transform (scipy.fft / cupyx.scipy.fft)."""
    return _scipy_fft(x).dctn(x, type=type, norm=norm)


def idctn(x, type=2, norm="ortho"):
    """Inverse N-dimensional discrete cosine transform."""
    return _scipy_fft(x).idctn(x, type=type, norm=norm)
