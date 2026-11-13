"""
Tests of the array backend (NumPy / CuPy) and of the backend independent parts of
the reconstruction.

GPU tests are skipped automatically if CuPy or a CUDA device is not available.

    pytest tests/test_backend.py -v
"""

import math

import cv2 as cv
import numpy as np
import pytest

import offaxisholo as oah
from offaxisholo import backend
from offaxisholo.reconstruction.numericalPropagation import angularSpectrum
from offaxisholo.reconstruction.phaseUnwrapping import (
    phase_unwrapping_fast2d,
    phase_unwrapping_ls_dct,
)
from offaxisholo.simulation import simulate as sim
from offaxisholo.single_functions.functions import reconstruct_hologram

N = 512
DHM = {
    "name": "synthetic",
    "dcRadius": 60,
    "pixelPitch": [0.1e-6, 0.1e-6],
    "wavelength": 0.675e-6,
    "propagationDistance": 1.1e-7,
}
MATERIAL = {"refractiveIndex": 1.5}


def _gpu_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


GPU = _gpu_available()
requires_gpu = pytest.mark.skipif(not GPU, reason="CuPy with CUDA device required")


@pytest.fixture(autouse=True)
def _restore_backend():
    """Every test starts with and restores the default backend settings."""
    with backend.use_backend("numpy", precision="double"):
        yield


def _bump(n=N, height=9.0, width=40.0):
    y, x = np.indices((n, n), dtype=float)
    return height * np.exp(-(((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * width**2)))


def _synthetic_hologram(obj_phase, seed):
    """Off-axis hologram |O + R|^2 with defocus-like aberration and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((N, N), dtype=float)
    carrier = np.exp(2j * np.pi * (x * 128 / N + y * 128 / N))
    aberration = 3.0 * (((x - N / 2) / N) ** 2 + ((y - N / 2) / N) ** 2)
    obj = np.exp(1j * (obj_phase + aberration))
    return np.abs(obj + carrier) ** 2 + 0.05 * rng.random((N, N))


@pytest.fixture(scope="module")
def holograms():
    return _synthetic_hologram(_bump(), 1), _synthetic_hologram(np.zeros((N, N)), 2)


def _reconstruct(holograms, **kwargs):
    holo, bg = holograms
    return reconstruct_hologram(
        holo,
        bg,
        dhm_dictionary=DHM,
        material_dictionary=MATERIAL,
        propagation_padding=True,
        **kwargs,
    )


# ------------------------------------------------------------- backend state
def test_default_backend_and_precision():
    assert oah.get_backend() == "numpy"
    assert oah.get_precision() == "double"
    assert backend.real_dtype() == np.float64
    assert backend.complex_dtype() == np.complex128


def test_use_backend_restores_state():
    with oah.use_backend(precision="single"):
        assert backend.real_dtype() == np.float32
        assert backend.complex_dtype() == np.complex64
    assert oah.get_precision() == "double"


def test_invalid_backend_and_precision():
    with pytest.raises(ValueError):
        oah.set_backend("torch")
    with pytest.raises(ValueError):
        oah.set_precision("half")


@pytest.mark.skipif(GPU, reason="only relevant without GPU")
def test_cupy_unavailable_raises():
    with pytest.raises(RuntimeError):
        oah.set_backend("cupy")
    assert oah.get_backend() == "numpy"


def test_to_numpy_passthrough():
    assert oah.to_numpy(None) is None
    assert oah.to_numpy(1.5) == 1.5
    a = np.arange(3)
    assert oah.to_numpy(a) is a


# ------------------------------------------------------- CPU reconstruction
def test_opencv_blur_matches_ndimage_filter():
    """The GPU path of locate_order replaces cv.GaussianBlur by the ndimage
    filter. Both must be equivalent (kernel radius 4 sigma, reflect-101)."""
    data = np.random.default_rng(0).random((256, 128))
    for sigma in (4, 16):
        ref = cv.GaussianBlur(data, None, sigma)
        res = backend.gaussian_filter(data, sigma=sigma, mode="mirror", truncate=4.0)
        np.testing.assert_allclose(res, ref, rtol=1e-6, atol=1e-12)


def test_reconstruction_cpu_double(holograms):
    proc = _reconstruct(holograms)
    assert proc.hologram.first_diffraction_order_pos == (-128, -128)
    assert proc.field_propagated.dtype == np.complex128
    assert isinstance(proc.phase_map, np.ndarray)
    # unwrapped bump (9 rad) is recovered
    phase = proc.phase_map - np.median(proc.phase_map)
    assert 8.5 < phase.max() < 9.5


def test_reconstruction_cpu_single_matches_double(holograms):
    ref = _reconstruct(holograms)
    res = _reconstruct(holograms, precision="single")
    assert res.field_propagated.dtype == np.complex64
    assert res.phase_map.dtype == np.float32
    assert res.hologram.first_diffraction_order_pos == (
        ref.hologram.first_diffraction_order_pos
    )
    np.testing.assert_allclose(res.phase_map, ref.phase_map, atol=1e-3)


def test_simulation_single_precision():
    with oah.use_backend(precision="single"):
        field = sim.planar(64, 0.1, 0.3)
        assert field.dtype == np.complex64
        assert sim.aperture(64, 10.0, 1.0).dtype == np.float32


# --------------------------------------------------------- phase unwrapping
def test_ls_dct_recovers_smooth_phase():
    truth = _bump(256, height=25.0, width=30.0)
    wrapped = np.angle(np.exp(1j * truth))
    res = phase_unwrapping_ls_dct(wrapped)
    np.testing.assert_allclose(res - res.mean(), truth - truth.mean(), atol=1e-9)


def test_ls_dct_congruent():
    truth = _bump(128, height=15.0, width=20.0)
    noisy = truth + 0.3 * np.random.default_rng(3).standard_normal(truth.shape)
    wrapped = np.angle(np.exp(1j * noisy))
    res = phase_unwrapping_ls_dct(wrapped, congruent=True)
    k = (res - wrapped) / (2 * math.pi)
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)


def test_ls_dct_agrees_with_fast2d_without_noise():
    truth = _bump(256, height=25.0, width=30.0)
    wrapped = np.angle(np.exp(1j * truth))
    fast = phase_unwrapping_fast2d(wrapped)
    ls = phase_unwrapping_ls_dct(wrapped)
    np.testing.assert_allclose(ls - ls.mean(), fast - fast.mean(), atol=1e-9)


def test_unwrapping_default_and_option(holograms):
    proc = _reconstruct(holograms)
    assert proc.reconstruction_dict["phaseUnwrappingMethod"] == "Fast 2D"
    proc_ls = _reconstruct(holograms, phase_unwrapping_method="Least Squares")
    assert proc_ls.phase_map.shape == proc.phase_map.shape
    with pytest.raises(ValueError):
        proc.phase_unwrapping(proc.phase_wrapped, method="unknown")


# ------------------------------------------------------- GPU parity (CuPy)
@requires_gpu
@pytest.mark.parametrize("padding", [False, True])
def test_gpu_angular_spectrum_parity(padding):
    rng = np.random.default_rng(5)
    field = np.exp(1j * rng.random((256, 256))) * (1 + 0.1 * rng.random((256, 256)))
    ref = angularSpectrum(field, 2e-6, 0.675e-6, 0.1e-6, 0.1e-6, padding=padding)
    with oah.use_backend("cupy"):
        res = angularSpectrum(field, 2e-6, 0.675e-6, 0.1e-6, 0.1e-6, padding=padding)
    assert backend.is_gpu(res)
    np.testing.assert_allclose(oah.to_numpy(res), ref, rtol=1e-10, atol=1e-12)


@requires_gpu
def test_gpu_reconstruction_parity_double(holograms):
    ref = _reconstruct(holograms)
    res = _reconstruct(holograms, backend="cupy")
    assert backend.is_gpu(res.phase_map)
    assert res.hologram.first_diffraction_order_pos == (
        ref.hologram.first_diffraction_order_pos
    )
    assert res.hologram.radius_mask == ref.hologram.radius_mask
    res.to_cpu()
    assert isinstance(res.phase_map, np.ndarray)
    assert isinstance(res.hologram.reconstructed_field, np.ndarray)
    np.testing.assert_allclose(res.field_propagated, ref.field_propagated, atol=1e-9)
    np.testing.assert_allclose(res.phase_map, ref.phase_map, atol=1e-8)
    np.testing.assert_allclose(res.height_profile, ref.height_profile, atol=1e-15)


@requires_gpu
def test_gpu_reconstruction_parity_single(holograms):
    ref = _reconstruct(holograms)
    res = _reconstruct(holograms, backend="cupy", precision="single", return_numpy=True)
    assert res.phase_map.dtype == np.float32
    np.testing.assert_allclose(res.phase_map, ref.phase_map, atol=1e-3)


@requires_gpu
def test_gpu_ls_dct_parity():
    truth = _bump(256, height=25.0, width=30.0)
    wrapped = np.angle(np.exp(1j * truth))
    ref = phase_unwrapping_ls_dct(wrapped)
    with oah.use_backend("cupy"):
        res = phase_unwrapping_ls_dct(backend.asarray(wrapped))
    np.testing.assert_allclose(oah.to_numpy(res), ref, atol=1e-9)


@requires_gpu
def test_gpu_simulation_parity():
    def simulate():
        n = 256
        wl = 0.675
        dhm = sim.HoloMicroscope(
            fieldShape=n,
            objectiveFocalLength=8250 / wl,
            pupilRadius=2970 / wl,
            tubeFocalLength=175000 / wl,
            sensorDistance=2000.0 / wl,
            referencePosition=n // 4,
        )
        outer = oah.to_numpy(sim.aperture(n, 20.0, 1.0))
        inner = oah.to_numpy(sim.aperture(n, 10.0, 1.0))
        obj = outer * np.exp(1j * inner)  # NumPy input, moved to the backend
        holo, _ = dhm.hologram(obj, 1.0)
        return holo

    ref = simulate()
    with oah.use_backend("cupy"):
        res = simulate()
    assert backend.is_gpu(res)
    np.testing.assert_allclose(oah.to_numpy(res), ref, rtol=1e-9, atol=1e-12)
