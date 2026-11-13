"""Building blocks of the angular spectrum method (shared offline/realtime)."""

import numpy as np

from offaxisholo import backend
from offaxisholo.reconstruction.numericalPropagation import (
    angularSpectrum,
    apply_transfer_function,
    border_mean,
    crop_centered,
    pad_centered,
    plane_wave_factor,
    transfer_function,
)

WL, DX = 0.675e-6, 0.1e-6


def _field(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.random((n, n))) * (1 + 0.1 * rng.random((n, n)))


def test_blocks_reproduce_angular_spectrum():
    field = _field()
    H = transfer_function(field.shape, DX, DX, WL, 2e-6)
    np.testing.assert_array_equal(
        apply_transfer_function(field, H), angularSpectrum(field, 2e-6, WL, DX, DX)
    )


def test_blocks_reproduce_padding_variant():
    field = _field()
    c = border_mean(field)
    padded = pad_centered(field - c)
    assert padded.shape == (128, 128)
    H = transfer_function(padded.shape, DX, DX, WL, -1.5e-6)
    res = crop_centered(apply_transfer_function(padded, H), field.shape)
    res = res + c * plane_wave_factor(-1.5e-6, WL)
    ref = angularSpectrum(field, -1.5e-6, WL, DX, DX, padding=True)
    np.testing.assert_array_equal(res, ref)


def test_forward_backward_is_identity_in_passband():
    field = _field()
    H = transfer_function(field.shape, DX, DX, WL, 3e-6, band_limit=False)
    Hb = transfer_function(field.shape, DX, DX, WL, -3e-6, band_limit=False)
    passband = np.abs(H) > 0
    np.testing.assert_allclose((H * Hb)[passband], 1.0, atol=1e-12)


def test_explicit_namespace_and_dtype():
    xp = backend.namespace("numpy")
    H = transfer_function((32, 32), DX, DX, WL, 1e-6, xp=xp, dtype=xp.float32)
    assert H.dtype == np.complex64
