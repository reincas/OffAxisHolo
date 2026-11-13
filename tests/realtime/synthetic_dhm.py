"""Synthetic off-axis holograms and configurations for the realtime tests."""

from dataclasses import replace

import numpy as np
import pytest

from offaxisholo.realtime import (
    AveragingConfig,
    CompensationConfig,
    DemodConfig,
    DHMConfig,
    PropagationConfig,
    RealtimeConfig,
)

WAVELENGTH = 0.675e-6
PITCH = 0.1e-6


def gpu_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


requires_gpu = pytest.mark.skipif(
    not gpu_available(), reason="CuPy with CUDA device required"
)


class SyntheticDHM:
    """
    Off-axis hologram |O + R|^2 with integer carrier (kx, ky) cycles per image,
    smooth aberration, optional object phase, global piston phase and additive
    camera noise. The object occupies the centre; the border is object free.
    """

    def __init__(self, n=128, carrier=(32, 32), seed=0, object_height=6.0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        y, x = np.indices((n, n), dtype=float)
        self.reference = np.exp(2j * np.pi * (carrier[0] * x + carrier[1] * y) / n)
        self.aberration = 3.0 * (((x - n / 2) / n) ** 2 + ((y - n / 2) / n) ** 2)
        r2 = (x - n / 2) ** 2 + (y - n / 2) ** 2
        self.object_phase = object_height * np.exp(-r2 / (2 * (n / 10) ** 2))
        self.envelope = 0.7 + 0.3 * np.exp(-r2 / (2 * (n / 1.5) ** 2))

    def frame(self, obj=True, piston=0.0, noise=0.0):
        phase = self.aberration + piston + (self.object_phase if obj else 0.0)
        holo = np.abs(self.envelope * np.exp(1j * phase) + self.reference) ** 2
        if noise:
            holo = holo + noise * self.rng.standard_normal(holo.shape)
        return np.clip(holo, 0, None)

    def frames(self, count, obj=True, piston_std=0.0, noise=0.0):
        for _ in range(count):
            piston = self.rng.normal(0.0, piston_std) if piston_std else 0.0
            yield self.frame(obj=obj, piston=piston, noise=noise)


def make_config(order=(-32, -32), size=128, precision="double", **overrides):
    """
    Realtime configuration for the synthetic model. ``overrides`` maps a part
    ('demod', 'compensation', 'averaging', 'propagation') to a dict of fields.
    """
    config = RealtimeConfig(
        dhm=DHMConfig(wavelength=WAVELENGTH, pixel_pitch=(PITCH, PITCH), dc_radius=10),
        demod=DemodConfig(size=size, order=order),
        averaging=AveragingConfig(),
        compensation=CompensationConfig(),
        propagation=PropagationConfig(),
        precision=precision,
    )
    for part, values in overrides.items():
        config = replace(config, **{part: replace(getattr(config, part), **values)})
    return config
