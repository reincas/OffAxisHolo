"""
Realtime off-axis DHM kernel: demodulation, piston correction, averaging,
background compensation and angular spectrum propagation.

Pure computation: no logging, no file IO, no plots, no network. The array
namespace is passed explicitly (NumPy on the CPU, CuPy on the GPU); the global
state of offaxisholo.backend is never read (thread rule).

Processing (see docs/realtime/anweisung_realtime.txt, section 3):

    per frame (hot path, push):
        U = IFFT_n( crop_n( FFT_N(I) around (fx, fy) ) * mask )    demodulation
        U = U * conj(p) / |p|,  p = sum_M U * conj(Ub)            piston phase
        accumulate U                                              block/sliding/ema
    once per background (set_background):
        Ub = mean of the piston aligned background fields
        K  = conj(Ub) / (|Ub|^2 + eps)
    once per result (result / results):
        E   = mean(U) * K                                         compensation
        E_z = IFFT( FFT(E) * H(z) )                               propagation

All steps after the demodulation are linear, therefore averaging before the
compensation and propagation is exact and these steps run once per result
instead of once per frame. The crop to n x n changes the pixel pitch of the
result to dx' = dx * N / n (config.crop_pitch).

Parity mode (crop=None, mask='circle', edge_alpha=0, eps_rel='machine',
piston=False, precision='double') reproduces the offline HologramProcessor
(spatial filtering -> compensation -> propagation) up to round-off.

Hot path rules (push): no host synchronisation (.item(), bool(), float() of
device values, nonzero, boolean indexing), all tables precomputed, no logging.

Reference skeleton: docs/realtime/reference/holo_gpu_pipeline.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import backend
from ..reconstruction import fieldops
from ..reconstruction.numericalPropagation import (
    border_mean,
    crop_centered,
    pad_centered,
    plane_wave_factor,
    transfer_function,
)
from .accumulators import make_accumulator
from .config import RealtimeConfig

_TINY = 1e-300  # avoids 0/0 in the unit phasor (complex128 reduction)


def tukey(n, alpha):
    """Tukey (tapered cosine) window of length n (scipy.signal.windows.tukey)."""
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        alpha = 1.0
    x = np.arange(n) / (n - 1)
    w = np.ones(n)
    left = x < alpha / 2
    right = x >= 1 - alpha / 2
    w[left] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[left] - alpha / 2)))
    w[right] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[right] - 1 + alpha / 2)))
    return w


class _PhasorStats:
    """Device accumulator of unit phasors p/|p| -> circular standard deviation."""

    def __init__(self, xp):
        self.xp = xp
        self.sum = xp.zeros((), dtype=xp.complex128)
        self.count = 0

    def add(self, unit_phasor):
        self.sum += unit_phasor
        self.count += 1

    def circular_std(self):
        """Circular standard deviation in rad (host value, synchronises)."""
        if self.count == 0:
            return None
        r = abs(complex(backend.to_numpy(self.sum))) / self.count
        return math.sqrt(-2.0 * math.log(min(max(r, 1e-300), 1.0)))


@dataclass
class RealtimeResult:
    """Compensated (and propagated) complex field with derived outputs."""

    field: object  # complex array (device)
    z: float
    meta: dict = field(default_factory=dict)

    @property
    def phase(self):
        """Wrapped phase in rad."""
        return fieldops.phase(self.field)

    @property
    def amplitude(self):
        """|E| (relative amplitude transmission, as compensated)."""
        xp = backend.get_namespace(self.field)
        return xp.abs(self.field)

    def intensity(self, mode="linear"):
        """|E|^2 ('linear') or 10*log10(|E|^2 + 1e-12) ('db') as offline."""
        return fieldops.intensity(self.field, mode=mode)

    def to_numpy(self, outputs=("phase", "amplitude")) -> dict:
        """Host copies of the requested outputs ('phase', 'amplitude',
        'intensity', 'intensity_db', 'field')."""
        getters = {
            "phase": lambda: self.phase,
            "amplitude": lambda: self.amplitude,
            "intensity": lambda: self.intensity("linear"),
            "intensity_db": lambda: self.intensity("db"),
            "field": lambda: self.field,
        }
        return {name: backend.to_numpy(getters[name]()) for name in outputs}


class OffAxisPipeline:
    """
    Realtime reconstruction kernel.

        pipe = OffAxisPipeline(config)          # config with calibrated order
        pipe.set_background(background_frames)
        for frame in object_frames:
            pipe.push(frame)
        res = pipe.result(z=-2.6e-6)            # RealtimeResult
        phase = res.to_numpy()["phase"]
        pipe.reset()                            # next measurement
    """

    def __init__(self, config: RealtimeConfig, xp=None):
        self.config = config.validate()
        self.config_hash = config.config_hash()
        self.xp = xp if xp is not None else backend.namespace(config.backend)
        xp = self.xp
        self.rdtype = backend.real_dtype(xp, config.precision)
        self.cdtype = backend.complex_dtype(xp, config.precision)

        d = config.demod
        self.size, self.n = d.size, d.n
        fx, fy = d.order
        # crop indices directly in IFFT order (0, 1, ..., -1): no fftshift/roll
        offsets = np.fft.ifftshift(np.arange(-(self.n // 2), self.n // 2))
        self._rows = xp.asarray((fy + offsets) % self.size)
        self._cols = xp.asarray((fx + offsets) % self.size)

        if d.mask == "circle":
            oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
            radius = config.radius
            self._mask = xp.asarray(oy * oy + ox * ox <= radius * radius)
            self._sideband_window = None
        else:
            sideband = tukey(self.n, d.sideband_alpha)
            window = np.fft.ifftshift(np.outer(sideband, sideband))
            self._mask = None
            self._sideband_window = xp.asarray(window, dtype=self.rdtype)

        if d.edge_alpha > 0:
            edge_1d = tukey(self.size, d.edge_alpha)
            edge = np.outer(edge_1d, edge_1d)
            self._edge_window = xp.asarray(edge, dtype=self.rdtype)
        else:
            self._edge_window = None

        border = config.compensation.ref_border
        if border is None:
            self._ref_weight = None  # whole field
        else:
            weight = np.zeros((self.n, self.n))
            weight[:border, :] = weight[-border:, :] = 1
            weight[:, :border] = weight[:, -border:] = 1
            self._ref_weight = xp.asarray(weight, dtype=self.rdtype)

        self.crop_pitch = config.crop_pitch
        self._transfer_cache = {}
        self._acc = make_accumulator(xp, config.averaging)
        self._piston_obj = _PhasorStats(xp)
        self._piston_bg = _PhasorStats(xp)
        self.Ub = None
        self.K = None
        self._bg_halves = None
        self.n_background = 0

    # ------------------------------------------------------------ building blocks
    def validate_frame(self, frame):
        """Host-side checks of a camera frame (shape, positivity). Not used in
        the hot path; GPU frames are only checked for their shape."""
        if tuple(frame.shape) != (self.size, self.size):
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} != ({self.size}, {self.size})."
            )
        if not backend.is_gpu(frame):
            frame = np.asarray(frame)
            if frame.min() < 0 or not np.all(np.isfinite(frame)):
                raise ValueError("Real, finite, positive hologram image required.")

    def demodulate(self, frame):
        """Camera frame (N, N), any real dtype -> complex sideband field (n, n)."""
        xp = self.xp
        h = xp.astype(xp.asarray(frame), self.rdtype)
        if self._edge_window is not None:
            h = h * self._edge_window
        spectrum = xp.fft.fftn(h, axes=(-2, -1))
        crop = xp.take(xp.take(spectrum, self._rows, axis=0), self._cols, axis=1)
        if self._mask is not None:
            crop = xp.where(self._mask, crop, 0)
        else:
            crop = crop * self._sideband_window
        return xp.fft.ifftn(crop, axes=(-2, -1))

    def _piston(self, field, reference, weight):
        """Unit phasor exp(-i arg p) of p = sum(field * conj(reference) * w),
        reduced in complex128, as 0-d device arrays (no synchronisation)."""
        xp = self.xp
        product = field * xp.conj(reference)
        if weight is not None:
            product = product * weight
        p = xp.sum(product, dtype=xp.complex128)
        unit = p / (xp.abs(p) + _TINY)
        return unit, xp.astype(xp.conj(unit), self.cdtype)

    def _transfer(self, shape, z):
        key = (tuple(shape), float(z))
        H = self._transfer_cache.get(key)
        if H is None:
            p = self.config.propagation
            dx, dy = self.crop_pitch
            H = transfer_function(
                shape,
                dx,
                dy,
                self.config.dhm.wavelength,
                z,
                n_medium=p.n_medium,
                band_limit=p.band_limit,
                xp=self.xp,
                dtype=self.rdtype,
            )
            self._transfer_cache[key] = H
        return H

    # ---------------------------------------------------------------- background
    def set_background(self, frames):
        """
        Background from an iterable of frames (streamed, not stored): every
        field is piston aligned to the first one (if enabled), then averaged.
        Computes K = conj(Ub) / (|Ub|^2 + eps) once. Resets the object average.
        """
        xp = self.xp
        piston = self.config.compensation.piston
        self._piston_bg = _PhasorStats(xp)
        total = reference = halves = None
        count = 0
        for frame in frames:
            if count == 0:
                self.validate_frame(frame)
            U = self.demodulate(frame)
            if reference is None:
                reference = U
                total = xp.zeros_like(U)
                halves = [xp.zeros_like(U), xp.zeros_like(U)]
            elif piston:
                unit, correction = self._piston(U, reference, None)
                U = U * correction
                self._piston_bg.add(unit)
            total += U
            halves[count % 2] += U
            count += 1
        if count == 0:
            raise ValueError("At least one background frame is required.")

        Ub = total / count
        intensity = xp.abs(Ub) ** 2
        eps_rel = self.config.compensation.eps_rel
        if eps_rel == "machine":
            eps_rel = xp.finfo(intensity.dtype).eps
        eps = eps_rel * xp.max(intensity)
        self.Ub = Ub
        self.K = xp.conj(Ub) / (intensity + eps)
        self._bg_halves = halves if count >= 2 else None
        self.n_background = count
        self.reset()

    def background_quality(self) -> dict:
        """Quality figures of the background (host values, synchronises)."""
        if self.Ub is None:
            raise RuntimeError("No background set.")
        xp = self.xp
        amplitude = xp.abs(self.Ub)
        a_max = float(xp.max(amplitude))
        quality = {
            "n_bg": self.n_background,
            "piston_std_rad": self._piston_bg.circular_std(),
            "amplitude_min": float(xp.min(amplitude)),
            "amplitude_max": a_max,
            "dark_fraction": float(
                xp.mean(xp.astype(amplitude < 0.05 * a_max, xp.float64))
            ),
            "phase_noise_estimate_rad": None,
        }
        if self._bg_halves is not None:
            # split half: phase difference of the two half means, sigma_full ~
            # std(diff) / 2 (pixels with sufficient amplitude only)
            even, odd = self._bg_halves
            diff = fieldops.phase(even * xp.conj(odd))
            valid = backend.to_numpy(amplitude > 0.2 * a_max)
            quality["phase_noise_estimate_rad"] = float(
                np.std(backend.to_numpy(diff)[valid]) / 2
            )
        return quality

    # ------------------------------------------------------------- object frames
    def reset(self):
        """Start a new object average (background is kept)."""
        self._acc.reset()
        self._piston_obj = _PhasorStats(self.xp)

    def push(self, frame):
        """Hot path: demodulate, piston correct and accumulate one frame."""
        if self.K is None:
            raise RuntimeError("Background required before object frames.")
        if tuple(frame.shape) != (self.size, self.size):
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} != ({self.size}, {self.size})."
            )
        U = self.demodulate(frame)
        if self.config.compensation.piston:
            unit, correction = self._piston(U, self.Ub, self._ref_weight)
            U = U * correction
            self._piston_obj.add(unit)
        self._acc.add(U)

    @property
    def n_frames(self) -> int:
        return self._acc.count

    # ------------------------------------------------------------------- results
    def compensated_field(self):
        """mean(U) * K in the hologram plane (crop grid)."""
        if self.K is None:
            raise RuntimeError("Background required.")
        return self._acc.mean() * self.K

    def results(self, z_list=None) -> list[RealtimeResult]:
        """
        Results for several propagation distances (default: config z). One FFT
        of the compensated field is shared by all distances; z = 0 is the
        hologram plane.
        """
        xp = self.xp
        z_list = self.config.propagation.z if z_list is None else tuple(z_list)
        if not z_list:
            z_list = (0.0,)
        E = self.compensated_field()
        padding = self.config.propagation.padding

        spectrum = background = None
        if any(float(z) != 0.0 for z in z_list):
            source = E
            if padding:
                background = border_mean(E)
                source = pad_centered(E - background)
            spectrum = xp.fft.fftn(source, axes=(-2, -1))

        meta = self._metadata()
        out = []
        for z in z_list:
            z = float(z)
            if z == 0.0:
                field_z = E
            else:
                field_z = xp.fft.ifftn(
                    spectrum * self._transfer(spectrum.shape, z), axes=(-2, -1)
                )
                if padding:
                    field_z = crop_centered(field_z, E.shape)
                    field_z = field_z + background * plane_wave_factor(
                        z,
                        self.config.dhm.wavelength,
                        self.config.propagation.n_medium,
                    )
            out.append(RealtimeResult(field=field_z, z=z, meta=dict(meta, z=z)))
        return out

    def result(self, z=None) -> RealtimeResult:
        """Result for one distance (default: the single configured z, or the
        hologram plane if none is configured)."""
        if z is None:
            configured = self.config.propagation.z
            if len(configured) > 1:
                raise ValueError("Several z configured; use results().")
            z = configured[0] if configured else 0.0
        return self.results((z,))[0]

    def _metadata(self) -> dict:
        c = self.config
        return {
            "n_obj": self._acc.count,
            "n_effective": self._acc.n_effective,
            "n_bg": self.n_background,
            "mode": c.averaging.mode,
            "n": self.n,
            "size": self.size,
            "crop_pitch": self.crop_pitch,
            "precision": c.precision,
            "backend": c.backend,
            "config_hash": self.config_hash,
            "piston_stats": self._piston_obj,  # use piston_std() for the value
        }

    def piston_std(self):
        """Circular std (rad) of the object piston phases since reset (sync)."""
        return self._piston_obj.circular_std()

    def warmup(self):
        """Create FFT plans and transfer functions before the first frame."""
        self.demodulate(np.zeros((self.size, self.size), dtype=np.uint16))
        shape = (
            (2 * self.n, 2 * self.n)
            if self.config.propagation.padding
            else (
                self.n,
                self.n,
            )
        )
        for z in self.config.propagation.z:
            if float(z) != 0.0:
                self._transfer(shape, z)
