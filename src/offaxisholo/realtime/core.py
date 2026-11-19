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
from .accumulators import FieldStore, make_accumulator
from .background import BackgroundModel, utc_now
from .config import RealtimeConfig
from .metrics import (
    METRIC_NAMES,
    MetricsRecorder,
    intensity_stats,
    piston_phase,
    reference_phase_noise,
    sideband_snr,
    sideband_stats,
)
from .timing import StageTimer

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


def _noise_block(size, half_extent, order):
    """Noise block of the sideband SNR (K8): rows / cols slices or (None, None).

    A square of ``max(4, size // 32)`` spectrum pixels around the Nyquist
    corner ``(size / 2, size / 2)`` of the unshifted spectrum, the highest
    spatial frequencies, far from DC. Rejected (None) when it touches the box
    ``order ± half_extent`` of the sideband or of its conjugate.
    """
    b = max(4, size // 32)
    start = size // 2 - b // 2
    block = set(range(start, start + b))
    fy, fx = order
    reach = range(-int(half_extent), int(half_extent) + 1)
    for sign in (1, -1):
        rows = {int((sign * fy + o) % size) for o in reach}
        cols = {int((sign * fx + o) % size) for o in reach}
        if block & rows and block & cols:
            return None, None
    return slice(start, start + b), slice(start, start + b)


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

    Stage timings of the current measurement are in ``pipe.timer.timings()``
    (realtime.timing); ``profile=True`` synchronises the device for exact GPU
    stage times and must not be used in production runs.

    ``keep_fields=True`` (block mode only) keeps the corrected field of every
    frame until ``apply_selection(mask)`` rebuilds the average from the used
    frames (frame selection, realtime.selection); memory ``n_obj * n**2``
    complex values.
    """

    def __init__(
        self,
        config: RealtimeConfig,
        xp=None,
        profile: bool = False,
        keep_fields: bool = False,
    ):
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

        # sideband SNR (K8): effective sideband area and a noise block at the
        # Nyquist corner of the spectrum, free of both sidebands
        if d.mask == "circle":
            self._sideband_pixels = float((oy * oy + ox * ox <= radius * radius).sum())
            half_extent = min(math.ceil(radius), self.n // 2)
        else:
            self._sideband_pixels = float((window * window).sum())
            half_extent = self.n // 2
        self._noise_rows, self._noise_cols = _noise_block(
            self.size, half_extent, (fy, fx)
        )

        if d.edge_alpha > 0:
            edge_1d = tukey(self.size, d.edge_alpha)
            edge = np.outer(edge_1d, edge_1d)
            self._edge_window = xp.asarray(edge, dtype=self.rdtype)
        else:
            self._edge_window = None

        # reference region (SA4 ladder, K8): a-priori mask of the capture
        # (set_reference_mask) -> border ring (compensation.ref_border) ->
        # whole field
        border = config.compensation.ref_border
        if border is None:
            self._border_mask = None
        else:
            ring = np.zeros((self.n, self.n), dtype=bool)
            ring[:border, :] = ring[-border:, :] = True
            ring[:, :border] = ring[:, -border:] = True
            self._border_mask = ring
        self.set_reference_mask(None)

        self.crop_pitch = config.crop_pitch
        self._transfer_cache = {}
        self._acc = make_accumulator(xp, config.averaging)
        self.timer = StageTimer(xp, profile=profile)
        if keep_fields and config.averaging.mode != "block":
            raise ValueError(
                "keep_fields (frame selection) needs averaging mode 'block'; "
                "sliding and EMA can only gate frames by fixed thresholds."
            )
        self._fields = FieldStore(xp, config.averaging.n_obj) if keep_fields else None
        m = config.metrics
        if m.enabled:
            a = config.averaging
            capacity = m.capacity or max(a.n_obj, a.window)
            self._metrics = MetricsRecorder(xp, capacity)
            self._quantiles = xp.asarray([0.05, 0.5, 0.95])
            self._offsets = xp.asarray(offsets)
        else:
            self._metrics = None
        # frame gating of the sliding / EMA modes (K17): (column, lower, upper)
        # per metric; validated by RealtimeConfig (needs the metrics)
        column = {name: i for i, name in enumerate(METRIC_NAMES)}
        self._gate = (
            tuple(
                (column[name], lower, upper)
                for name, (lower, upper) in sorted(config.averaging.gate.items())
            )
            or None
        )
        self._piston_obj = _PhasorStats(xp)
        self._piston_bg = _PhasorStats(xp)
        self.Ub = None
        self.K = None
        self._bg_halves = None
        self.n_background = 0
        self._bg_created_utc = None  # creation time of the current background
        self._bg_model = None  # BackgroundModel cache (built on request)
        self._bg_loaded_quality = None  # quality of a loaded model

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

    def set_reference_mask(self, mask):
        """Use an a-priori reference mask for piston and noise (G10 SA1, K8).

        Call it before the first object frame of a measurement; it stays until
        it is replaced (``None`` returns to the border ring or the whole
        field). The background is not affected.

        Parameters
        ----------
        mask : array_like of bool or None
            True = structure-free pixel of the reconstruction grid (n, n), e.g.
            from :func:`offaxisholo.realtime.reference_mask.reference_mask`.

        Raises
        ------
        ValueError
            For a wrong shape or an empty mask.
        """
        if mask is None:
            if self._border_mask is None:
                self.reference_source = "whole_field"
                self.reference_region = None
                self._ref_weight = None
            else:
                self.reference_source = "border_ring"
                self.reference_region = self._border_mask
                self._ref_weight = self.xp.asarray(self._border_mask, dtype=self.rdtype)
            return
        region = np.asarray(mask, dtype=bool)
        if region.shape != (self.n, self.n):
            raise ValueError(
                f"Reference mask shape {region.shape} != ({self.n}, {self.n})."
            )
        if not region.any():
            raise ValueError("The reference mask holds no reference pixel.")
        self.reference_source = "mask_a_priori"
        self.reference_region = region
        self._ref_weight = self.xp.asarray(region, dtype=self.rdtype)

    def demodulate(self, frame):
        """Camera frame (N, N), any real dtype -> complex sideband field (n, n)."""
        return self._demodulate_parts(frame)[2]

    def _demodulate_parts(self, frame):
        """Demodulation with its intermediate results (for the frame metrics):
        windowed real frame (N, N), masked sideband crop (n, n), field (n, n),
        full spectrum (N, N)."""
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
        return h, crop, xp.fft.ifftn(crop, axes=(-2, -1)), spectrum

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

    def _transfer(self, shape, z, n_medium=None, cache=True):
        """Transfer function H(z) of the crop grid; ``n_medium`` defaults to
        ``propagation.n_medium``. Cached per (shape, z, n_medium) unless
        ``cache`` is False (focus sweeps)."""
        p = self.config.propagation
        n_medium = p.n_medium if n_medium is None else float(n_medium)
        key = (tuple(shape), float(z), n_medium)
        H = self._transfer_cache.get(key)
        if H is None:
            dx, dy = self.crop_pitch
            H = transfer_function(
                shape,
                dx,
                dy,
                self.config.dhm.wavelength,
                z,
                n_medium=n_medium,
                band_limit=p.band_limit,
                xp=self.xp,
                dtype=self.rdtype,
            )
            if cache:
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
        self._bg_created_utc = utc_now().isoformat(timespec="milliseconds")
        self._bg_model = None
        self._bg_loaded_quality = None
        self.reset()

    def background_model(self) -> BackgroundModel:
        """The current background as a reusable model (host copy, cached).

        Returns
        -------
        BackgroundModel
            With field, compensation factor, quality, background hash of the
            configuration, creation time and id. Synchronises once per
            background.
        """
        if self.Ub is None:
            raise RuntimeError("No background set.")
        if self._bg_model is None:
            self._bg_model = BackgroundModel(
                field=backend.to_numpy(self.Ub),
                compensation=backend.to_numpy(self.K),
                n_frames=self.n_background,
                config_hash=self.config.background_hash(),
                created_utc=self._bg_created_utc,
                quality=self.background_quality(),
                config=self.config.to_dict(),
            )
        return self._bg_model

    def use_background(self, model: BackgroundModel):
        """Use a stored background instead of recording a new one.

        Parameters
        ----------
        model : BackgroundModel
            Must fit the active configuration (``background_hash``).

        Raises
        ------
        ValueError
            The model was made with another demodulation, microscope,
            compensation or precision setting, or has another shape.
        """
        model.check_config(self.config)
        if model.field.shape != (self.n, self.n):
            raise ValueError(
                f"Background shape {model.field.shape} != crop ({self.n}, {self.n})."
            )
        xp = self.xp
        self.Ub = xp.asarray(model.field, dtype=self.cdtype)
        self.K = xp.asarray(model.compensation, dtype=self.cdtype)
        self.n_background = model.n_frames
        self._bg_halves = None
        self._piston_bg = _PhasorStats(xp)
        self._bg_created_utc = model.created_utc
        self._bg_model = model
        self._bg_loaded_quality = dict(model.quality)
        self.reset()

    def background_quality(self) -> dict:
        """Quality figures of the background (host values, synchronises)."""
        if self.Ub is None:
            raise RuntimeError("No background set.")
        if self._bg_loaded_quality is not None:
            return dict(self._bg_loaded_quality)
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
        """Start a new object average, timings and frame metrics (background
        is kept)."""
        self._acc.reset()
        self.timer.reset()
        if self._metrics is not None:
            self._metrics.reset()
        if self._fields is not None:
            self._fields.reset()
        self._piston_obj = _PhasorStats(self.xp)

    def push(self, frame):
        """Hot path: upload, demodulate, piston correct and accumulate one frame."""
        if self.K is None:
            raise RuntimeError("Background required before object frames.")
        if tuple(frame.shape) != (self.size, self.size):
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} != ({self.size}, {self.size})."
            )
        timer = self.timer
        t = timer.start()
        frame = self.xp.asarray(frame)
        t = timer.lap("upload", t)
        real_frame, crop, U, spectrum = self._demodulate_parts(frame)
        t = timer.lap("demodulation", t)
        raw = U
        unit = correction = None
        if self.config.compensation.piston:
            unit, correction = self._piston(U, self.Ub, self._ref_weight)
            U = U * correction
            self._piston_obj.add(unit)
            t = timer.lap("piston", t)
        values = weight = None
        if self._gate is not None:
            values = self._metric_values(
                frame, real_frame, crop, raw, unit, correction, spectrum
            )
            weight = self._gate_weight(values)
            t = timer.lap("metrics", t)
            self._acc.add(U, weight)
        else:
            self._acc.add(U)
        if self._fields is not None:
            self._fields.add(U)
        t = timer.lap("averaging", t)
        if self._metrics is not None:
            if values is None:
                values = self._metric_values(
                    frame, real_frame, crop, raw, unit, correction, spectrum
                )
            self._metrics.record(values, weight)
            timer.lap("metrics", t)

    def _gate_weight(self, values):
        """Hot path: 0/1 gate weight (0-d device array, real dtype) of a frame
        from the fixed thresholds of ``averaging.gate`` (K17). A NaN metric
        fails its bounds and rejects the frame."""
        xp = self.xp
        accepted = None
        for column, lower, upper in self._gate:
            value = xp.astype(xp.asarray(values[column]), xp.float64)
            ok = value == value  # False for NaN
            if lower is not None:
                ok = ok & (value >= lower)
            if upper is not None:
                ok = ok & (value <= upper)
            accepted = ok if accepted is None else accepted & ok
        return xp.astype(accepted, self.rdtype)

    def _metric_values(
        self, frame, real_frame, crop, field, unit, correction, spectrum
    ):
        """Hot path: per-frame metrics (K2) as a tuple of 0-d device values in
        the order of ``METRIC_NAMES``."""
        xp = self.xp
        m = self.config.metrics
        if unit is None:  # piston correction disabled: measure it anyway
            unit, correction = self._piston(field, self.Ub, self._ref_weight)
        saturated, black, p05, median, p95 = intensity_stats(
            xp, frame, m.saturation_level, m.black_level, self._quantiles
        )
        fraction, peak, dx, dy = sideband_stats(xp, real_frame, crop, self._offsets)
        snr = sideband_snr(
            xp,
            spectrum,
            crop,
            self._sideband_pixels,
            self._noise_rows,
            self._noise_cols,
        )
        noise = reference_phase_noise(xp, field, self.Ub, correction, self._ref_weight)
        return (
            saturated,
            black,
            p05,
            median,
            p95,
            fraction,
            peak,
            dx,
            dy,
            snr,
            piston_phase(xp, unit),
            noise,
        )

    @property
    def n_frames(self) -> int:
        return self._acc.count

    def apply_selection(self, mask):
        """Rebuild the block average from the used frames only.

        Parameters
        ----------
        mask : sequence of bool
            One entry per frame pushed since the last reset (for example
            ``Selection.mask``); True = used.

        Raises
        ------
        RuntimeError
            Without ``keep_fields=True``.
        ValueError
            Wrong length or no used frame.

        Notes
        -----
        The piston statistics keep all frames. Calling it again with another
        mask is allowed (the stored fields are not changed).
        """
        if self._fields is None:
            raise RuntimeError(
                "apply_selection needs OffAxisPipeline(keep_fields=True)."
            )
        mask = [bool(m) for m in mask]
        if len(mask) != self._fields.count:
            raise ValueError(
                f"Mask of {len(mask)} entries for {self._fields.count} frames."
            )
        if not any(mask):
            raise ValueError("The selection must keep at least one frame.")
        timer = self.timer
        t = timer.start()
        self._acc.reset()
        for index, used in enumerate(mask):
            if used:
                self._acc.add(self._fields[index])
        timer.lap("averaging", t)

    def frame_metrics(self) -> list[tuple[int, dict]]:
        """Metrics of the frames since the last reset, oldest first (host
        copy, synchronises). Empty when metrics are disabled."""
        if self._metrics is None:
            return []
        return self._metrics.records()

    @property
    def gate(self) -> dict:
        """The frame gate of the sliding / EMA modes (``averaging.gate``, K17);
        empty when frames are not gated."""
        return dict(self.config.averaging.gate)

    def frame_gate(self) -> list[tuple[int, bool]]:
        """Gate verdict of the frames since the last reset, aligned with
        :meth:`frame_metrics` (host copy, synchronises).

        Returns
        -------
        list of (int, bool)
            Frame index and whether the frame passed the gate (True for all
            frames when no gate is configured). Empty when metrics are
            disabled.
        """
        if self._metrics is None:
            return []
        return [(index, weight > 0.5) for index, weight in self._metrics.weights()]

    # ------------------------------------------------------------------- results
    def compensated_field(self):
        """mean(U) * K in the hologram plane (crop grid)."""
        if self.K is None:
            raise RuntimeError("Background required.")
        return self._acc.mean() * self.K

    def _propagate(self, E, z_list, n_medium, cache=True):
        """Yield ``(z, field_z)`` for every distance: one FFT of ``E`` shared by
        all distances, one transfer function each (z = 0 returns ``E``)."""
        xp = self.xp
        padding = self.config.propagation.padding
        spectrum = background = None
        if any(float(z) != 0.0 for z in z_list):
            source = E
            if padding:
                background = border_mean(E)
                source = pad_centered(E - background)
            spectrum = xp.fft.fftn(source, axes=(-2, -1))
        for z in z_list:
            z = float(z)
            if z == 0.0:
                yield z, E
                continue
            H = self._transfer(spectrum.shape, z, n_medium, cache=cache)
            field_z = xp.fft.ifftn(spectrum * H, axes=(-2, -1))
            if padding:
                field_z = crop_centered(field_z, E.shape)
                field_z = field_z + background * plane_wave_factor(
                    z, self.config.dhm.wavelength, n_medium
                )
            yield z, field_z

    def _propagation_index(self, propagation_index):
        """The index of a request, default ``propagation.n_medium``."""
        if propagation_index is None:
            return float(self.config.propagation.n_medium)
        n = float(propagation_index)
        if not (math.isfinite(n) and n > 0):
            raise ValueError(f"propagation_index must be positive, got {n}.")
        return n

    def results(self, z_list=None, propagation_index=None) -> list[RealtimeResult]:
        """
        Results for several propagation distances (default: config z). One FFT
        of the compensated field is shared by all distances; z = 0 is the
        hologram plane.

        Parameters
        ----------
        z_list : sequence of float, optional
            Distances in metres; default ``propagation.z``.
        propagation_index : float, optional
            Refractive index of the medium that belongs to these distances
            (G4: a z always comes with its index); default
            ``propagation.n_medium``. Stored in ``meta["propagation_index"]``.
        """
        z_list = self.config.propagation.z if z_list is None else tuple(z_list)
        if not z_list:
            z_list = (0.0,)
        n_medium = self._propagation_index(propagation_index)
        timer = self.timer
        t = timer.start()
        E = self.compensated_field()
        t = timer.lap("compensation", t)
        meta = dict(self._metadata(), propagation_index=n_medium)
        out = [
            RealtimeResult(field=field_z, z=z, meta=dict(meta, z=z))
            for z, field_z in self._propagate(E, z_list, n_medium)
        ]
        timer.lap("propagation", t)
        return out

    def result(self, z=None, propagation_index=None) -> RealtimeResult:
        """Result for one distance (default: the single configured z, or the
        hologram plane if none is configured); ``propagation_index`` as in
        :meth:`results`."""
        if z is None:
            configured = self.config.propagation.z
            if len(configured) > 1:
                raise ValueError("Several z configured; use results().")
            z = configured[0] if configured else 0.0
        return self.results((z,), propagation_index)[0]

    def focus_sweep(self, z_um, *, roi=None, propagation_index=None):
        """Sharpness of the compensated field over a list of distances (K10).

        One FFT of the compensated field, one (uncached) transfer function
        per distance; only one propagated field is alive at a time.

        Parameters
        ----------
        z_um : sequence of float
            Distances in µm (at least three for a refined minimum).
        roi : array_like of bool, optional
            Pixels of the crop grid (n, n) that enter the metric (True), e.g.
            the structure region; default the whole field.
        propagation_index : float, optional
            Index that belongs to the distances; default
            ``propagation.n_medium``.

        Returns
        -------
        offaxisholo.realtime.focus.FocusSweep
            Synchronises once per distance (one scalar each).
        """
        from .focus import FocusSweep, amplitude_cv

        n_medium = self._propagation_index(propagation_index)
        weight = None
        if roi is not None:
            region = np.asarray(roi, dtype=bool)
            if region.shape != (self.n, self.n):
                raise ValueError(f"ROI shape {region.shape} != ({self.n}, {self.n}).")
            weight = self.xp.asarray(region, dtype=self.rdtype)
        E = self.compensated_field()
        z_m = [float(z) * 1e-6 for z in z_um]
        values = [
            float(amplitude_cv(self.xp, field_z, weight))
            for _, field_z in self._propagate(E, z_m, n_medium, cache=False)
        ]
        return FocusSweep.from_values(z_um, values, propagation_index=n_medium)

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
