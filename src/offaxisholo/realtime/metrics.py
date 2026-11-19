"""
Per-frame metrics of the realtime kernel (K2).

PROCESSING: pure, backend-agnostic functions on device arrays plus a recorder
that keeps the values on the device until a result is built. Every function
here is hot-path safe: no host synchronisation (no ``float()``, ``.item()``,
``bool()`` of device values, no boolean indexing, no ``nonzero``), results are
0-d device arrays.

Metrics (column order of :data:`METRIC_NAMES`):

    saturated_count          pixels >= saturation_level
    black_count              pixels <= black_level
    intensity_p05_counts     5 % quantile of the raw frame (camera counts)
    intensity_median_counts  median of the raw frame
    intensity_p95_counts     95 % quantile of the raw frame
    sideband_fraction        energy of the masked sideband / total spectral
                             energy of the frame (Parseval, DC included); drops
                             with the fringe contrast, e.g. with incoherent
                             light that only raises the DC term
    sideband_peak_fraction   strongest sideband pixel / total spectral energy
    sideband_peak_dx_px      position of the strongest sideband pixel relative
    sideband_peak_dy_px      to the calibrated order (spectrum pixels); a
                             shifted carrier shows up here
    sideband_snr             mean spectral power per pixel of the masked
                             sideband / mean power per pixel of a noise block
                             at the Nyquist corner of the spectrum (linear,
                             K8); NaN when the block would touch a sideband
    piston_rad               global phase of the frame against the background
                             (reference region), before the piston correction
    phase_noise_rad          standard deviation of the piston-corrected phase
                             against the background in the reference region;
                             NaN without a reference region (a-priori mask
                             or ``compensation.ref_border``; SA4, K8)

The quantiles use ``xp.quantile`` (linear interpolation; one sort per frame).
A histogram would be cheaper for 8-bit frames but ``cupy.bincount`` reads the
maximum on the host; any optimisation waits for the K14 profile (C13).
"""

from __future__ import annotations

import math

METRIC_NAMES = (
    "saturated_count",
    "black_count",
    "intensity_p05_counts",
    "intensity_median_counts",
    "intensity_p95_counts",
    "sideband_fraction",
    "sideband_peak_fraction",
    "sideband_peak_dx_px",
    "sideband_peak_dy_px",
    "sideband_snr",
    "piston_rad",
    "phase_noise_rad",
)
_INTEGER_METRICS = {"saturated_count", "black_count"}
_TINY = 1e-300


def intensity_stats(xp, frame, saturation_level, black_level, quantiles):
    """Saturation, black pixels and intensity quantiles of a raw frame.

    Parameters
    ----------
    xp : module
        Array namespace.
    frame : array
        Raw camera frame on the device, shape (N, N), any real dtype.
    saturation_level, black_level : float
        Pixels ``>= saturation_level`` count as saturated, ``<= black_level``
        as black.
    quantiles : array
        Device array ``[0.05, 0.5, 0.95]`` (float64).

    Returns
    -------
    tuple of 0-d device arrays
        ``(saturated_count, black_count, p05, median, p95)``.
    """
    saturated = xp.count_nonzero(frame >= saturation_level)
    black = xp.count_nonzero(frame <= black_level)
    q = xp.quantile(xp.reshape(xp.astype(frame, xp.float64), (-1,)), quantiles)
    return saturated, black, q[0], q[1], q[2]


def sideband_stats(xp, real_frame, crop, offsets):
    """Sideband energy fraction and peak position from the demodulation FFT.

    Parameters
    ----------
    xp : module
        Array namespace.
    real_frame : array
        The (windowed) real frame that was transformed, shape (N, N).
    crop : array
        Masked sideband crop of its FFT, shape (n, n), IFFT order (offsets
        ``0, 1, ..., -1`` around the order).
    offsets : array
        Device array of the crop offsets in IFFT order, length n (int).

    Returns
    -------
    tuple of 0-d device arrays
        ``(sideband_fraction, sideband_peak_fraction, peak_dx_px, peak_dy_px)``.

    Notes
    -----
    By Parseval the total spectral energy of an unnormalised FFT is
    ``N_pixels * sum(h**2)``; no second FFT is needed.
    """
    h = xp.astype(real_frame, xp.float64)
    n_pixels = h.shape[-1] * h.shape[-2]
    total = n_pixels * xp.sum(h * h)
    power = xp.abs(crop) ** 2
    sideband = xp.sum(power, dtype=xp.float64)
    flat = xp.reshape(power, (-1,))
    peak_index = xp.argmax(flat)
    n = crop.shape[-1]
    peak_dx = offsets[peak_index % n]
    peak_dy = offsets[peak_index // n]
    return (
        sideband / (total + _TINY),
        xp.astype(flat[peak_index], xp.float64) / (total + _TINY),
        peak_dx,
        peak_dy,
    )


def sideband_snr(xp, spectrum, crop, sideband_pixels, noise_rows, noise_cols):
    """Sideband signal-to-noise ratio of one frame (K8).

    Parameters
    ----------
    xp : module
        Array namespace.
    spectrum : array
        Unnormalised FFT of the (windowed) frame, shape (N, N).
    crop : array
        Masked (or windowed) sideband crop of ``spectrum``, shape (n, n).
    sideband_pixels : float
        Effective number of sideband pixels (mask area, or the sum of the
        squared sideband window).
    noise_rows, noise_cols : slice or None
        Noise block of the spectrum (high spatial frequencies, free of both
        sidebands and DC); None returns NaN.

    Returns
    -------
    0-d device array
        Mean sideband power per pixel / mean noise power per pixel (linear).
    """
    if noise_rows is None:
        return xp.asarray(math.nan, dtype=xp.float64)
    signal = xp.sum(xp.abs(crop) ** 2, dtype=xp.float64) / sideband_pixels
    block = spectrum[noise_rows, noise_cols]
    noise = xp.mean(xp.astype(xp.abs(block) ** 2, xp.float64))
    return signal / (noise + _TINY)


def piston_phase(xp, unit_phasor):
    """Global phase in rad of the unit phasor ``exp(i arg p)`` (0-d device)."""
    return xp.angle(unit_phasor)


def reference_phase_noise(xp, field, background, correction, weight):
    """Phase noise of a piston-corrected field in the reference region.

    Parameters
    ----------
    xp : module
        Array namespace.
    field : array
        Demodulated field of the frame (before the piston correction).
    background : array
        Background field ``Ub``.
    correction : array or scalar
        Piston correction ``exp(-i arg p)`` (1 when the piston is not used).
    weight : array or None
        Reference region (1 inside, 0 outside). None returns NaN.

    Returns
    -------
    0-d device array
        Weighted standard deviation of ``arg(field * correction * conj(Ub))``
        in rad.
    """
    if weight is None:
        return xp.asarray(math.nan, dtype=xp.float64)
    phase = xp.astype(xp.angle(field * correction * xp.conj(background)), xp.float64)
    w = xp.astype(weight, xp.float64)
    count = xp.sum(w)
    mean = xp.sum(w * phase) / count
    variance = xp.sum(w * (phase - mean) ** 2) / count
    return xp.sqrt(variance)


class MetricsRecorder:
    """Device ring buffer of per-frame metrics.

    Parameters
    ----------
    xp : module
        Array namespace.
    capacity : int
        Number of frames kept; older frames are overwritten (ring buffer).

    Notes
    -----
    :meth:`record` writes one row with one small device copy per metric and
    never synchronises. :meth:`records` copies the table to the host once.
    """

    def __init__(self, xp, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1.")
        self.xp = xp
        self.capacity = int(capacity)
        self._table = xp.full((self.capacity, len(METRIC_NAMES)), math.nan)
        self._weights = xp.ones((self.capacity,))  # gate weights (K17)
        self.count = 0

    def reset(self):
        """Forget all frames (the buffer is reused, not reallocated)."""
        self._table[...] = math.nan
        self._weights[...] = 1.0
        self.count = 0

    def record(self, values, weight=None):
        """Store the metrics of the next frame.

        Parameters
        ----------
        values : sequence
            One 0-d device array (or Python number) per entry of
            :data:`METRIC_NAMES`, in that order.
        weight : 0-d array, optional
            Gate weight of the frame (K17); default 1 (accepted).
        """
        row = self.count % self.capacity
        for column, value in enumerate(values):
            self._table[row, column] = value
        self._weights[row] = 1.0 if weight is None else weight
        self.count += 1

    def weights(self) -> list[tuple[int, float]]:
        """Host copy of the gate weights, aligned with :meth:`records`.

        Returns
        -------
        list of (int, float)
            Frame index since the last reset and its gate weight (1.0 for
            frames recorded without a gate).
        """
        from .. import backend

        kept = min(self.count, self.capacity)
        weights = backend.to_numpy(self._weights)
        first = self.count - kept
        return [
            (index, float(weights[index % self.capacity]))
            for index in range(first, self.count)
        ]

    def records(self) -> list[tuple[int, dict]]:
        """Host copy of the kept frames, oldest first (synchronises).

        Returns
        -------
        list of (int, dict)
            Frame index since the last reset and its metrics; counts are
            ``int``, the other values ``float`` (NaN where not available).
        """
        from .. import backend

        kept = min(self.count, self.capacity)
        table = backend.to_numpy(self._table)
        first = self.count - kept
        out = []
        for index in range(first, self.count):
            row = table[index % self.capacity]
            metrics = {}
            for name, value in zip(METRIC_NAMES, row, strict=True):
                value = float(value)
                if name in _INTEGER_METRICS and math.isfinite(value):
                    value = int(value)
                metrics[name] = value
            out.append((index, metrics))
        return out
