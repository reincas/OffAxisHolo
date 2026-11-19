"""
Frame selection (K3): from the per-frame metrics to a used / rejected mask.

PROCESSING, host side: :func:`select_frames` is a pure function of the metrics
table (``OffAxisPipeline.frame_metrics()``) and a :class:`SelectionPolicy`. It
runs once per measurement, after the last frame, never in the hot path.

Policies
--------
``all``        every frame is used (default until the N-study K13, decision O3).
``threshold``  fixed bounds per metric, e.g. ``{"saturated_count": (None, 100)}``.
               A NaN value of a bounded metric rejects the frame ("not
               available"), so a missing reference region is not silently
               ignored.
``robust``     outliers against the other frames of the same measurement:
               ``|x - median| > k * 1.4826 * MAD`` for each metric in
               ``robust_metrics``. NaN values and metrics with MAD = 0 are not
               tested.
``best_k``     the ``k`` frames with the best value of ``rank_by``
               (``ascending=True``: lower is better); NaN ranks last.

``min_frames`` is always enforced: when a policy leaves fewer frames, the best
rejected frames (by ``rank_by``, then by index) are used again. They keep
their reason, prefixed with ``"forced by min_frames: "``, and
:attr:`Selection.min_frames_enforced` is True.

Design note: averaging modes and memory
---------------------------------------
The kernel averages in ``push`` (hot path), before the selection is known.

* Block mode keeps the N demodulated, piston-corrected fields on the device
  until the selection (``OffAxisPipeline(..., keep_fields=True)``,
  ``FieldStore``), then rebuilds the average from the used frames
  (``apply_selection``). Memory: ``n_obj * n * n`` complex values, i.e.
  ``n_obj * n**2 * 8`` bytes in single and ``* 16`` in double precision.
  Example: 16 frames with a 512 x 512 crop need 32 MiB (single); without a crop
  (1024 x 1024) 128 MiB (single) or 256 MiB (double).
* Sliding and EMA modes have no end of measurement and cannot keep the fields.
  They can only use fixed thresholds decided per frame on the device (a 0/1
  weight per frame, no host decision): ``AveragingConfig.gate`` (K17), with
  the same bounds semantics as the ``threshold`` policy. ``apply_selection``
  refuses these modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

POLICIES = ("all", "threshold", "robust", "best_k")
_MAD_SCALE = 1.4826  # MAD -> standard deviation for normal data
FORCED_PREFIX = "forced by min_frames: "


@dataclass
class SelectionPolicy:
    """Parameters of the frame selection.

    Parameters
    ----------
    policy : str
        One of :data:`POLICIES`.
    thresholds : dict, optional
        ``threshold`` policy: metric name -> ``(lower, upper)``; ``None`` for an
        open bound.
    robust_metrics : tuple of str, optional
        ``robust`` policy: metrics tested for outliers.
    robust_k : float, optional
        ``robust`` policy: rejection limit in robust standard deviations.
    k : int or None, optional
        ``best_k`` policy: number of frames kept.
    rank_by : str, optional
        Metric that ranks frames for ``best_k`` and for ``min_frames``.
    ascending : bool, optional
        True if lower values of ``rank_by`` are better.
    min_frames : int, optional
        Minimum number of used frames, always enforced.
    """

    policy: str = "all"
    thresholds: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )
    robust_metrics: tuple[str, ...] = ("sideband_fraction", "phase_noise_rad")
    robust_k: float = 3.5
    k: int | None = None
    rank_by: str = "phase_noise_rad"
    ascending: bool = True
    min_frames: int = 1

    def validate(self) -> SelectionPolicy:
        """Check the parameters; raises ValueError with a clear message."""
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown policy '{self.policy}'. Possible: {POLICIES}.")
        if self.min_frames < 1:
            raise ValueError("min_frames must be >= 1.")
        if self.policy == "best_k" and (self.k is None or self.k < 1):
            raise ValueError("Policy 'best_k' needs k >= 1.")
        if self.policy == "threshold" and not self.thresholds:
            raise ValueError("Policy 'threshold' needs at least one threshold.")
        for name, bounds in self.thresholds.items():
            if len(bounds) != 2:
                raise ValueError(f"Threshold of {name} must be (lower, upper).")
        if self.robust_k <= 0:
            raise ValueError("robust_k must be positive.")
        return self

    def to_dict(self) -> dict:
        """JSON-able form (tuples as lists)."""
        return {
            "policy": self.policy,
            "thresholds": {k: list(v) for k, v in self.thresholds.items()},
            "robust_metrics": list(self.robust_metrics),
            "robust_k": self.robust_k,
            "k": self.k,
            "rank_by": self.rank_by,
            "ascending": self.ascending,
            "min_frames": self.min_frames,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SelectionPolicy:
        """Inverse of :meth:`to_dict`; unknown keys raise ValueError."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown keys for SelectionPolicy: {unknown}")
        values = dict(data)
        if "thresholds" in values:
            values["thresholds"] = {
                k: tuple(v) for k, v in values["thresholds"].items()
            }
        if "robust_metrics" in values:
            values["robust_metrics"] = tuple(values["robust_metrics"])
        return cls(**values)


@dataclass
class Selection:
    """Result of :func:`select_frames`.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        True for used frames, in the order of the metrics table.
    reasons : list of str or None
        Reject reason per frame; None for used frames, ``"forced by
        min_frames: …"`` for frames used only because of ``min_frames``.
    indices : list of int
        Frame index of every row (from the metrics table).
    policy : str
        The applied policy.
    min_frames_enforced : bool
        True if ``min_frames`` added frames back.
    min_frames : int
        The requested minimum; ``n_used`` is lower only when the measurement
        has fewer frames.
    """

    mask: np.ndarray
    reasons: list[str | None]
    indices: list[int]
    policy: str
    min_frames_enforced: bool = False
    min_frames: int = 1

    @property
    def n_used(self) -> int:
        """Number of used frames."""
        return int(np.count_nonzero(self.mask))

    def quality(self) -> dict:
        """Entries for ``ReconstructionResult.quality``."""
        return {
            "selection_policy": self.policy,
            "n_frames_total": len(self.indices),
            "n_frames_used": self.n_used,
            "selection_min_frames": self.min_frames,
            "selection_min_frames_enforced": self.min_frames_enforced,
        }


def _column(rows: list[dict], name: str) -> np.ndarray:
    try:
        return np.array([float(row[name]) for row in rows], dtype=float)
    except KeyError:
        raise ValueError(f"Unknown metric '{name}'.") from None


def _rank_order(values: np.ndarray, ascending: bool) -> np.ndarray:
    """Indices from best to worst; NaN last, ties by position (stable)."""
    key = values if ascending else -values
    key = np.where(np.isnan(key), np.inf, key)
    return np.argsort(key, kind="stable")


def _format(value: float) -> str:
    return f"{value:.6g}"


def threshold_reason(metrics: dict, thresholds: dict) -> str | None:
    """Reject reason of one frame against fixed thresholds, or None.

    Shared by the ``threshold`` policy and the frame gate of the sliding / EMA
    modes (K17), so both report the same reason for the same bounds.

    Parameters
    ----------
    metrics : dict
        Metrics of the frame (metric name -> value).
    thresholds : dict
        Metric name -> ``(lower, upper)``; None for an open bound. Tested in
        the order of the dict; the first failing bound gives the reason.

    Returns
    -------
    str or None
        ``"<name>: not available"`` for NaN, ``"<name> <value> < <lower>"`` or
        ``"<name> <value> > <upper>"``; None when all bounds hold.
    """
    for name, (lower, upper) in thresholds.items():
        value = float(metrics[name])
        if math.isnan(value):
            return f"{name}: not available"
        if lower is not None and value < lower:
            return f"{name} {_format(value)} < {_format(lower)}"
        if upper is not None and value > upper:
            return f"{name} {_format(value)} > {_format(upper)}"
    return None


def select_frames(records, policy: SelectionPolicy | None = None) -> Selection:
    """Select the frames of one measurement from their metrics.

    Parameters
    ----------
    records : list of (int, dict)
        ``OffAxisPipeline.frame_metrics()``: frame index and metrics, oldest
        first.
    policy : SelectionPolicy, optional
        Default: ``SelectionPolicy()`` (policy ``all``).

    Returns
    -------
    Selection

    Raises
    ------
    ValueError
        Invalid policy, unknown metric name, or no frames.
    """
    policy = (policy or SelectionPolicy()).validate()
    records = list(records)
    if not records:
        raise ValueError("No frames to select from.")
    indices = [int(index) for index, _ in records]
    rows = [metrics for _, metrics in records]
    count = len(rows)
    reasons: list[str | None] = [None] * count

    if policy.policy == "threshold":
        for name in policy.thresholds:
            _column(rows, name)  # unknown metric -> ValueError
        for i, row in enumerate(rows):
            reasons[i] = threshold_reason(row, policy.thresholds)

    elif policy.policy == "robust":
        for name in policy.robust_metrics:
            values = _column(rows, name)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            median = float(np.median(finite))
            mad = float(np.median(np.abs(finite - median)))
            if mad == 0.0:
                continue
            limit = policy.robust_k * _MAD_SCALE * mad
            for i, value in enumerate(values):
                if reasons[i] is None and np.isfinite(value):
                    if abs(value - median) > limit:
                        reasons[i] = (
                            f"{name} {_format(value)} outlier (median "
                            f"{_format(median)}, limit ±{_format(limit)})"
                        )

    elif policy.policy == "best_k":
        order = _rank_order(_column(rows, policy.rank_by), policy.ascending)
        for i in order[policy.k :]:
            reasons[i] = f"not among the best {policy.k} by {policy.rank_by}"

    mask = np.array([reason is None for reason in reasons], dtype=bool)
    enforced = False
    missing = min(policy.min_frames, count) - int(mask.sum())
    if missing > 0:
        order = _rank_order(_column(rows, policy.rank_by), policy.ascending)
        for i in order:
            if missing == 0:
                break
            if not mask[i]:
                mask[i] = True
                reasons[i] = FORCED_PREFIX + str(reasons[i])
                missing -= 1
        enforced = True

    return Selection(
        mask=mask,
        reasons=reasons,
        indices=indices,
        policy=policy.policy,
        min_frames_enforced=enforced,
        min_frames=policy.min_frames,
    )
