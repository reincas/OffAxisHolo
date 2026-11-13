"""
Calibration of the first diffraction order (outside of the hot path).

The order position (fx, fy) is located once with the offline algorithm
(Hologram.locate_order) on one or several frames, usually background frames,
and written into the RealtimeConfig. It is never searched while streaming.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

from .. import backend
from .config import mask_radius


@dataclass
class OrderCalibration:
    order: tuple[int, int]  # most frequent position over the frames
    radius: float  # mask radius as Hologram._calc_radius_mask
    weight: float  # mean peak weight at ``order`` (blurred spectrum normalised
    # to its corner value, as returned by Hologram.locate_order; larger = clearer)
    positions: list = field(default_factory=list)  # per frame
    weights: list = field(default_factory=list)  # per frame

    @property
    def stable(self) -> bool:
        """True if the order was found at the same position in every frame."""
        return len(set(self.positions)) == 1


def calibrate_order(frames, dc_radius, *, size=16) -> OrderCalibration:
    """
    Locate the first diffraction order in every frame with
    Hologram.locate_order (smoothing size ``size``) and return the most frequent
    position, the mask radius and the peak weights. A warning is issued if the
    position is not identical in all frames or not found in some frames.
    """
    from ..reconstruction.hologram_class import Hologram

    # locate_order only needs the spectrum helpers of the instance; creating it
    # without __init__ avoids the automatic search and mask radius check
    locator = Hologram.__new__(Hologram)
    positions, weights = [], []
    n = None
    for frame in frames:
        frame = backend.to_numpy(frame)
        if frame.ndim != 2 or frame.shape[0] != frame.shape[1]:
            raise ValueError("Quadratic 2D frames required.")
        n = frame.shape[0]
        _, x, y, weight = locator.locate_order(holo=frame, size=size)
        if x is None:
            warnings.warn("First order not found in a calibration frame.", stacklevel=2)
            continue
        positions.append((int(x), int(y)))
        weights.append(float(weight))
    if not positions:
        raise RuntimeError("First diffraction order not found in any frame.")

    order, _ = Counter(positions).most_common(1)[0]
    calibration = OrderCalibration(
        order=order,
        radius=mask_radius(order, n, dc_radius),
        weight=sum(w for p, w in zip(positions, weights, strict=True) if p == order)
        / positions.count(order),
        positions=positions,
        weights=weights,
    )
    if not calibration.stable:
        warnings.warn(
            f"Order position differs between frames: {sorted(set(positions))}; "
            f"using {order}.",
            stacklevel=2,
        )
    return calibration
