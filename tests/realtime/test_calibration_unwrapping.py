"""Order calibration (C3) and unwrapping helpers (C10)."""

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM

from offaxisholo.realtime import TemporalUnwrapper, calibrate_order, unwrap


def test_calibration_matches_offline_hologram(dhm):
    from offaxisholo import Hologram

    frames = list(dhm.frames(3, obj=False, noise=0.05))
    cal = calibrate_order(frames, dc_radius=10)
    holo = Hologram(frames[0], {"name": "x", "dcRadius": 10})
    assert cal.order == tuple(holo.first_diffraction_order_pos)
    assert cal.radius == holo.radius_mask
    assert cal.stable
    assert cal.weight > 1  # peak clearly above the spectrum corner


def test_calibration_warns_if_unstable():
    a = SyntheticDHM(carrier=(32, 32)).frame(obj=False)
    b = SyntheticDHM(carrier=(30, 33)).frame(obj=False)
    with pytest.warns(UserWarning, match="differs"):
        cal = calibrate_order([a, a, b], dc_radius=10)
    assert not cal.stable
    assert cal.order == calibrate_order([a], dc_radius=10).order


def test_unwrap_dispatch():
    y, x = np.indices((64, 64), dtype=float)
    truth = 20 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 10.0**2))
    wrapped = np.angle(np.exp(1j * truth))
    assert unwrap(wrapped, "none") is wrapped
    for method in ("fast2d", "least_squares"):
        res = unwrap(wrapped, method)
        np.testing.assert_allclose(res - res.mean(), truth - truth.mean(), atol=1e-8)
    with pytest.raises(ValueError):
        unwrap(wrapped, "kamui")


def test_temporal_unwrapper_follows_growing_phase():
    tracker = TemporalUnwrapper()
    phase = None
    for step in range(40):
        phase = tracker.update(np.full((4, 4), np.exp(1j * 0.5 * step)))
    np.testing.assert_allclose(phase, 0.5 * 39, atol=1e-12)  # far beyond pi
    tracker.reset()
    assert tracker.phase is None
