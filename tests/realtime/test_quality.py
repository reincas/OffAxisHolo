"""Quality numbers of a result (K8): SA4 reference ladder, noise, sideband SNR."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config

from offaxisholo.realtime import OffAxisPipeline, reconstruct_frames
from offaxisholo.realtime.core import _noise_block
from offaxisholo.realtime.quality import (
    median_metric,
    reference_phase_noise_after_plane,
    reference_quality,
)
from offaxisholo.realtime.result import FrameRecord

CROP = 64


def _config(ref_border=None):
    return make_config(
        demod={"crop": CROP, "radius": 20},
        compensation={"ref_border": ref_border},
    )


def _object_free_mask(radius_px=22):
    """Structure-free region of the synthetic object (Gaussian, sigma 6.4 px)."""
    y, x = np.indices((CROP, CROP))
    return (y - CROP / 2) ** 2 + (x - CROP / 2) ** 2 > radius_px**2


def _reconstruct(noise=0.5, **kwargs):
    dhm = SyntheticDHM(seed=11)
    frames = list(dhm.frames(6, piston_std=0.3, noise=noise))
    background = list(SyntheticDHM(seed=12).frames(6, obj=False, noise=noise))
    return reconstruct_frames(
        frames, background, _config(kwargs.pop("ref_border", None)), **kwargs
    )


# --------------------------------------------------------------- functions
def test_noise_after_plane_recovers_known_noise():
    rng = np.random.default_rng(0)
    y, x = np.indices((64, 64))
    sigma = 0.05
    phase = 3.1 + 0.02 * y - 0.03 * x + rng.normal(0, sigma, (64, 64))
    phase = np.angle(np.exp(1j * phase))  # wrapped around +-pi
    region = np.ones((64, 64), bool)
    region[20:40, 20:40] = False
    noise = reference_phase_noise_after_plane(phase, region)
    # 3600 samples: relative standard error of the std ~ 1.2 %
    assert noise == pytest.approx(sigma, rel=0.05)
    assert reference_phase_noise_after_plane(phase, np.zeros((64, 64), bool)) is None
    with pytest.raises(ValueError, match="shape"):
        reference_phase_noise_after_plane(phase, region[:10])


def test_reference_quality_ladder_and_warning():
    phase = np.zeros((32, 32))
    q, w = reference_quality(phase, None, "whole_field", min_reference_fraction=0.1)
    assert q == {
        "reference_source": "whole_field",
        "reference_fraction": 0.0,
        "phase_noise_rad": None,
    }
    assert w and "below min_reference_fraction" in w[0]
    region = np.zeros((32, 32), bool)
    region[:4] = True
    q, w = reference_quality(phase, region, "mask_a_priori", min_reference_fraction=0.1)
    assert q["reference_source"] == "mask_a_priori"
    assert q["reference_fraction"] == pytest.approx(0.125)
    assert q["phase_noise_rad"] == pytest.approx(0.0)
    assert w == []


def test_median_metric_uses_used_finite_frames_only():
    frames = [
        FrameRecord(index=0, metrics={"sideband_snr": 10.0}),
        FrameRecord(index=1, metrics={"sideband_snr": float("nan")}),
        FrameRecord(index=2, metrics={"sideband_snr": 30.0}),
        FrameRecord(index=3, metrics={"sideband_snr": 99.0}, used=False),
    ]
    assert median_metric(frames, "sideband_snr") == 20.0
    assert median_metric([], "sideband_snr") is None


def test_noise_block_avoids_the_sidebands():
    rows, cols = _noise_block(1024, 64, (-300, 200))
    assert rows == slice(496, 528) and cols == slice(496, 528)
    # carrier close to Nyquist: the block would touch a sideband
    assert _noise_block(1024, 64, (-500, 500)) == (None, None)


# --------------------------------------------------------------- results
def test_a_priori_mask_gives_the_reference_numbers():
    mask = _object_free_mask()
    res = _reconstruct(reference_mask=mask, min_reference_fraction=0.9)
    q = res.quality
    assert q["reference_source"] == "mask_a_priori"
    assert q["reference_fraction"] == pytest.approx(mask.mean())
    assert 0 < q["phase_noise_rad"] < 0.5
    assert q["sideband_snr"] > 1
    assert any("below min_reference_fraction" in w for w in res.warnings)
    # per-frame noise uses the same region
    assert all(np.isfinite(r.metrics["phase_noise_rad"]) for r in res.frames)


def test_noise_and_snr_follow_the_camera_noise():
    mask = _object_free_mask()
    low = _reconstruct(noise=0.2, reference_mask=mask).quality
    high = _reconstruct(noise=2.0, reference_mask=mask).quality
    assert high["phase_noise_rad"] > 2 * low["phase_noise_rad"]
    assert high["sideband_snr"] < low["sideband_snr"] / 2


def test_border_ring_and_whole_field_fallbacks():
    ring = _reconstruct(ref_border=6).quality
    assert ring["reference_source"] == "border_ring"
    assert 0 < ring["reference_fraction"] < 1 and ring["phase_noise_rad"] > 0
    whole = _reconstruct().quality
    assert whole["reference_source"] == "whole_field"
    assert whole["reference_fraction"] == 0.0 and whole["phase_noise_rad"] is None


def test_mask_validation():
    pipe = OffAxisPipeline(_config())
    with pytest.raises(ValueError, match="shape"):
        pipe.set_reference_mask(np.ones((10, 10), bool))
    with pytest.raises(ValueError, match="no reference pixel"):
        pipe.set_reference_mask(np.zeros((CROP, CROP), bool))
    pipe.set_reference_mask(_object_free_mask())
    assert pipe.reference_source == "mask_a_priori"
    pipe.set_reference_mask(None)
    assert pipe.reference_source == "whole_field"


def test_sideband_snr_has_no_host_synchronisation():
    from offaxisholo.realtime import metrics

    tree = ast.parse(textwrap.dedent(inspect.getsource(metrics.sideband_snr)))
    names = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not names & {"float", "int", "bool", "item", "get", "to_numpy", "nonzero"}
