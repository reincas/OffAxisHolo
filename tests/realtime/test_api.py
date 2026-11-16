"""
Entry points of the product interface (K6): reconstruct_frames and
ReconstructionSession.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config, requires_gpu

from offaxisholo.realtime import (
    METRIC_NAMES,
    OffAxisPipeline,
    ReconstructionResult,
    ReconstructionSession,
    SelectionPolicy,
    reconstruct_frames,
)

Z = -2.6e-6


def _config(**overrides):
    parts = {
        "demod": {"crop": 64, "radius": 20},
        "compensation": {"ref_border": 8},
        "propagation": {"z": (Z,)},
        "metrics": {"saturation_level": 3.5},
    }
    for part, values in overrides.items():
        parts[part] = {**parts.get(part, {}), **values}
    return make_config(**parts)


def _background():
    return list(SyntheticDHM().frames(4, obj=False))


def _objects(count=5, seed=7):
    return list(SyntheticDHM(seed=seed).frames(count, piston_std=0.3, noise=0.02))


# ------------------------------------------------------- reconstruct_frames
def test_reconstruct_frames_matches_the_kernel():
    frames, background = _objects(), _background()
    res = reconstruct_frames(frames, background, _config())

    pipe = OffAxisPipeline(_config())
    pipe.set_background(background)
    for frame in frames:
        pipe.push(frame)
    kernel = pipe.result().to_numpy(("phase", "intensity"))

    assert isinstance(res, ReconstructionResult)
    np.testing.assert_allclose(res.phase_rad, kernel["phase"], atol=1e-6)
    np.testing.assert_allclose(res.intensity, kernel["intensity"], rtol=1e-6)
    assert res.z_um == pytest.approx(Z * 1e6)
    assert res.field is None


def test_defaults_metrics_on_and_policy_all():
    res = reconstruct_frames(_objects(5), _background(), _config())
    assert len(res.frames) == 5
    assert all(r.used for r in res.frames)
    assert set(res.frames[0].metrics) == set(METRIC_NAMES)
    assert res.quality["selection_policy"] == "all"
    assert res.quality["n_frames_used"] == 5
    assert res.config["metrics"]["enabled"] is True
    assert res.config["averaging"]["n_obj"] == 5
    assert res.config["averaging"]["mode"] == "block"


def test_metrics_can_be_switched_off():
    res = reconstruct_frames(_objects(3), _background(), _config(), metrics=False)
    assert res.frames == []
    assert res.config["metrics"]["enabled"] is False
    assert "selection_policy" not in res.quality


def test_frame_info_is_passed_through():
    infos = [{"frame_id": 100 + i, "exposure_s": 0.0168} for i in range(3)]
    res = reconstruct_frames(_objects(3), _background(), _config(), frame_info=infos)
    assert [r.acquisition for r in res.frames] == infos
    with pytest.raises(ValueError, match="frame_info"):
        reconstruct_frames(_objects(3), _background(), _config(), frame_info=infos[:2])


def test_frame_info_without_metrics():
    infos = [{"frame_id": i} for i in range(2)]
    res = reconstruct_frames(
        _objects(2), _background(), _config(), frame_info=infos, metrics=False
    )
    assert [(r.index, r.acquisition) for r in res.frames] == [
        (0, infos[0]),
        (1, infos[1]),
    ]


def test_outputs():
    res = reconstruct_frames(
        _objects(2), _background(), _config(), outputs=("phase", "intensity", "field")
    )
    assert res.field is not None and res.field.dtype == np.complex64
    with pytest.raises(ValueError, match="Unknown outputs"):
        reconstruct_frames(_objects(2), _background(), _config(), outputs=("height",))


def test_selection_through_the_entry_point():
    frames = _objects(5)
    frames.insert(2, SyntheticDHM(carrier=(34, 31)).frame())  # drifted carrier
    res = reconstruct_frames(
        frames, _background(), _config(), selection=SelectionPolicy("robust")
    )
    assert [r.used for r in res.frames] == [True, True, False, True, True, True]
    assert res.quality["n_obj"] == 5
    clean = reconstruct_frames(
        frames[:2] + frames[3:], _background(), _config(), selection=None
    )
    np.testing.assert_allclose(res.phase_rad, clean.phase_rad, atol=1e-6)


def test_iterables_and_empty_input():
    res = reconstruct_frames(iter(_objects(2)), iter(_background()), _config())
    assert res.quality["n_obj"] == 2
    with pytest.raises(ValueError, match="At least one"):
        reconstruct_frames([], _background(), _config())


def test_z_argument():
    res = reconstruct_frames(_objects(2), _background(), _config(), z=0.0)
    assert res.z_um == 0.0


# ------------------------------------------------------- ReconstructionSession
def test_session_life_cycle():
    session = ReconstructionSession(_config(averaging={"n_obj": 4}))
    session.warmup()
    quality = session.set_background(_background())
    assert quality["n_bg"] == 4

    for i, frame in enumerate(_objects(3)):
        session.push(frame, {"frame_id": i})
    first = session.result(timings_s={"grab": 0.1})
    assert session.n_frames == 3
    assert [r.acquisition["frame_id"] for r in first.frames] == [0, 1, 2]
    assert first.timings_s["grab"] == 0.1
    assert first.timings_s["demodulation"] > 0

    session.reset()
    assert session.n_frames == 0
    for frame in _objects(2, seed=11):
        session.push(frame)
    second = session.result()
    assert [r.index for r in second.frames] == [0, 1]
    assert all(r.acquisition == {} for r in second.frames)
    assert second.quality["n_obj"] == 2


def test_session_with_selection_needs_metrics_and_block_mode():
    with pytest.raises(ValueError, match="needs the frame metrics"):
        ReconstructionSession(
            _config(), selection=SelectionPolicy("robust"), metrics=False
        )
    with pytest.raises(ValueError, match="block"):
        ReconstructionSession(
            _config(averaging={"mode": "sliding"}), selection=SelectionPolicy("robust")
        )


def test_session_in_sliding_mode_with_policy_all():
    session = ReconstructionSession(_config(averaging={"mode": "sliding", "window": 3}))
    session.set_background(_background())
    for frame in _objects(5):
        session.push(frame, {"t": 1})
    res = session.result()
    # metrics and infos keep the last max(n_obj, window) frames
    assert res.quality["n_obj"] == 5
    assert res.frames[-1].index == 4


def test_result_message_from_the_session_is_json():
    session = ReconstructionSession(_config())
    session.set_background(_background())
    session.push(_objects(1)[0], {"frame_id": np.int64(3)})
    header, buffers = session.result().to_message()
    same = ReconstructionResult.from_message(json.loads(json.dumps(header)), buffers)
    assert same.frames[0].acquisition == {"frame_id": 3}


def test_service_needs_only_the_public_names():
    """The service / GUI import surface: entry points and configuration."""
    code = textwrap.dedent(
        """
        from offaxisholo.realtime import (
            DemodConfig, DHMConfig, RealtimeConfig, SelectionPolicy,
            ReconstructionSession, reconstruct_frames,
        )
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok"


@pytest.mark.gpu
@requires_gpu
def test_reconstruct_frames_on_the_gpu():
    import cupy

    frames, background = _objects(), _background()
    cpu = reconstruct_frames(frames, background, _config())
    gpu = reconstruct_frames(frames, background, _config(), xp=cupy)
    assert gpu.versions["cupy"] == cupy.__version__
    np.testing.assert_allclose(gpu.phase_rad, cpu.phase_rad, atol=1e-3)
