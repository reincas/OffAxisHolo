"""
Stage timings of the realtime kernel (K5): StageTimer, pipeline integration,
pass-through of caller values, device synchronisation only with profile=True.
"""

import json
import types

import numpy as np
import pytest
from synthetic_dhm import make_config, requires_gpu

from offaxisholo.realtime import FrameRecord, OffAxisPipeline, ReconstructionResult
from offaxisholo.realtime.timing import STAGES, StageTimer


class FakeClock:
    """Clock that advances by a fixed step on every reading."""

    def __init__(self, step=1.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def _fake_cupy():
    """Namespace with the cuda.get_current_stream().synchronize() chain of CuPy."""
    calls = []
    stream = types.SimpleNamespace(synchronize=lambda: calls.append(1))
    cuda = types.SimpleNamespace(get_current_stream=lambda: stream)
    return types.SimpleNamespace(cuda=cuda), calls


def _measure(dhm, profile=False, piston=True, n_obj=5, xp=None):
    config = make_config(
        demod={"crop": 64, "radius": 20},
        propagation={"z": (-2.6e-6,)},
        compensation={"piston": piston},
    )
    pipe = OffAxisPipeline(config, xp=xp, profile=profile)
    pipe.set_background(dhm.frames(4, obj=False, noise=0.01))
    for frame in dhm.frames(n_obj, obj=True, piston_std=0.2, noise=0.01):
        pipe.push(frame)
    return pipe


# ------------------------------------------------------------- StageTimer
def test_stage_timer_with_fake_clock():
    timer = StageTimer(np, clock=FakeClock())
    assert timer.timings() == {**dict.fromkeys(STAGES, 0.0), "total": 0.0}
    t = timer.start()  # 1
    t = timer.lap("upload", t)  # 2
    t = timer.lap("demodulation", t)  # 3
    timer.lap("averaging", t)  # 4
    t = timer.start()  # 5 (second frame; total keeps the first start)
    timer.lap("upload", t)  # 6
    timings = timer.timings()
    assert timings["upload"] == 2.0
    assert timings["demodulation"] == 1.0
    assert timings["averaging"] == 1.0
    assert timings["piston"] == 0.0
    assert timings["total"] == 5.0  # 6 - 1, includes the gap between frames
    assert sum(timings[s] for s in STAGES) <= timings["total"]

    timer.reset()
    assert timer.timings()["total"] == 0.0
    assert timer.timings()["upload"] == 0.0


def test_stage_timer_rejects_unknown_stage():
    timer = StageTimer(np)
    with pytest.raises(KeyError):
        timer.lap("unknown", timer.start())


@pytest.mark.parametrize("profile, expected_syncs", [(False, 0), (True, 3)])
def test_device_synchronisation_only_with_profile(profile, expected_syncs):
    xp, calls = _fake_cupy()
    timer = StageTimer(xp, profile=profile, clock=FakeClock())
    t = timer.start()
    t = timer.lap("upload", t)
    timer.lap("demodulation", t)
    assert len(calls) == expected_syncs
    assert timer.profile is profile


def test_profile_has_no_effect_with_numpy():
    timer = StageTimer(np, profile=True, clock=FakeClock())
    timer.lap("upload", timer.start())
    assert timer.timings()["upload"] == 1.0


# ------------------------------------------------------ pipeline and result
def test_pipeline_fills_all_stages(dhm):
    pipe = _measure(dhm)
    res = ReconstructionResult.from_realtime(pipe.result(), pipe)
    timings = res.timings_s
    assert set(timings) == set(STAGES) | {"total"}
    for stage in ("demodulation", "piston", "averaging", "propagation", "download"):
        assert timings[stage] > 0, stage
    assert all(timings[s] >= 0 for s in STAGES)
    assert sum(timings[s] for s in STAGES) <= timings["total"]
    assert res.timings_profiled is False


def test_stage_sum_does_not_exceed_total_with_profile(dhm):
    pipe = _measure(dhm, profile=True)
    res = ReconstructionResult.from_realtime(pipe.result(), pipe)
    assert sum(res.timings_s[s] for s in STAGES) <= res.timings_s["total"]
    assert res.timings_profiled is True


def test_piston_stage_is_zero_without_piston_correction(dhm):
    pipe = _measure(dhm, piston=False)
    assert pipe.timer.timings()["piston"] == 0.0
    assert pipe.timer.timings()["demodulation"] > 0


def test_reset_and_new_background_clear_the_timings(dhm):
    pipe = _measure(dhm)
    assert pipe.timer.timings()["total"] > 0
    pipe.reset()
    assert pipe.timer.timings() == {**dict.fromkeys(STAGES, 0.0), "total": 0.0}
    pipe.push(dhm.frame())
    pipe.set_background(dhm.frames(2, obj=False))
    assert pipe.timer.timings()["upload"] == 0.0


def test_several_results_accumulate_download(dhm):
    pipe = _measure(dhm)
    first = ReconstructionResult.from_realtime(pipe.result(), pipe)
    second = ReconstructionResult.from_realtime(pipe.result(), pipe)
    assert second.timings_s["download"] > first.timings_s["download"]
    assert second.timings_s["total"] >= first.timings_s["total"]


def test_caller_values_pass_through_unchanged(dhm):
    pipe = _measure(dhm)
    acquisition = {
        "exposure_s": 0.0168,
        "t_camera_s": 1234.5678,
        "t_host_utc": "2026-11-16T10:00:00.000+00:00",
        "grab_s": 0.021,
        "frame_id": 17,
    }
    frames = [FrameRecord(index=0, acquisition=dict(acquisition))]
    res = ReconstructionResult.from_realtime(
        pipe.result(), pipe, frames=frames, timings_s={"grab": 0.27}
    )
    assert res.timings_s["grab"] == 0.27
    assert res.frames[0].acquisition == acquisition

    header, buffers = res.to_message()
    same = ReconstructionResult.from_message(json.loads(json.dumps(header)), buffers)
    assert same.frames[0].acquisition == acquisition
    assert same.timings_s == res.timings_s
    assert same.timings_profiled is False


@pytest.mark.gpu
@requires_gpu
def test_gpu_profile_timings(dhm):
    import cupy

    pipe = _measure(dhm, profile=True, xp=cupy)
    res = ReconstructionResult.from_realtime(pipe.result(), pipe)
    assert res.timings_profiled is True
    assert res.timings_s["demodulation"] > 0
    assert sum(res.timings_s[s] for s in STAGES) <= res.timings_s["total"]
