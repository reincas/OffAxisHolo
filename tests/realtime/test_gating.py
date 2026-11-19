"""
Frame gating of the sliding and EMA modes (K17): fixed thresholds of the frame
metrics are turned into a 0/1 weight on the device; rejected frames do not
enter the average and are reported in the frame records.
"""

import ast
import inspect
import textwrap
from dataclasses import replace

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config, requires_gpu

from offaxisholo import backend
from offaxisholo.realtime import (
    EMAAccumulator,
    OffAxisPipeline,
    RealtimeConfig,
    ReconstructionSession,
    SlidingAccumulator,
)
from offaxisholo.realtime.accumulators import BlockAccumulator
from offaxisholo.realtime.result import GATE_PREFIX
from offaxisholo.realtime.selection import SelectionPolicy, threshold_reason

XP = backend.namespace("numpy")
# A constant offset (incoherent light) lowers the sideband energy fraction; the
# good synthetic frames have about 0.06, the bright ones below 0.01.
GATE = {"sideband_fraction": (0.03, None)}
OFFSET = 20.0


def _fields(count, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
        for _ in range(count)
    ]


def _w(value):
    return np.asarray(float(value))


# ------------------------------------------------------------ accumulators
def test_sliding_gated_mean_is_mean_of_accepted_in_window():
    fields = _fields(9, seed=1)
    weights = [1, 0, 1, 1, 0, 0, 1, 1, 0]
    acc = SlidingAccumulator(XP, window=4, renorm_every=3)
    for i, (f, w) in enumerate(zip(fields, weights, strict=True)):
        acc.add(f, _w(w))
        window = range(max(0, i - 3), i + 1)
        accepted = [fields[j] for j in window if weights[j]]
        assert acc.n_effective == len(accepted)
        if accepted:
            np.testing.assert_allclose(acc.mean(), np.mean(accepted, axis=0))
        else:
            with pytest.raises(RuntimeError, match="gate"):
                acc.mean()
    assert acc.gated
    acc.reset()
    assert acc.n_effective == 0
    acc.add(fields[0], _w(1))
    np.testing.assert_allclose(acc.mean(), fields[0])


def test_sliding_window_of_rejected_frames_raises():
    acc = SlidingAccumulator(XP, window=2)
    for f in _fields(3):
        acc.add(f, _w(0))
    with pytest.raises(RuntimeError, match="passed the gate"):
        acc.mean()


@pytest.mark.parametrize("weights", [[1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]])
def test_ema_gated_equals_ungated_on_accepted_frames(weights):
    fields = _fields(len(weights), seed=2)
    gated = EMAAccumulator(XP, alpha=0.3)
    plain = EMAAccumulator(XP, alpha=0.3)
    for f, w in zip(fields, weights, strict=True):
        gated.add(f, _w(w))
        if w:
            plain.add(f)
    np.testing.assert_allclose(gated.mean(), plain.mean(), rtol=1e-12)
    assert gated.n_effective == plain.n_effective == min(sum(weights), 1.7 / 0.3)


def test_ema_without_accepted_frame_raises_and_reset_restarts():
    acc = EMAAccumulator(XP, alpha=0.5)
    fields = _fields(3)
    acc.add(fields[0], _w(0))
    with pytest.raises(RuntimeError, match="gate"):
        acc.mean()
    acc.add(fields[1], _w(1))
    np.testing.assert_allclose(acc.mean(), fields[1])
    acc.reset()
    acc.add(fields[2], _w(1))  # first accepted frame after reset is copied
    np.testing.assert_allclose(acc.mean(), fields[2])
    assert acc.n_effective == 1


# ---------------------------------------------------------------- pipeline
def _config(mode, gate=GATE, **averaging):
    config = make_config(
        demod={"crop": 64, "radius": 20},
        compensation={"ref_border": 8},
        metrics={"enabled": True, "saturation_level": 1e9},
    )
    return replace(
        config,
        averaging=replace(config.averaging, mode=mode, gate=dict(gate), **averaging),
    )


def _frames():
    """Good frames with piston and noise; frames 1 and 4 are bright (bad)."""
    good = list(SyntheticDHM(seed=5).frames(6, piston_std=0.3, noise=0.05))
    frames = list(good)
    frames[1] = frames[1] + OFFSET
    frames[4] = frames[4] + OFFSET
    return frames, [0, 2, 3, 5]


def _background():
    return list(SyntheticDHM(seed=6).frames(4, obj=False, noise=0.05))


def _run(config, frames):
    pipe = OffAxisPipeline(config)
    pipe.set_background(_background())
    for frame in frames:
        pipe.push(frame)
    return pipe


def test_gated_sliding_equals_block_average_of_accepted_frames():
    frames, good = _frames()
    gated = _run(_config("sliding", window=len(frames)), frames)
    block = _run(_config("block", gate={}), [frames[i] for i in good])
    np.testing.assert_allclose(
        gated.compensated_field(), block.compensated_field(), rtol=1e-10, atol=1e-12
    )
    assert gated.result().meta["n_effective"] == len(good)
    verdict = dict(gated.frame_gate())
    assert [i for i, ok in verdict.items() if ok] == good


def test_gated_ema_equals_ema_of_accepted_frames():
    frames, good = _frames()
    gated = _run(_config("ema", ema_alpha=0.25), frames)
    plain = _run(_config("ema", gate={}, ema_alpha=0.25), [frames[i] for i in good])
    np.testing.assert_allclose(
        gated.compensated_field(), plain.compensated_field(), rtol=1e-10, atol=1e-12
    )
    assert gated.result().meta["n_effective"] == pytest.approx(len(good))


def test_ungated_pipeline_reports_all_frames_passed():
    frames, _ = _frames()
    pipe = _run(_config("sliding", gate={}), frames[:3])
    assert pipe.gate == {}
    assert pipe.frame_gate() == [(0, True), (1, True), (2, True)]


def test_session_reports_gated_frames_in_records_and_quality():
    frames, good = _frames()
    session = ReconstructionSession(_config("sliding", window=len(frames)))
    session.set_background(_background())
    for index, frame in enumerate(frames):
        session.push(frame, {"frame_id": index})
    result = session.result()
    used = [record.index for record in result.frames if record.used]
    assert used == good
    for record in result.frames:
        assert record.acquisition == {"frame_id": record.index}
        if record.used:
            assert record.reject_reason is None
        else:
            assert record.reject_reason.startswith(GATE_PREFIX + "sideband_fraction ")
            assert record.reject_reason.endswith("< 0.03")
    q = result.quality
    assert q["n_frames_total"] == len(frames)
    assert q["n_frames_used"] == len(good)
    assert q["n_frames_gated"] == len(frames) - len(good)
    assert q["n_effective"] == len(good)
    assert result.config["averaging"]["gate"] == {"sideband_fraction": [0.03, None]}
    # the message round trip keeps the records
    restored = type(result).from_message(*result.to_message())
    assert [r.used for r in restored.frames] == [r.used for r in result.frames]


def test_nan_metric_rejects_the_frame():
    # without a reference region the phase noise is NaN -> gate rejects all
    config = make_config(
        demod={"crop": 64, "radius": 20},
        metrics={"enabled": True, "saturation_level": 1e9},
    )
    config = replace(
        config,
        averaging=replace(
            config.averaging, mode="ema", gate={"phase_noise_rad": (None, 1.0)}
        ),
    )
    pipe = _run(config, list(SyntheticDHM().frames(2)))
    assert [ok for _, ok in pipe.frame_gate()] == [False, False]
    with pytest.raises(RuntimeError, match="gate"):
        pipe.compensated_field()


def test_gate_round_trips_through_json():
    config = _config("ema")
    loaded = RealtimeConfig.from_json(config.to_json())
    assert loaded.config_hash() == config.config_hash()
    assert OffAxisPipeline(loaded).gate == {"sideband_fraction": (0.03, None)}


@pytest.mark.parametrize(
    ("mode", "gate", "metrics", "message"),
    [
        ("block", GATE, True, "sliding and EMA"),
        ("ema", GATE, False, "metrics.enabled"),
        ("ema", {"unknown": (0, 1)}, True, "Unknown gate metric"),
        ("ema", {"piston_rad": (None, None)}, True, "no bound"),
        ("ema", {"piston_rad": (1.0, 0.0)}, True, "lower"),
        ("ema", {"piston_rad": (1.0,)}, True, "lower, upper"),
    ],
)
def test_invalid_gates(mode, gate, metrics, message):
    config = _config(mode, gate=gate)
    config = replace(config, metrics=replace(config.metrics, enabled=metrics))
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_threshold_reason_matches_threshold_policy():
    from offaxisholo.realtime.selection import select_frames

    rows = [
        {"a": 1.0, "b": 5.0},
        {"a": float("nan"), "b": 5.0},
        {"a": 3.0, "b": 9.0},
    ]
    bounds = {"a": (None, 2.0), "b": (None, 8.0)}
    selection = select_frames(
        list(enumerate(rows)), SelectionPolicy("threshold", thresholds=bounds)
    )
    assert selection.reasons == [threshold_reason(row, bounds) for row in rows]
    assert selection.reasons == [None, "a: not available", "a 3 > 2"]


# ---------------------------------------------------------------- hot path
FORBIDDEN_CALLS = {
    "float",
    "int",
    "bool",
    "complex",
    "to_numpy",
    "item",
    "get",
    "nonzero",
    "tolist",
}


def test_gating_hot_path_has_no_host_synchronisation():
    for function in (
        OffAxisPipeline.push,
        OffAxisPipeline._gate_weight,
        OffAxisPipeline._metric_values,
        SlidingAccumulator.add,
        EMAAccumulator.add,
        BlockAccumulator.add,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                assert name not in FORBIDDEN_CALLS, f"{function.__name__} calls {name}"


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", ["sliding", "ema"])
def test_cupy_gating_without_synchronisation(mode):
    import cupy
    import cupyx

    frames, good = _frames()
    config = _config(mode, window=len(frames))
    cpu = _run(config, frames)
    gpu = OffAxisPipeline(replace(config, backend="cupy"))
    gpu.set_background(cupy.asarray(f) for f in _background())
    device_frames = [cupy.asarray(f) for f in frames]
    gpu.push(device_frames[0])  # warm-up: FFT plans, buffers
    gpu.reset()
    with cupyx.allow_synchronize(False):
        for frame in device_frames:
            gpu.push(frame)
    assert [ok for _, ok in gpu.frame_gate()] == [i in good for i in range(6)]
    np.testing.assert_allclose(
        backend.to_numpy(gpu.compensated_field()),
        cpu.compensated_field(),
        rtol=1e-9,
        atol=1e-12,
    )
