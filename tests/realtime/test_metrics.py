"""
Per-frame metrics (K2): saturation, quantiles, sideband energy and peak,
piston phase and phase noise; integration into the pipeline and the result.
"""

import ast
import inspect
import json
import math
import textwrap

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config, requires_gpu

from offaxisholo.realtime import (
    METRIC_NAMES,
    OffAxisPipeline,
    RealtimeConfig,
    ReconstructionResult,
)
from offaxisholo.realtime import metrics as metrics_module

SATURATION = 3.5  # synthetic frames reach about 4


def _pipeline(capacity=None, ref_border=8, piston=True, enabled=True, xp=None):
    config = make_config(
        demod={"crop": 64, "radius": 20},
        compensation={"ref_border": ref_border, "piston": piston},
        metrics={
            "enabled": enabled,
            "saturation_level": SATURATION,
            "black_level": 0.0,
            "capacity": capacity,
        },
    )
    pipe = OffAxisPipeline(config, xp=xp)
    pipe.set_background(SyntheticDHM().frames(4, obj=False))
    return pipe


def _metrics_of(frames, **kwargs):
    pipe = _pipeline(**kwargs)
    for frame in frames:
        pipe.push(frame)
    return [m for _, m in pipe.frame_metrics()]


# ------------------------------------------------------------ single metrics
def test_intensity_statistics_match_numpy(dhm):
    frame = dhm.frame()
    (m,) = _metrics_of([frame])
    assert m["saturated_count"] == np.count_nonzero(frame >= SATURATION)
    assert m["black_count"] == np.count_nonzero(frame <= 0.0)
    p05, median, p95 = np.quantile(frame, [0.05, 0.5, 0.95])
    assert m["intensity_p05_counts"] == pytest.approx(p05, rel=1e-12)
    assert m["intensity_median_counts"] == pytest.approx(median, rel=1e-12)
    assert m["intensity_p95_counts"] == pytest.approx(p95, rel=1e-12)
    assert isinstance(m["saturated_count"], int)


def test_saturation_is_counted(dhm):
    clean = dhm.frame()
    clipped = np.minimum(clean * 1.5, SATURATION)  # overexposed camera
    before, after = _metrics_of([clean, clipped])
    assert after["saturated_count"] > before["saturated_count"]
    assert after["saturated_count"] == np.count_nonzero(clipped >= SATURATION)
    assert after["intensity_p95_counts"] == pytest.approx(SATURATION)


def test_shifted_carrier_moves_the_sideband_peak():
    aligned = SyntheticDHM().frame()
    shifted = SyntheticDHM(carrier=(34, 31)).frame()  # order calibrated at 32, 32
    a, s = _metrics_of([aligned, shifted])
    assert (a["sideband_peak_dx_px"], a["sideband_peak_dy_px"]) == (0.0, 0.0)
    # -1st order at (-34, -31) against the calibrated (-32, -32)
    assert (s["sideband_peak_dx_px"], s["sideband_peak_dy_px"]) == (-2.0, 1.0)
    # a carrier that drifted against the background also shows as phase noise
    assert s["phase_noise_rad"] > 10 * a["phase_noise_rad"]


def test_reduced_fringe_contrast_lowers_the_sideband_fraction(dhm):
    clean = dhm.frame()
    incoherent = clean + 2.0  # incoherent light: only the DC term grows
    weak = SyntheticDHM(object_height=6.0).frame() * 0.5 + 1.0
    c, i, w = _metrics_of([clean, incoherent, weak])
    assert 0 < i["sideband_fraction"] < 0.5 * c["sideband_fraction"]
    assert w["sideband_fraction"] < c["sideband_fraction"]
    assert c["sideband_peak_fraction"] <= c["sideband_fraction"] <= 1.0


def test_piston_phase_against_the_background(dhm):
    m0, m1 = _metrics_of([dhm.frame(piston=0.0), dhm.frame(piston=0.5)])
    assert m1["piston_rad"] - m0["piston_rad"] == pytest.approx(0.5, abs=1e-3)


def test_piston_metric_without_piston_correction(dhm):
    m0, m1 = _metrics_of([dhm.frame(piston=0.0), dhm.frame(piston=0.5)], piston=False)
    assert m1["piston_rad"] - m0["piston_rad"] == pytest.approx(0.5, abs=1e-3)


def test_phase_noise_in_the_reference_region():
    noisy_model = SyntheticDHM(seed=1)
    clean, noisy = _metrics_of([noisy_model.frame(), noisy_model.frame(noise=0.3)])
    assert clean["phase_noise_rad"] < 0.01
    assert noisy["phase_noise_rad"] > 5 * clean["phase_noise_rad"]


def test_phase_noise_needs_a_reference_region(dhm):
    (m,) = _metrics_of([dhm.frame()], ref_border=None)
    assert math.isnan(m["phase_noise_rad"])


# --------------------------------------------------------------- pipeline
def test_metrics_do_not_change_the_result(dhm):
    frames = list(dhm.frames(5, piston_std=0.3, noise=0.05))
    fields = []
    for enabled in (False, True):
        pipe = _pipeline(enabled=enabled)
        for frame in frames:
            pipe.push(frame)
        fields.append(pipe.result().field)
    np.testing.assert_array_equal(fields[0], fields[1])


def test_disabled_metrics(dhm):
    pipe = _pipeline(enabled=False)
    pipe.push(dhm.frame())
    assert pipe.frame_metrics() == []
    res = ReconstructionResult.from_realtime(pipe.result(), pipe)
    assert res.frames == []
    assert res.timings_s["metrics"] == 0.0


def test_ring_buffer_keeps_the_last_frames(dhm):
    pipe = _pipeline(capacity=3)
    for piston in (0.0, 0.1, 0.2, 0.3, 0.4):
        pipe.push(dhm.frame(piston=piston))
    records = pipe.frame_metrics()
    assert [index for index, _ in records] == [2, 3, 4]
    pistons = [m["piston_rad"] for _, m in records]
    np.testing.assert_allclose(np.diff(pistons), 0.1, atol=1e-3)
    pipe.reset()
    assert pipe.frame_metrics() == []


def test_default_capacity_covers_the_block(dhm):
    pipe = _pipeline()
    assert pipe._metrics.capacity == max(
        pipe.config.averaging.n_obj, pipe.config.averaging.window
    )


def test_frame_records_in_the_result(dhm):
    pipe = _pipeline()
    for frame in dhm.frames(4, piston_std=0.2):
        pipe.push(frame)
    res = ReconstructionResult.from_realtime(pipe.result(), pipe)
    assert [r.index for r in res.frames] == [0, 1, 2, 3]
    assert all(r.used for r in res.frames)
    assert set(res.frames[0].metrics) == set(METRIC_NAMES)
    assert res.timings_s["metrics"] > 0

    header, buffers = res.to_message()
    text = json.dumps(header, allow_nan=False)
    same = ReconstructionResult.from_message(json.loads(text), buffers)
    assert same.frames[2].metrics == res.frames[2].metrics


def test_frame_records_given_by_the_caller_win(dhm):
    from offaxisholo.realtime import FrameRecord

    pipe = _pipeline()
    pipe.push(dhm.frame())
    own = [FrameRecord(index=0, used=False, reject_reason="manual")]
    res = ReconstructionResult.from_realtime(pipe.result(), pipe, frames=own)
    assert res.frames == own


def test_metrics_configuration_round_trip_and_validation():
    config = make_config(metrics={"enabled": True, "saturation_level": 4095.0})
    loaded = RealtimeConfig.from_json(config.to_json())
    assert loaded.metrics == config.metrics
    assert loaded.config_hash() == config.config_hash()
    with pytest.raises(ValueError, match="capacity"):
        make_config(metrics={"capacity": 0}).validate()
    with pytest.raises(ValueError, match="black_level"):
        make_config(metrics={"black_level": 10.0, "saturation_level": 5.0}).validate()


def test_metrics_disabled_by_default():
    assert make_config().metrics.enabled is False


# ------------------------------------------------------------- hot path
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


def test_metric_functions_have_no_host_synchronisation():
    for function in (
        metrics_module.intensity_stats,
        metrics_module.sideband_stats,
        metrics_module.piston_phase,
        metrics_module.reference_phase_noise,
        metrics_module.MetricsRecorder.record,
        OffAxisPipeline._record_metrics,
        OffAxisPipeline._demodulate_parts,
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
def test_cupy_parity_without_synchronisation():
    import cupy
    import cupyx

    frames = list(SyntheticDHM(seed=3).frames(4, piston_std=0.3, noise=0.05))
    cpu = _pipeline()
    gpu = _pipeline(xp=cupy)
    for frame in frames:
        cpu.push(frame)
    device_frames = [cupy.asarray(frame) for frame in frames]
    with cupyx.allow_synchronize(False):
        for frame in device_frames:
            gpu.push(frame)
    for (_, c), (_, g) in zip(cpu.frame_metrics(), gpu.frame_metrics(), strict=True):
        for name in METRIC_NAMES:
            assert g[name] == pytest.approx(c[name], rel=1e-6, abs=1e-9), name
