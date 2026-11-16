"""
ReconstructionResult and FrameRecord (K4): construction from the kernel,
message round trip, validation of headers and buffers.
"""

import json

import numpy as np
import pytest
from synthetic_dhm import PITCH, WAVELENGTH, make_config, requires_gpu

from offaxisholo.realtime import (
    FrameRecord,
    OffAxisPipeline,
    RealtimeConfig,
    ReconstructionResult,
)
from offaxisholo.realtime.result import SCHEMA_VERSION

Z = -2.6e-6


def _pipeline(dhm, crop=64, z=(Z,), xp=None):
    config = make_config(
        demod={"crop": crop, "radius": 20},
        propagation={"z": z},
        compensation={"piston": True},
    )
    pipe = OffAxisPipeline(config, xp=xp)
    pipe.set_background(dhm.frames(8, obj=False, piston_std=0.3, noise=0.01))
    for frame in dhm.frames(6, obj=True, piston_std=0.3, noise=0.01):
        pipe.push(frame)
    return pipe


def _records():
    return [
        FrameRecord(index=0, metrics={"saturated_count": 0, "piston_rad": 0.1}),
        FrameRecord(
            index=1,
            used=False,
            reject_reason="saturated",
            metrics={"saturated_count": np.int64(12), "piston_rad": float("nan")},
            acquisition={"exposure_s": 0.0168, "frame_id": 4711},
        ),
    ]


@pytest.fixture
def result(dhm):
    pipe = _pipeline(dhm)
    return ReconstructionResult.from_realtime(
        pipe.result(),
        pipe,
        include_field=True,
        frames=_records(),
        timings_s={"total": 0.01},
    )


def test_from_realtime_matches_kernel(dhm):
    pipe = _pipeline(dhm)
    kernel = pipe.result()
    res = ReconstructionResult.from_realtime(kernel, pipe, include_field=True)
    host = kernel.to_numpy(("phase", "intensity", "field"))

    assert res.shape == (64, 64)
    assert res.phase_rad.dtype == np.float32
    assert res.intensity.dtype == np.float32
    assert res.field.dtype == np.complex64
    # float32 storage of double results: tolerance of the cast
    np.testing.assert_allclose(res.phase_rad, host["phase"], atol=1e-6)
    np.testing.assert_allclose(res.intensity, host["intensity"], rtol=1e-6)
    np.testing.assert_allclose(res.field, host["field"], rtol=1e-6, atol=1e-7)

    # crop 128 -> 64 doubles the pitch; metres -> micrometres
    assert res.pixel_pitch_um == pytest.approx((2 * PITCH * 1e6, 2 * PITCH * 1e6))
    assert res.z_um == pytest.approx(Z * 1e6)
    assert res.wavelength_um == pytest.approx(WAVELENGTH * 1e6)
    assert res.n_medium == 1.0
    assert res.config_hash == pipe.config_hash
    assert RealtimeConfig.from_dict(res.config).config_hash() == pipe.config_hash
    assert res.quality["n_obj"] == 6
    assert res.quality["n_bg"] == 8
    assert res.quality["piston_std_rad"] > 0
    assert res.versions["offaxisholo"]
    assert res.versions["numpy"] == np.__version__
    assert res.created_utc.endswith("+00:00")
    assert res.schema_version == SCHEMA_VERSION


def test_field_is_optional(dhm):
    pipe = _pipeline(dhm)
    res = ReconstructionResult.from_realtime(pipe.result(), pipe)
    assert res.field is None
    header, buffers = res.to_message()
    assert [name for name, _ in buffers] == ["phase_rad", "intensity"]
    assert [d["name"] for d in header["buffers"]] == ["phase_rad", "intensity"]


def test_header_is_plain_json(result):
    header, buffers = result.to_message()
    text = json.dumps(header, allow_nan=False)  # raises on NaN / NumPy types
    assert json.loads(text) == header
    assert header["message_type"] == "ReconstructionResult"
    assert header["schema_version"] == SCHEMA_VERSION
    # NaN metric becomes null, NumPy integer becomes int
    metrics = header["frames"][1]["metrics"]
    assert metrics["piston_rad"] is None
    assert metrics["saturated_count"] == 12 and type(metrics["saturated_count"]) is int
    for descriptor, (name, array) in zip(header["buffers"], buffers, strict=True):
        assert descriptor["name"] == name
        assert descriptor["byte_order"] == "little"
        assert descriptor["shape"] == list(array.shape)
        assert array.flags.c_contiguous
        assert array.dtype.byteorder in ("<", "=")


def _assert_same(a, b):
    np.testing.assert_array_equal(a.phase_rad, b.phase_rad)
    np.testing.assert_array_equal(a.intensity, b.intensity)
    np.testing.assert_array_equal(a.field, b.field)
    for name in (
        "pixel_pitch_um",
        "z_um",
        "wavelength_um",
        "n_medium",
        "config_hash",
        "timings_s",
        "versions",
        "config",
        "created_utc",
        "schema_version",
    ):
        assert getattr(a, name) == getattr(b, name), name


def test_round_trip_through_json_and_bytes(result):
    """Simulates a network boundary: header as JSON text, buffers as raw bytes."""
    header, buffers = result.to_message()
    received_header = json.loads(json.dumps(header))
    received_buffers = [array.tobytes() for _, array in buffers]

    same = ReconstructionResult.from_message(received_header, received_buffers)

    _assert_same(result, same)
    assert same.frames[0] == result.frames[0]
    assert same.frames[1].used is False
    assert same.frames[1].reject_reason == "saturated"
    assert same.frames[1].metrics == {"saturated_count": 12, "piston_rad": None}
    assert same.frames[1].acquisition == {"exposure_s": 0.0168, "frame_id": 4711}
    assert same.n_frames_used == 1
    assert same.quality == result.quality
    # the received config rebuilds the same kernel configuration
    assert RealtimeConfig.from_dict(same.config).config_hash() == result.config_hash


def test_round_trip_with_named_arrays(result):
    header, buffers = result.to_message()
    _assert_same(result, ReconstructionResult.from_message(header, buffers))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda h, b: h.update(message_type="FrameRecord"), "Not a"),
        (lambda h, b: h.update(schema_version="1.0"), "Incompatible schema_version"),
        (lambda h, b: b.pop(), "buffers described"),
        (lambda h, b: h["buffers"][0].update(shape=[32, 32]), "values for shape"),
        (lambda h, b: h["buffers"][0].update(dtype="float64"), "dtype"),
        (lambda h, b: h["buffers"][0].update(byte_order="big"), "little-endian"),
    ],
)
def test_from_message_rejects_invalid(result, change, message):
    header, buffers = result.to_message()
    header = json.loads(json.dumps(header))
    raw = [array.tobytes() for _, array in buffers]
    change(header, raw)
    with pytest.raises(ValueError, match=message):
        ReconstructionResult.from_message(header, raw)


def test_from_message_rejects_swapped_named_buffers(result):
    header, buffers = result.to_message()
    with pytest.raises(ValueError, match="expected"):
        ReconstructionResult.from_message(header, [buffers[1], buffers[0], buffers[2]])


def test_minor_schema_versions_are_accepted(result):
    header, buffers = result.to_message()
    header["schema_version"] = "0.99"
    header["future_field"] = {"ignored": True}
    same = ReconstructionResult.from_message(header, buffers)
    assert same.schema_version == "0.99"


def test_shape_validation():
    phase = np.zeros((4, 4))
    with pytest.raises(ValueError, match="intensity shape"):
        ReconstructionResult(phase, np.zeros((4, 5)), (0.1, 0.1), 0, 0.675, 1, "h")
    with pytest.raises(ValueError, match="field shape"):
        ReconstructionResult(
            phase, phase, (0.1, 0.1), 0, 0.675, 1, "h", field=np.zeros((2, 2))
        )
    with pytest.raises(ValueError, match="2-D"):
        ReconstructionResult(np.zeros(4), np.zeros(4), (0.1, 0.1), 0, 0.675, 1, "h")


def test_results_for_several_z(dhm):
    pipe = _pipeline(dhm, z=(0.0, Z))
    plane, focus = (ReconstructionResult.from_realtime(r, pipe) for r in pipe.results())
    assert plane.z_um == 0.0
    assert focus.z_um == pytest.approx(Z * 1e6)
    assert plane.config_hash == focus.config_hash


@pytest.mark.gpu
@requires_gpu
def test_from_realtime_cupy_parity(dhm):
    import cupy

    cpu = _pipeline(dhm)
    gpu = _pipeline(type(dhm)(), xp=cupy)
    res_cpu = ReconstructionResult.from_realtime(cpu.result(), cpu)
    res_gpu = ReconstructionResult.from_realtime(gpu.result(), gpu)
    assert isinstance(res_gpu.phase_rad, np.ndarray)
    assert res_gpu.versions["cupy"] == cupy.__version__
    np.testing.assert_allclose(res_gpu.phase_rad, res_cpu.phase_rad, atol=1e-3)
