"""The product result against the shared contract ``ReconstructionResult`` 1.0 (K16).

The schema and example messages are vendored from DHMConnect
(``dhmconnect/schemas``, DL-A3) into ``tests/contracts/`` with
``python -m dhmconnect.schemas.export tests/contracts --only
ReconstructionResult,FrameRecord --commit <sha>``. Without ``dhmconnect`` the
structural checks below still run; the full JSON Schema validation runs when
``dhmconnect`` (or ``jsonschema``) is installed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config

from offaxisholo.realtime import ReconstructionResult, reconstruct_frames
from offaxisholo.realtime.result import SCHEMA_VERSION

CONTRACTS = Path(__file__).parents[1] / "contracts"


def _load(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


SCHEMA = _load("ReconstructionResult.schema.json")
FRAME_SCHEMA = _load("FrameRecord.schema.json")
COMMON = _load("common.schema.json")
EXAMPLE = _load("ReconstructionResult.example.json")


def _config(z=-2.6e-6):
    return make_config(
        demod={"crop": 64, "radius": 20},
        compensation={"ref_border": 8},
        propagation={"z": (z,)},
    )


@pytest.fixture(scope="module")
def header():
    frames = list(SyntheticDHM(seed=3).frames(4, piston_std=0.3, noise=0.02))
    background = list(SyntheticDHM().frames(4, obj=False))
    res = reconstruct_frames(
        frames,
        background,
        _config(),
        frame_info=[{"frame_id": i} for i in range(4)],
        parameter_set_id="zeiss63x-test",
    )
    head, _ = res.to_message()
    # The header must survive JSON text unchanged (NaN -> null).
    return json.loads(json.dumps(head, allow_nan=False))


def test_vendored_files_match_their_manifest():
    manifest = _load("MANIFEST.json")
    assert manifest["schema_versions"]["ReconstructionResult"] == SCHEMA_VERSION
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((CONTRACTS / name).read_bytes()).hexdigest() == digest


def test_schema_version_matches_the_contract(header):
    assert SCHEMA_VERSION == SCHEMA["x-schema-version"] == "1.0"
    assert header["schema_version"] == SCHEMA_VERSION
    assert header["message_type"] == "ReconstructionResult"


def test_header_has_every_required_field(header):
    assert set(SCHEMA["required"]) <= set(header)
    quality_required = SCHEMA["properties"]["quality"]["required"]
    assert set(quality_required) <= set(header["quality"])
    for record in header["frames"]:
        assert set(FRAME_SCHEMA["required"]) <= set(record)


def test_header_values_follow_the_contract(header):
    sources = COMMON["$defs"]["reference_source"]["enum"]
    assert header["quality"]["reference_source"] in sources
    # _config() sets compensation.ref_border: SA4 ladder -> border ring (K8)
    assert header["quality"]["reference_source"] == "border_ring"
    assert 0.0 < header["quality"]["reference_fraction"] < 1.0
    assert header["quality"]["phase_noise_rad"] > 0
    assert header["quality"]["sideband_snr"] > 1
    assert header["parameter_set_id"] == "zeiss63x-test"
    assert len(header["config_hash"]) == 64
    assert len(header["pixel_um"]) == 2 and header["propagation_index"] > 0
    names = [d["name"] for d in header["buffers"]]
    assert names == ["phase_rad", "intensity_rel"]
    assert "intensity" not in names and "n_medium" not in header
    assert "pixel_pitch_um" not in header


def test_full_schema_validation(header):
    try:
        from dhmconnect import schemas
    except ImportError:
        schemas = None
    if schemas is not None:
        schemas.validate(header, "ReconstructionResult")
        return
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    registry = referencing.Registry().with_resources(
        (s["$id"], referencing.Resource.from_contents(s))
        for s in (SCHEMA, FRAME_SCHEMA, COMMON)
    )
    jsonschema.Draft202012Validator(SCHEMA, registry=registry).validate(header)


def test_vendored_example_is_read(header):
    buffers = [
        np.zeros(d["shape"], dtype=np.dtype(d["dtype"]).newbyteorder("<"))
        for d in EXAMPLE["buffers"]
    ]
    res = ReconstructionResult.from_message(EXAMPLE, buffers)
    assert res.parameter_set_id == EXAMPLE["parameter_set_id"]
    assert res.pixel_um == tuple(EXAMPLE["pixel_um"])
    again, _ = res.to_message()
    for key in SCHEMA["required"]:
        if key not in ("created_utc",):
            assert again[key] == EXAMPLE[key], key


def test_default_parameter_set_id_names_the_config():
    res = reconstruct_frames(
        list(SyntheticDHM(seed=1).frames(2)),
        list(SyntheticDHM().frames(2, obj=False)),
        _config(),
    )
    assert res.parameter_set_id == f"config-{res.config_hash[:12]}"


def test_intensity_rel_is_one_on_an_empty_field():
    """I_object / I_background: an object stack without structure gives 1.0."""
    empty = list(SyntheticDHM(seed=5).frames(4, obj=False, noise=0.005))
    background = list(SyntheticDHM(seed=6).frames(4, obj=False, noise=0.005))
    res = reconstruct_frames(empty, background, _config(z=0.0))
    inner = res.intensity_rel[8:-8, 8:-8]
    assert np.median(inner) == pytest.approx(1.0, abs=0.02)
