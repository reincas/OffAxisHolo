"""
Background model (K7): creation, save and load, validity against the
configuration, age warnings (O6) and use in the entry points.
"""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config

from offaxisholo.io.background import load_background, save_background
from offaxisholo.realtime import (
    BackgroundModel,
    OffAxisPipeline,
    ReconstructionSession,
    reconstruct_frames,
)
from offaxisholo.realtime.background import utc_now


def _config(**overrides):
    parts = {"demod": {"crop": 64, "radius": 20}, "propagation": {"z": (-2.6e-6,)}}
    parts.update(overrides)
    return make_config(**parts)


def _pipeline(config=None):
    pipe = OffAxisPipeline(config or _config())
    pipe.set_background(SyntheticDHM(seed=2).frames(6, obj=False, noise=0.02))
    return pipe


def _objects():
    return list(SyntheticDHM(seed=4).frames(3, piston_std=0.3, noise=0.02))


def test_model_from_the_pipeline():
    pipe = _pipeline()
    model = pipe.background_model()
    np.testing.assert_array_equal(model.field, pipe.Ub)
    np.testing.assert_array_equal(model.compensation, pipe.K)
    assert model.n_frames == 6
    assert model.config_hash == pipe.config.background_hash()
    assert len(model.model_id) == 16
    assert model.quality == pipe.background_quality()
    assert model.age_s() >= 0
    assert pipe.background_model() is model  # cached until the next background
    pipe.set_background(SyntheticDHM().frames(2, obj=False))
    assert pipe.background_model() is not model


def test_save_and_load_round_trip(tmp_path):
    model = _pipeline().background_model()
    path = save_background(model, tmp_path / "bg")
    assert path.suffix == ".npz"
    loaded = load_background(path)
    np.testing.assert_array_equal(loaded.field, model.field)
    np.testing.assert_array_equal(loaded.compensation, model.compensation)
    for name in ("n_frames", "config_hash", "created_utc", "model_id", "config"):
        assert getattr(loaded, name) == getattr(model, name), name
    assert loaded.quality == model.quality


def test_loaded_background_gives_the_same_result(tmp_path):
    first = _pipeline()
    path = save_background(first.background_model(), tmp_path / "bg.npz")

    second = OffAxisPipeline(_config())
    second.use_background(load_background(path))
    for frame in _objects():
        first.push(frame)
        second.push(frame)
    np.testing.assert_array_equal(first.result().field, second.result().field)
    assert second.background_quality() == first.background_model().quality
    assert second.n_background == 6


def test_validity_against_the_configuration():
    config = _config()
    model = _pipeline(config).background_model()
    # z, averaging, metrics, reference border and backend keep it valid
    other_z = _config(propagation={"z": (0.0,)}, averaging={"n_obj": 3})
    other_z = replace(other_z, compensation=replace(other_z.compensation, ref_border=8))
    assert model.is_valid_for(other_z)
    model.check_config(other_z)
    # demodulation, regularisation and precision invalidate it
    for changed in (
        _config(demod={"crop": 32, "radius": 10}),
        make_config(demod={"crop": 64, "radius": 20}, compensation={"eps_rel": 1e-2}),
        make_config(demod={"crop": 64, "radius": 20}, precision="single"),
    ):
        assert not model.is_valid_for(changed)
        with pytest.raises(ValueError, match="record a new background"):
            OffAxisPipeline(changed).use_background(model)


def test_shape_must_match_the_crop():
    config = _config()
    bad = BackgroundModel(
        field=np.ones((32, 32), complex),
        compensation=np.ones((32, 32), complex),
        n_frames=1,
        config_hash=config.background_hash(),
    )
    with pytest.raises(ValueError, match="shape"):
        OffAxisPipeline(config).use_background(bad)


def test_corrupted_models_are_rejected(tmp_path):
    model = _pipeline().background_model()
    with pytest.raises(ValueError, match="does not match its content"):
        replace(model, field=model.field * 1.001)

    path = save_background(model, tmp_path / "bg.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["field"] = arrays["field"] * 1.001
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **arrays)
    with pytest.raises(ValueError, match="does not match its content"):
        load_background(tampered)


def test_foreign_files_are_rejected(tmp_path):
    other = tmp_path / "other.npz"
    np.savez(other, data=np.zeros(3))
    with pytest.raises(ValueError, match="not a background file"):
        load_background(other)
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, field=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):  # allow_pickle=False
        load_background(pickled)


def test_model_validation():
    with pytest.raises(ValueError, match="complex"):
        BackgroundModel(np.ones((4, 4)), np.ones((4, 4)), 1, "h")
    with pytest.raises(ValueError, match="2-D"):
        BackgroundModel(np.ones((4, 4), complex), np.ones((4, 5), complex), 1, "h")
    with pytest.raises(ValueError, match="timezone"):
        BackgroundModel(
            np.ones((4, 4), complex),
            np.ones((4, 4), complex),
            1,
            "h",
            created_utc="2026-11-16T10:00:00",
        )


def test_age_warning_only_when_configured():
    model = _pipeline().background_model()
    later = utc_now() + timedelta(hours=2)
    assert model.warnings(None, now=later) == []
    assert model.warnings(3 * 3600, now=later) == []
    (warning,) = model.warnings(3600, now=later)
    assert model.model_id in warning and "maximum 3600 s" in warning


def test_result_reports_the_background():
    session = ReconstructionSession(_config(), background_max_age_s=0.0)
    session.set_background(SyntheticDHM(seed=2).frames(4, obj=False))
    for frame in _objects():
        session.push(frame)
    res = session.result()
    model = session.background_model()
    assert res.quality["background_id"] == model.model_id
    assert res.quality["background_created_utc"] == model.created_utc
    assert res.quality["background_age_s"] >= 0
    assert len(res.warnings) == 1  # max age 0 s: any background is too old
    header, _ = res.to_message()
    assert header["warnings"] == res.warnings


def test_reconstruct_frames_with_a_stored_model(tmp_path):
    pipe = _pipeline()
    path = save_background(pipe.background_model(), tmp_path / "bg.npz")
    frames = _objects()
    from_frames = reconstruct_frames(
        frames, SyntheticDHM(seed=2).frames(6, obj=False, noise=0.02), _config()
    )
    from_model = reconstruct_frames(frames, load_background(path), _config())
    np.testing.assert_allclose(from_model.phase_rad, from_frames.phase_rad, atol=1e-6)
    assert from_model.warnings == []
    assert from_model.quality["background_id"] == pipe.background_model().model_id
