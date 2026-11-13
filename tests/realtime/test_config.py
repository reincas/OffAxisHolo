"""RealtimeConfig: validation, serialisation, hash, presets."""

import math
import warnings

import numpy as np
import pytest
from synthetic_dhm import make_config

from offaxisholo.realtime import DHMConfig, RealtimeConfig, mask_radius


def test_valid_default_and_derived_values():
    config = make_config().validate()
    assert config.demod.n == 128
    assert config.radius == mask_radius((-32, -32), 128, 10)
    assert config.crop_pitch == (0.1e-6, 0.1e-6)
    cropped = make_config(demod={"crop": 64, "radius": 30})
    assert cropped.crop_pitch == pytest.approx((0.2e-6, 0.2e-6))


def test_mask_radius_matches_offline_hologram():
    from offaxisholo.reconstruction.hologram_class import Hologram

    holo = Hologram.__new__(Hologram)
    holo.data = np.zeros((128, 128))
    holo.params = {"name": "x", "dcRadius": 10}
    for order in [(-32, -32), (-40, 20), (25, -30)]:
        holo.first_diffraction_order_pos = order
        assert mask_radius(order, 128, 10) == holo._calc_radius_mask()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"demod": {"size": 127}}, "even"),
        ({"demod": {"crop": 63}}, "Crop"),
        ({"demod": {"crop": 256}}, "Crop"),
        ({"demod": {"order": None}}, "calibrate"),
        ({"demod": {"order": (-70, 0)}}, "outside"),
        ({"demod": {"crop": 32}}, "radius"),
        ({"demod": {"crop": 64}}, "radius"),  # auto radius 32 = n/2
        ({"demod": {"mask": "square"}}, "mask"),
        ({"averaging": {"mode": "median"}}, "mode"),
        ({"averaging": {"ema_alpha": 0.0}}, "ema_alpha"),
        ({"compensation": {"eps_rel": 0.0}}, "eps_rel"),
        ({"propagation": {"z": (math.inf,)}}, "finite"),
    ],
)
def test_invalid_configurations(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides).validate()


def test_invalid_precision_and_backend():
    with pytest.raises(ValueError):
        make_config(precision="half").validate()


def test_json_roundtrip_and_hash(tmp_path):
    config = make_config(
        propagation={"z": (-2.6e-6, 1e-6)}, demod={"crop": 64, "radius": 30}
    )
    path = tmp_path / "rt.json"
    config.to_json(path)
    loaded = RealtimeConfig.from_json(path)
    assert loaded == config
    assert RealtimeConfig.from_json(config.to_json()) == config
    assert loaded.config_hash() == config.config_hash()
    changed = make_config(
        propagation={"z": (-2.5e-6, 1e-6)}, demod={"crop": 64, "radius": 30}
    )
    assert changed.config_hash() != config.config_hash()


def test_unknown_keys_rejected():
    data = make_config().to_dict()
    data["demod"]["unknown"] = 1
    with pytest.raises(ValueError, match="Unknown keys"):
        RealtimeConfig.from_dict(data)


def test_with_order():
    config = make_config(order=None).with_order((-30, -31), radius=12)
    assert config.demod.order == (-30, -31)
    assert config.radius == 12.0


def test_dhm_from_preset():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dhm = DHMConfig.from_preset("Zeiss 63x")
    assert dhm.wavelength == pytest.approx(0.6749e-6)
    assert dhm.pixel_pitch == pytest.approx((0.0869e-6, 0.0869e-6))
    assert dhm.dc_radius == 304


def test_dhm_from_dict_warns_on_inconsistent_duplicates():
    preset = {
        "wavelength": 0.675e-6,
        "wavelengthUm": 0.7,
        "pixelPitch": [0.1e-6, 0.1e-6],
        "dcRadius": 10,
    }
    with pytest.warns(UserWarning, match="wavelengthUm"):
        dhm = DHMConfig.from_preset(preset)
    assert dhm.wavelength == 0.675e-6
