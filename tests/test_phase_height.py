"""Phase <-> height with explicit indices in ``fieldops`` (K20, G4)."""

import math

import numpy as np
import pytest

from offaxisholo.reconstruction.fieldops import height_to_phase_rad, phase_to_height_um

WAVELENGTH_UM = 0.674994
SZ2080_IN_SITU = {"n_structure": 1.52, "n_surround": 1.508}  # delta n = 0.012
SZ2080_DEVELOPED = {"n_structure": 1.52, "n_surround": 1.0}  # delta n = 0.52


def optics(indices, sign=1):
    return {"wavelength_um": WAVELENGTH_UM, "phase_sign": sign, **indices}


@pytest.mark.parametrize(
    "indices, delta_n", [(SZ2080_IN_SITU, 0.012), (SZ2080_DEVELOPED, 0.52)]
)
def test_analytic_step(indices, delta_n):
    """A 15 µm step: phase = 2 pi delta_n h / lambda, and back."""
    height = np.zeros((32, 32))
    height[8:24, 8:24] = 15.0
    phase = height_to_phase_rad(height, **optics(indices))
    expected = 2 * math.pi * delta_n * 15.0 / WAVELENGTH_UM
    assert phase[16, 16] == pytest.approx(expected, rel=1e-12)
    assert phase[0, 0] == 0.0
    np.testing.assert_allclose(
        phase_to_height_um(phase, **optics(indices)), height, atol=1e-12
    )


def test_in_situ_step_has_no_two_pi_ambiguity():
    """O5: a 15 µm SZ2080 structure in-situ stays below pi (about 1.68 rad)."""
    phase = height_to_phase_rad(15.0, **optics(SZ2080_IN_SITU))
    assert phase == pytest.approx(1.6757, abs=1e-3)
    assert phase < math.pi
    developed = height_to_phase_rad(15.0, **optics(SZ2080_DEVELOPED))
    assert developed == pytest.approx(72.6, abs=0.1)  # many 2 pi jumps in air


def test_sign_convention():
    """More optical path -> positive phase; n_structure > n_surround -> h > 0."""
    assert height_to_phase_rad(1.0, **optics(SZ2080_IN_SITU)) > 0
    assert phase_to_height_um(1.0, **optics(SZ2080_IN_SITU)) > 0
    assert phase_to_height_um(1.0, **optics(SZ2080_IN_SITU, sign=-1)) < 0
    inverted = {"n_structure": 1.0, "n_surround": 1.52}  # e.g. a hole in resin
    assert phase_to_height_um(1.0, **optics(inverted)) < 0


def test_no_default_indices():
    with pytest.raises(TypeError):
        phase_to_height_um(1.0, wavelength_um=WAVELENGTH_UM, phase_sign=1)
    with pytest.raises(TypeError):
        phase_to_height_um(1.0, WAVELENGTH_UM, 1.52, 1.508, 1)  # keywords only


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"phase_sign": 0}, "phase_sign"),
        ({"phase_sign": 2}, "phase_sign"),
        ({"wavelength_um": 0.0}, "wavelength_um"),
        ({"n_surround": 0.0}, "positive"),
        ({"n_surround": 1.52}, "equals"),
    ],
)
def test_invalid_parameters(changes, message):
    params = {**optics(SZ2080_IN_SITU), **changes}
    with pytest.raises(ValueError, match=message):
        phase_to_height_um(1.0, **params)
    with pytest.raises(ValueError, match=message):
        height_to_phase_rad(1.0, **params)


def test_matches_the_legacy_formula():
    """Same physics as the positional legacy function (metres there)."""
    from offaxisholo.reconstruction.fieldops import phase_to_height

    legacy_m = phase_to_height(2.0, WAVELENGTH_UM * 1e-6, 1.52, 1.508)
    new_um = phase_to_height_um(2.0, **optics(SZ2080_IN_SITU))
    assert new_um == pytest.approx(legacy_m * 1e6, rel=1e-12)
