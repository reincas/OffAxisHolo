"""
z handling and focus sweep (K10, G4): propagation index from the request,
sharpness metric, sweep on synthetic defocused phase objects, autofocus
reported in ``quality`` only.
"""

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config

from offaxisholo import backend
from offaxisholo.realtime import (
    AutofocusConfig,
    FocusSweep,
    OffAxisPipeline,
    ReconstructionResult,
    ReconstructionSession,
    focus_sweep,
    reconstruct_frames,
    sweep_range,
)
from offaxisholo.realtime.focus import amplitude_cv, refine_minimum
from offaxisholo.reconstruction.numericalPropagation import transfer_function

XP = backend.namespace("numpy")
N = 128
PIXEL_UM = 0.2
WAVELENGTH_UM = 0.675


def _phase_object(n=N):
    """In-focus field of a pure phase object (uniform amplitude)."""
    y, x = np.indices((n, n), dtype=float)
    r2 = (x - n / 2) ** 2 + (y - n / 2) ** 2
    bump = 1.5 * np.exp(-r2 / (2 * 6.0**2))
    square = ((abs(x - 40) < 8) & (abs(y - 80) < 8)) * 1.0
    return np.exp(1j * (bump + square))


def _propagate(field, z_um, index):
    H = transfer_function(
        field.shape,
        PIXEL_UM * 1e-6,
        PIXEL_UM * 1e-6,
        WAVELENGTH_UM * 1e-6,
        z_um * 1e-6,
        n_medium=index,
        band_limit=True,
        xp=XP,
        dtype=np.float64,
    )
    return np.fft.ifftn(np.fft.fftn(field) * H)


def _sweep(field, index, z=None):
    return focus_sweep(
        field,
        z or sweep_range(-5.0, 5.0, 0.25),
        pixel_um=(PIXEL_UM, PIXEL_UM),
        wavelength_um=WAVELENGTH_UM,
        propagation_index=index,
    )


# ------------------------------------------------------------ pure functions
@pytest.mark.parametrize(("z0", "index"), [(2.0, 1.0), (-3.1, 1.0), (2.6, 1.5)])
def test_sweep_finds_the_defocus_of_a_phase_object(z0, index):
    defocused = _propagate(_phase_object(), -z0, index)
    sweep = _sweep(defocused, index)
    assert sweep.best_z_um == pytest.approx(z0, abs=0.1)
    assert not sweep.at_edge
    assert sweep.propagation_index == index
    # in a medium n the same z range is a smaller effective defocus (z / n)
    assert sweep.sharpness[sweep.best_index] < 0.5 * np.nanmax(sweep.sharpness)


def test_focus_distance_scales_with_the_propagation_index():
    # paraxial: the same defocus needs n times the distance in a medium n
    defocused = _propagate(_phase_object(), -1.5, 1.0)
    in_air = _sweep(defocused, 1.0).best_z_um
    in_resin = _sweep(defocused, 1.5).best_z_um
    assert in_resin / in_air == pytest.approx(1.5, rel=0.1)


def test_sweep_with_roi_and_edge_flag():
    defocused = _propagate(_phase_object(), -2.0, 1.0)
    roi = np.zeros((N, N), dtype=bool)
    roi[40:90, 40:90] = True  # the bump
    assert _sweep(defocused, 1.0).best_z_um == pytest.approx(
        focus_sweep(
            defocused,
            sweep_range(-5.0, 5.0, 0.25),
            pixel_um=(PIXEL_UM, PIXEL_UM),
            wavelength_um=WAVELENGTH_UM,
            propagation_index=1.0,
            roi=roi,
        ).best_z_um,
        abs=0.15,
    )
    edge = _sweep(defocused, 1.0, z=sweep_range(-3.0, 0.0, 0.5))
    assert edge.at_edge and edge.best_z_um == 0.0
    with pytest.raises(ValueError, match="ROI shape"):
        focus_sweep(
            defocused,
            (0.0, 1.0, 2.0),
            pixel_um=(PIXEL_UM, PIXEL_UM),
            wavelength_um=WAVELENGTH_UM,
            propagation_index=1.0,
            roi=np.ones((4, 4), dtype=bool),
        )


def test_amplitude_cv_whole_field_and_weighted():
    field = np.array([[1.0, 3.0], [1.0, 3.0]]) * np.exp(1j)
    assert float(amplitude_cv(np, field)) == pytest.approx(0.5)
    weight = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert float(amplitude_cv(np, field, weight)) == pytest.approx(0.0)


def test_refine_minimum_of_a_sampled_parabola():
    z = np.arange(-2.0, 2.01, 0.5)
    values = (z - 0.37) ** 2 + 1.0
    index = int(np.argmin(values))
    assert refine_minimum(z, values, index) == pytest.approx(0.37)
    assert refine_minimum(z, values, 0) == z[0]


def test_focus_sweep_result_validation():
    sweep = FocusSweep.from_values(
        [1.0, -1.0, 0.0], [3.0, 2.0, 1.0], propagation_index=1
    )
    np.testing.assert_array_equal(sweep.z_um, [-1.0, 0.0, 1.0])
    assert sweep.best_index == 1
    q = sweep.quality()
    assert set(q) == {
        "autofocus_z_um",
        "autofocus_propagation_index",
        "autofocus_metric",
        "autofocus_at_edge",
        "autofocus_sharpness",
    }
    with pytest.raises(ValueError, match="Duplicate"):
        FocusSweep.from_values([0.0, 0.0], [1.0, 2.0], propagation_index=1)
    with pytest.raises(ValueError, match="finite"):
        FocusSweep.from_values([0.0, 1.0], [np.nan, np.nan], propagation_index=1)
    with pytest.raises(ValueError, match="One metric value"):
        FocusSweep.from_values([0.0, 1.0], [1.0], propagation_index=1)


def test_sweep_range_and_autofocus_config():
    assert sweep_range(-1.0, 1.0, 0.5) == (-1.0, -0.5, 0.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        sweep_range(1.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="three"):
        AutofocusConfig(z_um=(0.0, 1.0)).validate()
    with pytest.raises(ValueError, match="Duplicate"):
        AutofocusConfig(z_um=(0.0, 1.0, 1.0)).validate()
    with pytest.raises(ValueError, match="roi"):
        AutofocusConfig(z_um=(0.0, 1.0, 2.0), roi="other").validate()


# ------------------------------------------------------------------ kernel
def _pipeline(**propagation):
    config = make_config(demod={"crop": 64, "radius": 20}, propagation=propagation)
    pipe = OffAxisPipeline(config)
    dhm = SyntheticDHM()
    pipe.set_background(dhm.frames(2, obj=False))
    for frame in dhm.frames(2):
        pipe.push(frame)
    return pipe


def test_result_uses_the_propagation_index_of_the_request():
    pipe = _pipeline(n_medium=1.0)
    air = pipe.result(-2e-6)
    resin = pipe.result(-2e-6, propagation_index=1.5)
    assert air.meta["propagation_index"] == 1.0
    assert resin.meta["propagation_index"] == 1.5
    E = pipe.compensated_field()
    dy, dx = pipe.crop_pitch[1], pipe.crop_pitch[0]
    H = transfer_function(
        E.shape,
        dx,
        dy,
        pipe.config.dhm.wavelength,
        -2e-6,
        n_medium=1.5,
        band_limit=True,
        xp=XP,
        dtype=np.float64,
    )
    np.testing.assert_allclose(resin.field, np.fft.ifftn(np.fft.fftn(E) * H))
    assert not np.allclose(air.field, resin.field)
    with pytest.raises(ValueError, match="positive"):
        pipe.result(-2e-6, propagation_index=0.0)


def test_pipeline_sweep_equals_pure_sweep_and_does_not_grow_the_cache():
    pipe = _pipeline()
    cached = len(pipe._transfer_cache)
    z = sweep_range(-2.0, 2.0, 0.5)
    sweep = pipe.focus_sweep(z, propagation_index=1.2)
    assert len(pipe._transfer_cache) == cached
    dx, dy = pipe.crop_pitch
    pure = focus_sweep(
        pipe.compensated_field(),
        z,
        pixel_um=(dy * 1e6, dx * 1e6),
        wavelength_um=pipe.config.dhm.wavelength * 1e6,
        propagation_index=1.2,
    )
    np.testing.assert_allclose(sweep.sharpness, pure.sharpness, rtol=1e-10)
    with pytest.raises(ValueError, match="ROI shape"):
        pipe.focus_sweep(z, roi=np.ones((3, 3), dtype=bool))


def test_session_reports_autofocus_without_applying_it():
    dhm = SyntheticDHM()
    background = list(dhm.frames(2, obj=False))
    frames = list(dhm.frames(2))
    config = make_config(demod={"crop": 64, "radius": 20})
    autofocus = AutofocusConfig(z_um=sweep_range(-2.0, 2.0, 0.5))
    plain = reconstruct_frames(frames, background, config, z=-1e-6)
    focused = reconstruct_frames(
        frames,
        background,
        config,
        z=-1e-6,
        propagation_index=1.3,
        autofocus=autofocus,
    )
    assert focused.z_um == pytest.approx(-1.0)
    assert focused.propagation_index == 1.3
    assert focused.quality["autofocus_propagation_index"] == 1.3
    assert -2.0 <= focused.quality["autofocus_z_um"] <= 2.0
    assert "autofocus_z_um" not in plain.quality
    # the result plane is the requested z, not the autofocus estimate
    session = ReconstructionSession(config, autofocus=autofocus)
    session.set_background(background)
    for frame in frames:
        session.push(frame)
    again = session.result(-1e-6)
    assert again.quality["autofocus_propagation_index"] == 1.0
    np.testing.assert_allclose(
        plain.phase_rad,
        reconstruct_frames(frames, background, config, z=-1e-6).phase_rad,
    )
    restored = ReconstructionResult.from_message(*focused.to_message())
    assert restored.quality["autofocus_z_um"] == focused.quality["autofocus_z_um"]


def test_autofocus_uses_the_structure_region_of_an_a_priori_mask():
    dhm = SyntheticDHM()
    config = make_config(demod={"crop": 64, "radius": 20})
    mask = np.ones((64, 64), dtype=bool)
    mask[16:48, 16:48] = False  # structure in the centre
    session = ReconstructionSession(
        config, autofocus=AutofocusConfig(z_um=(-1.0, 0.0, 1.0))
    )
    session.set_reference_mask(mask)
    session.set_background(dhm.frames(2, obj=False))
    for frame in dhm.frames(2):
        session.push(frame)
    result = session.result()
    expected = session.pipeline.focus_sweep((-1.0, 0.0, 1.0), roi=~mask)
    assert result.quality["autofocus_z_um"] == expected.best_z_um
