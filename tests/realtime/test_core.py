"""
Realtime kernel OffAxisPipeline: gates G1.1-G1.6 of docs/realtime (CPU part;
GPU variants are marked and run on the measurement PC).
"""

import ast
import importlib.util
import inspect
import logging
import textwrap
from pathlib import Path

import numpy as np
import pytest
from synthetic_dhm import WAVELENGTH, SyntheticDHM, make_config, requires_gpu

from offaxisholo import backend
from offaxisholo.realtime import OffAxisPipeline
from offaxisholo.realtime.core import tukey

PARITY = {"compensation": {"eps_rel": "machine", "piston": False}}


def _offline(dhm, padding, z=1.1e-7):
    from offaxisholo.single_functions import reconstruct_hologram

    logging.disable(logging.INFO)
    try:
        return reconstruct_hologram(
            dhm.frame(obj=True),
            dhm.frame(obj=False),
            dhm_dictionary={
                "name": "synthetic",
                "dcRadius": 10,
                "pixelPitch": [0.1e-6, 0.1e-6],
                "wavelength": WAVELENGTH,
                "propagationDistance": z,
            },
            material_dictionary={"refractiveIndex": 1.5},
            propagation_padding=padding,
        )
    finally:
        logging.disable(logging.NOTSET)


def _run(config, background, objects, xp=None):
    pipe = OffAxisPipeline(config, xp=xp)
    pipe.set_background(background)
    for frame in objects:
        pipe.push(frame)
    return pipe


# ------------------------------------------------------------ G1.1 parity
@pytest.mark.parametrize("padding", [False, True])
def test_parity_with_offline_processor(dhm, padding):
    ref = _offline(dhm, padding)
    config = make_config(propagation={"z": (1.1e-7,), "padding": padding}, **PARITY)
    pipe = _run(config, [dhm.frame(obj=False)], [dhm.frame(obj=True)])

    np.testing.assert_array_equal(
        pipe.demodulate(dhm.frame(obj=True)), ref.hologram.reconstructed_field
    )
    np.testing.assert_allclose(
        pipe.compensated_field(), ref.field_compensated, rtol=0, atol=1e-12
    )
    res = pipe.result()
    np.testing.assert_allclose(res.field, ref.field_propagated, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.phase, ref.phase_wrapped, atol=1e-9)


def test_crop_samples_parity_field_exactly(dhm):
    """IFFT after cropping to n: the field is the parity field sampled with
    stride N/n (band limited signal, mask inside the crop)."""
    background, obj = [dhm.frame(obj=False)], [dhm.frame(obj=True)]
    parity = {"eps_rel": "machine", "piston": False}
    full = _run(
        make_config(demod={"radius": 30}, compensation=parity), background, obj
    ).compensated_field()
    cropped = _run(
        make_config(demod={"crop": 64, "radius": 30}, compensation=parity),
        background,
        obj,
    )
    assert cropped.crop_pitch == pytest.approx((0.2e-6, 0.2e-6))
    np.testing.assert_allclose(cropped.compensated_field(), full[::2, ::2], atol=1e-10)


def test_single_precision_close_to_double(dhm):
    background, obj = [dhm.frame(obj=False)], [dhm.frame(obj=True)]
    config = make_config(propagation={"z": (-2e-6,)}, **PARITY)
    ref = _run(config, background, obj).result()
    single = _run(
        make_config(precision="single", propagation={"z": (-2e-6,)}, **PARITY),
        background,
        obj,
    ).result()
    assert single.field.dtype == np.complex64
    d = np.angle(single.field * np.conj(ref.field))
    assert np.max(np.abs(d)) < 1e-3


# -------------------------------------------------------- G1.2 linearity
def test_mean_before_propagation_equals_propagation_before_mean(dhm):
    background = [dhm.frame(obj=False)]
    frames = list(dhm.frames(4, noise=0.2))
    config = make_config(
        propagation={"z": (-1.5e-6,), "padding": True},
        compensation={"piston": False},
    )
    averaged = _run(config, background, frames).result().field
    singles = [_run(config, background, [f]).result().field for f in frames]
    np.testing.assert_allclose(averaged, np.mean(singles, axis=0), atol=1e-12)


def test_results_share_fft_and_match_single_results(dhm):
    config = make_config(propagation={"z": (0.0, -1e-6, 2e-6)})
    pipe = _run(config, [dhm.frame(obj=False)], [dhm.frame()])
    many = pipe.results()
    for res in many:
        np.testing.assert_array_equal(res.field, pipe.result(res.z).field)
    np.testing.assert_array_equal(many[0].field, pipe.compensated_field())
    with pytest.raises(ValueError, match="results"):
        pipe.result()


# ----------------------------------------------------- G1.3 piston phase
@pytest.mark.parametrize("sigma", [0.5, 1.0])
def test_piston_correction_keeps_amplitude(sigma):
    dhm = SyntheticDHM(seed=7)
    background = [dhm.frame(obj=False)]
    frames = list(dhm.frames(300, piston_std=sigma))
    clean = _run(make_config(), background, [dhm.frame()]).compensated_field()
    border = np.zeros((128, 128), bool)
    border[:12, :] = border[-12:, :] = border[:, :12] = border[:, -12:] = True

    def amplitude_ratio(piston):
        cfg = make_config(compensation={"piston": piston, "ref_border": 12})
        field = _run(cfg, background, frames).compensated_field()
        return np.mean(np.abs(field)[border]) / np.mean(np.abs(clean)[border]), field

    ratio_off, _ = amplitude_ratio(False)
    ratio_on, corrected = amplitude_ratio(True)
    assert ratio_off == pytest.approx(np.exp(-(sigma**2) / 2), abs=0.08)
    assert ratio_on == pytest.approx(1.0, abs=1e-6)
    # phase identical to the clean result up to a global constant; the small
    # rest comes from DC/twin-image leakage into the mask, which does not
    # rotate with the piston phase
    d = np.angle(corrected * np.conj(clean))
    assert np.std(d) < 1e-4


def test_piston_statistics(dhm):
    pipe = _run(
        make_config(),
        list(dhm.frames(20, obj=False, piston_std=0.3)),
        list(dhm.frames(200, piston_std=0.3)),
    )
    assert pipe.piston_std() == pytest.approx(0.3, abs=0.06)
    quality = pipe.background_quality()
    assert quality["n_bg"] == 20
    # background frames are aligned to the first frame -> spread sigma
    assert quality["piston_std_rad"] == pytest.approx(0.3, abs=0.1)


# -------------------------------------------- G1.4 noise scaling / floor
def _phase_noise(n_obj, n_bg, seed=11):
    dhm = SyntheticDHM(seed=seed)
    background = list(dhm.frames(n_bg, obj=False, noise=1.0))
    config = make_config(demod={"crop": 64, "radius": 30})
    pipe = _run(config, background, dhm.frames(n_obj, noise=1.0))
    clean = _run(config, [dhm.frame(obj=False)], [dhm.frame()])
    d = np.angle(pipe.compensated_field() * np.conj(clean.compensated_field()))
    return np.std(d[4:12, 4:60])  # object free border band


def test_noise_decreases_with_object_frames():
    ratio = _phase_noise(1, 64) / _phase_noise(16, 64)
    expected = np.sqrt((1 + 1 / 64) / (1 / 16 + 1 / 64))  # ~3.6
    assert ratio == pytest.approx(expected, rel=0.25)


def test_background_noise_floor():
    # with one background frame more object frames hardly help
    assert _phase_noise(16, 1) / _phase_noise(64, 1) < 1.3


def test_background_quality_noise_estimate():
    dhm = SyntheticDHM(seed=3)
    config = make_config(demod={"crop": 64, "radius": 30})
    pipe = OffAxisPipeline(config)
    pipe.set_background(dhm.frames(32, obj=False, noise=1.0))
    clean = OffAxisPipeline(config)
    clean.set_background([dhm.frame(obj=False)])
    measured = np.std(np.angle(pipe.Ub * np.conj(clean.Ub)))
    estimate = pipe.background_quality()["phase_noise_estimate_rad"]
    assert estimate == pytest.approx(measured, rel=0.3)


# ------------------------------------------------ sliding / ema in the kernel
def test_sliding_mode_equals_block_of_last_frames(dhm):
    background = [dhm.frame(obj=False)]
    frames = list(dhm.frames(10, noise=0.3, piston_std=0.2))
    sliding = _run(
        make_config(averaging={"mode": "sliding", "window": 4}), background, frames
    )
    block = _run(make_config(), background, frames[-4:])
    np.testing.assert_allclose(
        sliding.compensated_field(), block.compensated_field(), atol=1e-10
    )
    assert sliding.result().meta["n_effective"] == 4


# --------------------------------------------- G1.6 hot path, error cases
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


def test_hot_path_has_no_host_synchronisation():
    """Static check of the per-frame methods (CPU substitute of the GPU guard)."""
    for method in (
        OffAxisPipeline.push,
        OffAxisPipeline.demodulate,
        OffAxisPipeline._piston,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                assert name not in FORBIDDEN_CALLS, f"{method.__name__} calls {name}"


@requires_gpu
def test_gpu_hot_path_without_synchronisation_and_allocation(dhm):
    import cupy
    import cupyx

    config = make_config(precision="single", demod={"crop": 64, "radius": 30})
    config = type(config)(**{**config.__dict__, "backend": "cupy"})
    pipe = OffAxisPipeline(config)
    pipe.set_background(cupy.asarray(f) for f in dhm.frames(4, obj=False))
    frames = [cupy.asarray(f) for f in dhm.frames(20)]
    for f in frames[:5]:
        pipe.push(f)  # warm-up (FFT plans, buffers)
    cupy.cuda.Device().synchronize()
    pool = cupy.get_default_memory_pool()
    before = pool.total_bytes()
    with cupyx.allow_synchronize(False):
        for f in frames[5:]:
            pipe.push(f)
    cupy.cuda.Device().synchronize()
    assert pool.total_bytes() == before


@requires_gpu
def test_gpu_parity_single_and_double(dhm):
    background, obj = [dhm.frame(obj=False)], [dhm.frame()]
    for precision, tol in (("double", 1e-9), ("single", 1e-3)):
        config = make_config(precision=precision, propagation={"z": (-2e-6,)}, **PARITY)
        ref = _run(config, background, obj).result()
        gpu = _run(config, background, obj, xp=backend.namespace("cupy")).result()
        d = np.angle(gpu.to_numpy(("field",))["field"] * np.conj(ref.field))
        assert np.max(np.abs(d)) < tol


def test_error_cases(dhm):
    pipe = OffAxisPipeline(make_config())
    with pytest.raises(RuntimeError, match="Background"):
        pipe.push(dhm.frame())
    with pytest.raises(RuntimeError):
        pipe.compensated_field()
    with pytest.raises(ValueError, match="At least one"):
        pipe.set_background([])
    with pytest.raises(ValueError, match="shape"):
        pipe.set_background([np.zeros((64, 64))])
    with pytest.raises(ValueError, match="positive"):
        pipe.set_background([-dhm.frame()])
    pipe.set_background([dhm.frame(obj=False)])
    with pytest.raises(ValueError, match="shape"):
        pipe.push(np.zeros((128, 64)))
    with pytest.raises(RuntimeError, match="No frames"):
        pipe.result()
    with pytest.raises(ValueError):
        OffAxisPipeline(make_config(order=None))


def test_raw_camera_dtypes_and_reset(dhm):
    pipe = OffAxisPipeline(make_config(precision="single"))
    scale = 60000 / dhm.frame().max()
    pipe.set_background([np.round(dhm.frame(obj=False) * scale).astype(np.uint16)])
    pipe.push(np.round(dhm.frame() * scale).astype(np.uint16))
    assert pipe.n_frames == 1
    res = pipe.result()
    assert res.phase.dtype == np.float32
    out = res.to_numpy(("phase", "amplitude", "intensity_db"))
    assert set(out) == {"phase", "amplitude", "intensity_db"}
    pipe.reset()
    assert pipe.n_frames == 0


def test_warmup_fills_transfer_cache():
    pipe = OffAxisPipeline(
        make_config(propagation={"z": (-1e-6, 2e-6), "padding": True})
    )
    pipe.warmup()
    assert len(pipe._transfer_cache) == 2


# ------------------------------------------------- reference skeleton (C)
def _load_skeleton():
    path = Path(__file__).parents[2] / "docs/realtime/reference/holo_gpu_pipeline.py"
    spec = importlib.util.spec_from_file_location("holo_gpu_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tukey_matches_scipy():
    from scipy.signal.windows import tukey as scipy_tukey

    for n, alpha in [(64, 0.5), (256, 0.1), (16, 1.0), (16, 0.0)]:
        np.testing.assert_allclose(tukey(n, alpha), scipy_tukey(n, alpha), atol=1e-15)


def test_matches_reference_skeleton():
    """Kernel with the skeleton settings (Tukey windows, crop, eps_rel, piston)
    reproduces holo_gpu_pipeline.OffAxisPipeline (CPU, float32)."""
    skeleton = _load_skeleton()
    if skeleton.GPU:
        pytest.skip("skeleton runs on the GPU here; comparison is CPU only")
    n, m, carrier = 256, 64, (0.25, 0.18)
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:n, 0:n].astype(float)
    obj = np.exp(1j * 1.5 * np.exp(-((x - 100) ** 2 + (y - 125) ** 2) / (2 * 15**2)))
    ref = np.exp(2j * np.pi * (carrier[0] * x + carrier[1] * y))

    def holo(with_object, piston):
        field = (obj if with_object else 1.0) * np.exp(1j * piston)
        return (np.abs(ref + 0.8 * field) ** 2).astype(np.float32)

    background = [holo(False, rng.normal(0, 0.5)) for _ in range(4)]
    objects = [holo(True, rng.normal(0, 0.5)) for _ in range(4)]

    sk = skeleton.OffAxisPipeline(
        n=n, crop=m, carrier=carrier, pixel_size_um=0.5, wavelength_um=0.532
    )
    sk.set_background(background)
    for h in objects:
        sk.push(h)

    order = (-round(carrier[0] * n), -round(carrier[1] * n))
    config = make_config(
        order=order,
        size=n,
        precision="single",
        demod={"crop": m, "mask": "tukey", "sideband_alpha": 0.5, "edge_alpha": 0.1},
        compensation={"eps_rel": 1e-3, "piston": True},
        propagation={"z": (0.0, 120e-6)},
    )
    config = type(config)(
        **{
            **config.__dict__,
            "dhm": type(config.dhm)(
                wavelength=0.532e-6, pixel_pitch=(0.5e-6, 0.5e-6), dc_radius=10
            ),
        }
    )
    pipe = _run(config, background, objects)
    plane, propagated = pipe.results()

    def max_rel(a, b):
        return np.max(np.abs(a - b)) / np.max(np.abs(b))

    assert max_rel(plane.field, sk.result()) < 1e-4
    assert max_rel(propagated.field, sk.result(120.0)) < 1e-3
