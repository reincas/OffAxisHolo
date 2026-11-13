"""
Phase noise vs. number of averaged object / background frames (C14, G1.4).

    python tools/noise_vs_n.py                          # synthetic frames
    python tools/noise_vs_n.py --background DIR --objects DIR --order FX FY
                               --dc-radius R [--crop n] [--region r0 r1 c0 c1]

The phase noise is the standard deviation of the phase difference between
two fully independent results (object AND background frames split into two
halves; with a common background its error would cancel), divided by sqrt(2),
in an object free region. Expected:

    sigma^2 ~ sigma_obj^2 / N_obj + sigma_bg^2 / N_bg

-> more object frames only help down to the background floor; the table gives
the numbers for sensible defaults of n_obj and n_bg (to be measured on the
measurement PC with a static scene, frames as tif/png files in two folders).
"""

import argparse
from pathlib import Path

import numpy as np

from offaxisholo.realtime import (
    DemodConfig,
    DHMConfig,
    OffAxisPipeline,
    RealtimeConfig,
)


def synthetic_frames(count, obj, n=256, noise=0.5, seed=0):
    rng = np.random.default_rng(seed + int(obj) * 1000)
    y, x = np.indices((n, n), dtype=float)
    reference = np.exp(2j * np.pi * (x + y) / 4)
    phase = 2 * np.exp(-((x - n / 2) ** 2 + (y - n / 2) ** 2) / (n / 10) ** 2) * obj
    field = np.exp(1j * (phase + rng.normal(0, 0.3)))
    for _ in range(count):
        field = field * np.exp(1j * rng.normal(0, 0.3))  # piston drift
        holo = np.abs(field + reference) ** 2 + noise * rng.standard_normal((n, n))
        yield np.clip(holo, 0, None)  # camera: no negative values


def folder_frames(folder, count):
    import cv2

    files = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in (".tif", ".png")
    )
    if len(files) < count:
        raise ValueError(f"{folder}: {len(files)} frames < {count} requested.")
    for path in files[:count]:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield frame


def phase_noise(config, backgrounds, objects, n_bg, n_obj, region):
    """Noise of one result from n_obj object and n_bg background frames
    (two independent results, each from its own frames)."""
    results = []
    for half in (0, 1):
        pipe = OffAxisPipeline(config)
        pipe.set_background(backgrounds[half * n_bg : (half + 1) * n_bg])
        for frame in objects[half * n_obj : (half + 1) * n_obj]:
            pipe.push(frame)
        results.append(pipe.compensated_field())
    first, second = results
    r0, r1, c0, c1 = region
    d = np.angle(first * np.conj(second))[r0:r1, c0:c1]
    return float(np.std(d) / np.sqrt(2))


def main():
    parser = argparse.ArgumentParser(description="Phase noise vs. N")
    parser.add_argument("--background", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--order", type=int, nargs=2)
    parser.add_argument("--dc-radius", type=float, default=10)
    parser.add_argument("--crop", type=int, default=None)
    parser.add_argument("--region", type=int, nargs=4, default=None)
    parser.add_argument("--n-obj", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--n-bg", type=int, nargs="*", default=[1, 4, 16, 64])
    args = parser.parse_args()

    synthetic = args.background is None
    max_obj, max_bg = 2 * max(args.n_obj), 2 * max(args.n_bg)
    if synthetic:
        n = 256
        objects = list(synthetic_frames(max_obj, obj=True))
        backgrounds = list(synthetic_frames(max_bg, obj=False))
        order = (-n // 4, -n // 4)
        crop = args.crop or n // 2
        region = args.region or (2, 14, 2, crop - 2)  # top border band
    else:
        objects = list(folder_frames(args.objects, max_obj))
        backgrounds = list(folder_frames(args.background, max_bg))
        n = objects[0].shape[0]
        order, crop, region = tuple(args.order), args.crop, args.region
        if region is None:
            raise SystemExit("--region r0 r1 c0 c1 (object free, crop pixels) required")

    config = RealtimeConfig(
        dhm=DHMConfig(
            wavelength=0.675e-6, pixel_pitch=(1e-7, 1e-7), dc_radius=args.dc_radius
        ),
        demod=DemodConfig(
            size=n,
            order=order,
            crop=crop,
            radius=None if crop is None else crop / 2 - 1,
        ),
    )
    print(
        f"{'synthetic' if synthetic else 'measured'} frames, N={n}, crop={crop}, "
        f"region={region}"
    )
    print("phase noise (mrad); rows: n_bg, columns: n_obj")
    print("n_bg \\ n_obj" + "".join(f"{k:>9d}" for k in args.n_obj))
    for n_bg in args.n_bg:
        row = [
            phase_noise(config, backgrounds, objects, n_bg, k, region) * 1e3
            for k in args.n_obj
        ]
        print(f"{n_bg:>12d}" + "".join(f"{v:9.2f}" for v in row))


if __name__ == "__main__":
    main()
