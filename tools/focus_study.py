"""
Focus study with the realtime kernel (K10, G4).

    python tools/focus_study.py [--data DIR] [--preset "Zeiss 63x"]
        [--start -6] [--stop 2] [--step 0.1] [--index 1.0 1.508] [--roi full|centre]

For every object / background pair of ``data/developed`` and ``data/print``
(same pairs as tools/compare_unwrapping.py): calibrate the first order on the
background, reconstruct in parity mode (no crop, circular mask), sweep z and
print the minimum of the amplitude coefficient of variation per propagation
index. The ROI ``centre`` is the central half of the field (where the test
structures are); ``full`` uses the whole field.

The result is an estimate per capture; the default z per objective is set by
the maintainer from it (K9 parameter set, marked ``unverified`` until checked
against Sensofar / AFM).
"""

import argparse
from pathlib import Path

import numpy as np

from offaxisholo.realtime import (
    DemodConfig,
    DHMConfig,
    OffAxisPipeline,
    RealtimeConfig,
    calibrate_order,
    sweep_range,
)

ROOT = Path(__file__).resolve().parents[1]


def pairs(data_dir):
    """(name, object, background) image pairs (loaded like the library)."""
    from offaxisholo import DataLoader

    found = []
    developed = data_dir / "developed"
    for obj, bg in [
        ("DOE1.tif", "DOE_background.tif"),
        ("rect1.tif", "rect_background.tif"),
        ("stair1.tif", "stair_background.tif"),
    ]:
        if (developed / obj).exists() and (developed / bg).exists():
            found.append((f"developed/{obj}", developed / obj, developed / bg))
    printed = data_dir / "print"
    for name in ("DOE1", "rect1", "stair1"):
        obj, bg = printed / f"{name}.png", printed / f"background {name}.png"
        if obj.exists() and bg.exists():
            found.append((f"print/{name}.png", obj, bg))
    return [
        (
            name,
            np.asarray(DataLoader(file_path=str(obj)).get_data(), dtype=float),
            np.asarray(DataLoader(file_path=str(bg)).get_data(), dtype=float),
        )
        for name, obj, bg in found
    ]


def square(image):
    """Central square crop (the kernel needs N x N frames)."""
    n = min(image.shape) // 2 * 2
    r0 = (image.shape[0] - n) // 2
    c0 = (image.shape[1] - n) // 2
    return image[r0 : r0 + n, c0 : c0 + n]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--preset", default="Zeiss 63x")
    parser.add_argument("--start", type=float, default=-6.0)
    parser.add_argument("--stop", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--index", type=float, nargs="+", default=[1.0, 1.508])
    parser.add_argument("--roi", choices=("full", "centre"), default="centre")
    args = parser.parse_args()

    dhm = DHMConfig.from_preset(args.preset)
    z = sweep_range(args.start, args.stop, args.step)
    print(
        f"preset {args.preset}, pitch {dhm.pixel_pitch[0] * 1e6:.4f} um, "
        f"sweep {args.start} .. {args.stop} um step {args.step}, roi {args.roi}"
    )
    print(f"{'capture':24s}" + "".join(f"   z(n={n:g})" for n in args.index))
    for name, obj, bg in pairs(args.data):
        obj, bg = square(obj), square(bg)
        n = obj.shape[0]
        calibration = calibrate_order([bg], dhm.dc_radius)
        config = RealtimeConfig(
            dhm=dhm,
            demod=DemodConfig(size=n, order=calibration.order),
            precision="double",
        ).with_order(calibration.order, calibration.radius)
        pipe = OffAxisPipeline(config)
        pipe.set_background([bg])
        pipe.push(obj)
        roi = None
        if args.roi == "centre":
            m = pipe.n
            roi = np.zeros((m, m), dtype=bool)
            roi[m // 4 : 3 * m // 4, m // 4 : 3 * m // 4] = True
        cells = []
        for index in args.index:
            sweep = pipe.focus_sweep(z, roi=roi, propagation_index=index)
            flag = "*" if sweep.at_edge else " "
            cells.append(f"{sweep.best_z_um:10.2f}{flag}")
        print(f"{name:24s}" + "".join(cells))
    print("* minimum at the edge of the sweep")


if __name__ == "__main__":
    main()
