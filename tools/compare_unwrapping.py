"""
Comparison of the phase unwrapping methods 'Fast 2D' (Herraez, scikit-image)
and 'Least Squares' (DCT, Ghiglia-Romero), gate G1.8 of docs/realtime.

    python tools/compare_unwrapping.py [--data DIR] [--out FILE]

1. Synthetic phases with known truth: smooth bump, steps below/above the
   sampling limit (pi per pixel), with and without noise -> RMS error.
2. Real holograms (default: data/developed and data/print, object +
   background pairs, preset 'Zeiss 63x'): no truth available -> difference
   between the methods, fraction of pixels differing by more than pi,
   runtime.

Writes a Markdown report (default docs/realtime/unwrapping_comparison.md).
"""

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np

from offaxisholo.reconstruction.phaseUnwrapping import (
    phase_unwrapping_fast2d,
    phase_unwrapping_ls_dct,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "Fast 2D": phase_unwrapping_fast2d,
    "Least Squares": phase_unwrapping_ls_dct,
    "Least Squares (congruent)": lambda p: phase_unwrapping_ls_dct(p, congruent=True),
}


def wrap(phase):
    return np.angle(np.exp(1j * phase))


def rms_error(result, truth):
    d = (result - truth) - np.mean(result - truth)  # arbitrary offset removed
    return float(np.sqrt(np.mean(d**2)))


def synthetic_cases(n=256, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n), dtype=float)
    bump = 25 * np.exp(-((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * (n / 8) ** 2))
    plateau = ((abs(x - n / 2) < n / 5) & (abs(y - n / 2) < n / 5)).astype(float)
    cases = {
        "smooth bump 25 rad": bump,
        "step 2 rad (< pi)": 2.0 * plateau,
        "step 5 rad (> pi, aliased)": 5.0 * plateau,
        "bump + step 2 rad": bump + 2.0 * plateau,
    }
    out = []
    for name, truth in cases.items():
        for noise in (0.0, 0.3, 0.6):
            noisy = truth + noise * rng.standard_normal(truth.shape)
            out.append((name, noise, truth, wrap(noisy)))
    return out


def real_pairs(data_dir):
    """(name, object, background) image pairs from data/developed and print."""
    pairs = []
    developed = data_dir / "developed"
    for obj, bg in [
        ("DOE1.tif", "DOE_background.tif"),
        ("rect1.tif", "rect_background.tif"),
        ("stair1.tif", "stair_background.tif"),
    ]:
        if (developed / obj).exists() and (developed / bg).exists():
            pairs.append((f"developed/{obj}", developed / obj, developed / bg))
    printed = data_dir / "print"
    for name in ("DOE1", "rect1", "stair1"):
        obj, bg = printed / f"{name}.png", printed / f"background {name}.png"
        if obj.exists() and bg.exists():
            pairs.append((f"print/{name}.png", obj, bg))
    from offaxisholo import DataLoader

    # load exactly as the library does (grayscale conversion of colour TIFs)
    return [
        (
            name,
            DataLoader(file_path=str(obj)).get_data(),
            DataLoader(file_path=str(bg)).get_data(),
        )
        for name, obj, bg in pairs
    ]


def timed(func, *args):
    t0 = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - t0


def run_synthetic():
    lines = [
        "| case | noise (rad) | " + " | ".join(METHODS) + " |",
        "|---|---|" + "---|" * len(METHODS),
    ]
    for name, noise, truth, wrapped in synthetic_cases():
        errors = [rms_error(f(wrapped), truth) for f in METHODS.values()]
        lines.append(
            f"| {name} | {noise:.1f} | " + " | ".join(f"{e:.3f}" for e in errors) + " |"
        )
    return lines


def run_real(data_dir):
    from offaxisholo.single_functions import reconstruct_hologram

    pairs = real_pairs(data_dir)
    if not pairs:
        return ["No real data found."]
    lines = [
        "| hologram | size | 2pi jumps in wrapped phase | unwrapped range (rad) "
        "| RMS(LS - Fast2D) | max abs diff | pixels > pi | LS congruent RMS diff "
        "| t Fast2D (s) | t LS (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, obj, bg in pairs:
        try:
            proc = reconstruct_hologram(obj, bg, dhm_preset="Zeiss 63x")
        except Exception as err:  # noqa: BLE001 - report and continue
            lines.append(f"| {name} | {obj.shape} | failed: {err} |" + " |" * 7)
            continue
        wrapped = proc.phase_wrapped
        fast, t_fast = timed(phase_unwrapping_fast2d, wrapped)
        ls, t_ls = timed(phase_unwrapping_ls_dct, wrapped)
        lsc = phase_unwrapping_ls_dct(wrapped, congruent=True)
        jumps = int(
            np.sum(np.abs(np.diff(wrapped, axis=0)) > np.pi)
            + np.sum(np.abs(np.diff(wrapped, axis=1)) > np.pi)
        )
        d = (ls - fast) - np.mean(ls - fast)
        dc = (lsc - fast) - np.median(lsc - fast)
        lines.append(
            f"| {name} | {wrapped.shape[0]}x{wrapped.shape[1]} | {jumps} "
            f"| {np.ptp(fast):.2f} | {np.sqrt(np.mean(d**2)):.3f} "
            f"| {np.max(np.abs(d)):.2f} "
            f"| {np.mean(np.abs(d) > np.pi) * 100:.2f} % "
            f"| {np.sqrt(np.mean(dc**2)):.3f} | {t_fast:.2f} | {t_ls:.2f} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs/realtime/unwrapping_comparison.md"
    )
    args = parser.parse_args()
    logging.disable(logging.INFO)

    report = [
        "# Phase unwrapping: Fast 2D vs. Least Squares (G1.8)",
        "",
        f"Generated {date.today().isoformat()} with `tools/compare_unwrapping.py` "
        "(CPU, NumPy, double precision).",
        "",
        "## Synthetic phases (RMS error vs. truth in rad, offset removed)",
        "",
        *run_synthetic(),
        "",
        "## Real holograms (no ground truth; preset Zeiss 63x, default propagation)",
        "",
        *run_real(args.data),
        "",
        "## Notes",
        "",
        "- 'Least Squares' distributes phase residues (noise, discontinuities) over "
        "the image instead of isolating them: smoother result, step heights can be "
        "underestimated.",
        "- 'congruent' adds W(psi - phi) to the LS result, so that it differs from "
        "the wrapped phase only by multiples of 2 pi.",
        "- Steps larger than pi between neighbouring pixels violate the sampling "
        "condition; no unwrapping method can recover them.",
        "- Real holograms without 2 pi jumps (wrapped phase already continuous) "
        "are trivially identical for both methods and do not discriminate them.",
        "- Decision on the default method: user (see ToDos.txt). Until then "
        "'Fast 2D' stays the default.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
