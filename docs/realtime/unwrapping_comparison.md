# Phase unwrapping: Fast 2D vs. Least Squares (G1.8)

Generated 2026-11-13 with `tools/compare_unwrapping.py` (CPU, NumPy, double precision).

## Synthetic phases (RMS error vs. truth in rad, offset removed)

| case | noise (rad) | Fast 2D | Least Squares | Least Squares (congruent) |
|---|---|---|---|---|
| smooth bump 25 rad | 0.0 | 0.000 | 0.000 | 0.000 |
| smooth bump 25 rad | 0.3 | 0.301 | 0.301 | 0.301 |
| smooth bump 25 rad | 0.6 | 0.613 | 0.607 | 0.618 |
| step 2 rad (< pi) | 0.0 | 0.000 | 0.000 | 0.000 |
| step 2 rad (< pi) | 0.3 | 0.301 | 0.302 | 0.301 |
| step 2 rad (< pi) | 0.6 | 0.601 | 0.627 | 0.601 |
| step 5 rad (> pi, aliased) | 0.0 | 2.314 | 2.314 | 2.314 |
| step 5 rad (> pi, aliased) | 0.3 | 2.334 | 2.334 | 2.334 |
| step 5 rad (> pi, aliased) | 0.6 | 2.391 | 2.366 | 2.391 |
| bump + step 2 rad | 0.0 | 0.000 | 0.000 | 0.000 |
| bump + step 2 rad | 0.3 | 0.300 | 0.307 | 0.378 |
| bump + step 2 rad | 0.6 | 0.600 | 0.742 | 2.359 |

## Real holograms (no ground truth; preset Zeiss 63x, default propagation)

| hologram | size | 2pi jumps in wrapped phase | unwrapped range (rad) | RMS(LS - Fast2D) | max abs diff | pixels > pi | LS congruent RMS diff | t Fast2D (s) | t LS (s) |
|---|---|---|---|---|---|---|---|---|---|
| developed/DOE1.tif | 1024x1024 | 6159 | 22.77 | 1.802 | 10.68 | 7.81 % | 3.062 | 0.81 | 0.25 |
| developed/rect1.tif | 1024x1024 | 752 | 11.11 | 0.654 | 4.93 | 0.05 % | 0.345 | 0.90 | 0.14 |
| developed/stair1.tif | 1024x1024 | 3151 | 17.24 | 2.327 | 8.49 | 21.92 % | 2.992 | 0.89 | 0.16 |
| print/DOE1.png | 1024x1024 | 0 | 2.34 | 0.000 | 0.00 | 0.00 % | 0.000 | 0.84 | 0.25 |
| print/rect1.png | 1024x1024 | 260 | 2.03 | 0.000 | 0.00 | 0.00 % | 0.000 | 0.62 | 0.12 |
| print/stair1.png | 1024x1024 | 0 | 1.69 | 0.000 | 0.00 | 0.00 % | 0.000 | 0.80 | 0.24 |

## Notes

- 'Least Squares' distributes phase residues (noise, discontinuities) over the image instead of isolating them: smoother result, step heights can be underestimated.
- 'congruent' adds W(psi - phi) to the LS result, so that it differs from the wrapped phase only by multiples of 2 pi.
- Steps larger than pi between neighbouring pixels violate the sampling condition; no unwrapping method can recover them.
- Real holograms without 2 pi jumps (wrapped phase already continuous) are trivially identical for both methods and do not discriminate them.
- Decision on the default method: user (see ToDos.txt). Until then 'Fast 2D' stays the default.

## K11-T — test plan: temporal vs. spatial unwrapping

Added 2026-11-18 from the decision session of 2026-11-17/18 (O5, work package 7). Executed by K11.

**Rationale.** Spatial unwrapping fails first at steep walls and at steps > π between neighbouring
pixels. Between two layers the phase changes little, so unwrapping the difference to the previous layer
avoids the ambiguity: φₙ = φₙ₋₁ + wrap(φₙ − φₙ₋₁) (`TemporalUnwrapper`). Printed structures are ≤ 15 µm
high; with SZ2080 in-situ (Δn = 0.012 at 674.994 nm) the phase stays ≤ ≈ 1.7 rad < π, so **no 2π
ambiguity is expected in-situ**. Jumps appear in-situ only for Δn > λ/(2h) ≈ 0.0225 (h = 15 µm). The
question matters for materials with a higher Δn, for the general choice of method, and for developed
structures in air (Δn ≈ 0.52 → ≈ 72 rad at 15 µm).

**Preconditions.** A fixed mosaic grid per structure, registration between layers (MIRA), and the
previous-layer state per structure on the DHM-PC.

**Data.**
- Synthetic layer sequences: stairs, pillars, a DOE profile, steep walls; final height ≤ 15 µm;
  Δn 0.01–0.05; noise 0.05–0.6 rad; coherent pattern noise; one missing layer and one mis-registered
  layer.
- Real in-situ series with `keep_raw = all`, plus an ex-situ reference (Sensofar / AFM) of the developed
  structure.
- Developed structures in air for `fast2d` vs. least squares.

**Methods.** `fast2d` per layer; least squares (`congruent=True` and `False`); `temporal`; `temporal`
with re-anchoring (spatial re-anchor every N layers or when the consistency check fails).

**Metrics.** RMS error against the truth, step-height error, error at walls, drift over the layer count,
recovery after a bad layer, residues, runtime per tile; real data: agreement of the final layer with the
ex-situ reference and layer-to-layer smoothness.

**Risk.** Errors propagate in `temporal` → re-anchoring and consistency checks; during the study store
the spatial **and** the temporal result per layer.

**Criterion.** `temporal` becomes the in-situ default only if it is never worse than `fast2d`, is clearly
better at walls / tall structures, shows no drift above the noise over the full layer count, and recovers
from a bad layer by re-anchoring. Least squares becomes a default only if the same comparison shows it
better than `fast2d` on developed structures.
