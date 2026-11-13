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
