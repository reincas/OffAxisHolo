# Order of the reconstruction steps and numerical propagation

Created 2026-11-12 (analysis with Claude Code) as the local note
`src/offaxisholo/Notes_reconstruction_order.txt`; versioned here since 2026-11-16 (B5).
Status: to be inspected in detail with real measurement data.

This note explains why `HologramProcessor.run` (and the realtime kernel) compensate the
background in the hologram plane **before** propagating, and how the padding option of the
angular spectrum propagation works. Open physics questions from section 6 are tracked as the
decisions O1 / O2 (G4) and the todos P16, P17 in `TODO.md`.

## 1. Order before the change (commit `b9ff53c`, `HologramProcessor.run`)

1. Spatial filtering of the object hologram → `E_obj` (hologram plane)
2. Propagation of `E_obj` by z → `P_z{E_obj}`
3. Spatial filtering of the background hologram → `E_ref` (hologram plane, **not** propagated)
4. "Compensation":
   - `phase = arg(P_z{E_obj}) - arg(E_ref)`
   - `intensity = I(P_z{E_obj}) - I(E_ref)` (linear **and** dB difference)
   - `field = sqrt(I_lin_diff) * sign(I_lin_diff) * exp(i * phase)`
5. Phase unwrapping of the compensated phase
6. Height from the unwrapped phase

The modes "print" and "developed" executed identical code.

## 2. Problems found

a) **Fields of different planes were combined.** The background field `E_ref` describes the
   aberrations A (wavefront curvature, tilt, …) in the hologram plane. After propagation the
   aberration of the object field is no longer A, because propagation does not commute with a
   multiplication:

   `P_z{E_struct · A} ≠ P_z{E_struct} · A` (unless A is constant)

   Dividing the propagated object field by the unpropagated reference therefore leaves an error
   that grows with the curvature of A and with z.

b) **Linear intensities were subtracted.** `I_obj - I_ref` has no physical meaning and can
   become negative (the `sign()` trick then produced negative amplitudes). Correct is the ratio
   `I_obj / I_ref` (= difference in dB).

c) **Old propagation** (`numericalPropagation.angularSpectrum`): evanescent waves were passed
   unchanged (kz = 0 → H = 1 instead of H = 0), the frequency grid was shifted by half a pixel
   for odd sizes, and there was no padding option.

## 3. Order after the change (current `HologramProcessor.run`)

1. Spatial filtering of object and background hologram → `E_obj`, `E_ref` (both in the
   hologram plane, same first-order position)
2. Compensation in the hologram plane:
   - `E_comp = E_obj · conj(E_ref) / |E_ref|²` (= `E_obj / E_ref`)
   - `phase = arg(E_comp)`, `intensity = I_obj / I_ref`
3. Propagation of the compensated field: `P_z{E_comp}`
4. Phase unwrapping of `arg(P_z{E_comp})`
5. Height: `h = λ0 · φ / (2π · (n_structure - n_medium))`
   - mode "developed": `n_medium = 1.0` (air)
   - mode "print": `n_medium = refractiveIndexLiquid` (liquid resin)

Reference for compensation in the hologram plane before propagation: T. Colomb, J. Kühn,
F. Charrière, C. Depeursinge, P. Marquet, N. Aspert, "Total aberrations compensation in digital
holographic microscopy with a reference conjugated hologram," Opt. Express 14, 4300–4306 (2006).

## 4. Numerical test (synthetic data)

Setup: pitch 86.9 nm, λ 674.9 nm, window 512 px (44.5 µm), field band-limited to NA 0.9,
phase structures = discs (radius 1–3 µm, 2 rad step). Ground truth: the same scene on a 4×
larger grid, propagated, centre cropped. Aberration
`A = exp(i · (q · r² / r_edge² + residual tilt of 0.4 FFT bins))`. Values: relative RMS error
of the complex field in the window centre (border band of width `z · tan(asin(NA))` excluded).

| q / rad | z / µm | old order | new order | new order + padding |
|---:|---:|---:|---:|---:|
| 0 | 1.0 | 0.017 | 0.006 | 0.004 |
| 0 | 10.0 | 0.059 | 0.010 | 0.008 |
| 10 | 1.0 | 0.042 | 0.006 | 0.004 |
| 10 | 10.0 | 0.123 | 0.010 | 0.008 |
| 40 | 1.0 | 0.445 | 0.006 | 0.004 |
| 40 | 5.0 | 0.664 | 0.010 | 0.007 |

Old order computed on the 4× larger grid (no wrap-around possible): q = 40, z = 1 µm: 0.447;
q = 40, z = 5 µm: 0.664, i.e. practically identical. **The large errors of the old order are
caused by the order itself (problem 2a), not by the wrap-around of the FFT.**

## 5. Wrap-around (periodic boundaries of the FFT)

The FFT treats the image as one period of an infinitely repeated pattern. Propagation via FFT
is therefore a circular convolution: light leaving the window on the right re-enters on the
left, while physically it would leave the field of view. In addition, if the field does not
match at opposite borders (tilt, curvature), the periodic continuation contains a jump that
diffracts like a sharp edge. This is a well-known error of FFT-based propagation.

Lateral displacement of light with propagation angle θ: `dx = z · tan(θ)`, maximum
`θ = asin(NA / n)`. Example NA 0.9: z = 1 µm → 24 px, z = 10 µm → 238 px (of 512 px).

Test results (relative error in the border band, structures only inside the window,
compensated field):

| z / µm | without padding | with padding |
|---:|---:|---:|
| 1.0 | 0.007 | 0.003 |
| 5.0 | 0.023 | 0.002 |
| 10.0 | 0.039 | 0.002 |

Wrap-around exists and grows with z; padding removes it.

Plain zero padding is **not** sufficient for DHM: the compensated field is ≈ 1 everywhere, so
zero padding creates a hard edge at the window border which diffracts as well (test with
smooth Gaussian structures: 16–20 % error in the border band and 1–3 % in the centre, i.e.
worse than without padding). Implemented solution (`angularSpectrum(..., padding=True)`):

- `c` = mean of the field on the window border,
- `P_z{E} = c · exp(2πi · n · z / λ0) + crop(P_z{zero_pad(E - c)})`.

This is exact (linearity) and gave the smallest errors in all tests.

**Fundamental limit:** light coming from outside the recorded field of view was never
recorded. A border band of width `z · tan(asin(NA / n))` can never be reconstructed correctly,
with any method (test: 5–7 % error remains there when structures exist outside the window).
Exclude this band from evaluation.

## 5b. Test with real data (`Data/print`, `Data/developed`; 2026-11-12)

Assumption: Zeiss 63x preset (pitch 86.9 nm, λ 674.9 nm).

- Background phase of all data sets: about 22–24 rad peak-to-valley over the field of view
  (about 4 waves of aberration), so the order matters.
- `print_DOE1` at the focus: the old order shows ring artefacts inside the structure and strong
  artefacts along the window border; the new order is clean.
- Padding vs. no padding at z = −2.6 µm: centre mean |diff| 0.0009, outer 50 px 0.032. For
  small z the difference is limited to the border.
- Focus (minimum of amplitude std/mean, ROI over the structure):
  `print_DOE1` −2.6 µm, `print_rect1` ≈ −2.0 µm, `print_stair1` ≈ −2.5 µm, `dev_DOE1` ≈ −3.5 µm
  (n_medium = 1.0). `print_DOE1` with n_medium = 1.508 (propagation in liquid resin): −4.0 µm.
  3 × 3 tiles: −2.4 … −3.1 µm, centre tile outlier −4.4 µm (weak contrast); no systematic tilt
  of the focal plane.
- The preset value `propagationDistance` = 0.11 µm does not match the measured focus
  (−2.6 µm); check and update (decision O1).

## 6. Open points to inspect

- Verify the new order with real data (object + background at several z).
- Alternative to test: propagate object **and** background field by z and divide afterwards,
  `P_z{E_obj} / P_z{E_ref}`. Not identical to the current approach; compare both with a known
  reference structure. → P16
- Where exactly is the hologram plane relative to the focal plane? Is the background recorded
  at the same focus position as the object? → O1
- Propagation medium: the 63x objective (NA 1.4) is an oil immersion objective. In which medium
  does the propagation take place (air above the structure, resin, glass)? Set
  `propagationMediumIndex` accordingly; the wavelength in the medium is `λ0 / n`. → O2
- Sign convention: `locate_order` always takes the first order in the left half of the
  spectrum. The chosen order determines the sign of phase and height (depends on the reference
  beam geometry). → O2
- Residual tilt: the carrier is removed by an integer pixel shift, leaving a residual tilt of up
  to 0.5 FFT bins. With background compensation it cancels; without compensation it remains in
  the phase.
- Option for later: numerical aberration compensation (Zernike / numerical parametric lens,
  Colomb et al., J. Opt. Soc. Am. A 23, 3177–3190 (2006)). → P17
