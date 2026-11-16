# Stitching and averaging: reference region, piston phase and combination of holograms

Status: open problem collection, 2026-11-16. Written at the end of the session that built the
product interface (K2–K7) because todo K8 (quality numbers) ran into it. To be worked on in a
cross-repository planning session (todo K18). No decision in this document is taken yet; the
decisions are listed in section 5.

Repositories involved: OffAxisHolo (this one), MIRA (detection, fusion, stitching),
nanofactorysystem (print job, `LayerTarget`), DHMConnect (contracts, service).

## 1. Terms

- **Piston phase** (German: Kolbenphase, constant phase offset): a phase offset that is the same
  for every pixel of a frame. Vibration and small optical path changes between the object and the
  reference arm shift the whole phase of every camera frame by a random amount.
- **Piston correction (per frame)**: before averaging, the realtime kernel estimates the offset of
  each frame against the background field (`p = Σ U · conj(Ub)` over a region) and removes it.
  Without it, averaging N complex fields with random offsets loses amplitude (factor
  `exp(−σ²/2)`) and the noise does not drop as `1/√N`.
- **Reference region / reference mask**: the pixels used for this estimate and for the phase noise
  measure. Ideal: only substrate, no structure.
- **Phase levelling (after averaging)**: setting the background of the final phase map to zero,
  as MIRA does today (Otsu threshold → structure / background → background offset removed). This
  fixes the absolute offset of one product.

## 2. What exists today

**OffAxisHolo (realtime kernel, state of this branch stack)**
- Per-frame piston correction against the background field, on by default
  (`CompensationConfig.piston`). Region: the whole field, or a border ring of `ref_border` pixels
  if configured.
- Per-frame metrics (K2) contain `piston_rad` and `phase_noise_rad`; the noise is NaN without a
  reference region (no ring configured).
- Block mode can keep all corrected fields of a measurement (`keep_fields`, K3), so a later
  re-alignment with a better region is technically possible without new frames.
- One background model per configuration (K7); it does not depend on the reference region.

**MIRA**
- Structure detection with Otsu threshold (`mira/processing/detection.py`), background set to 0.
- Planned: phase offset and tilt between products from their overlap (MI-21), speckle fusion
  (MI-30, decision M3: complex field after equalising offset and tilt), stitching with a global
  least-squares adjustment of positions and phase offsets (MI-40), `LayerTarget` resampled onto the
  product grid (MI-50).

**nanofactorysystem**
- Knows before the print where the structure is and what it should look like (`LayerTarget`:
  height map and mask in µm, position).

## 3. Findings

1. **Two different offset problems.** The per-frame alignment before averaging (OffAxisHolo) and
   the levelling of the final map (MIRA) are different steps.
   - For the alignment only the *relative* offsets between the frames of one measurement matter.
     The estimate over the whole field is biased by the structure, but the bias is the same for all
     frames as long as the structure does not change during the measurement (about 16 frames,
     a fraction of a second). Then the whole-field estimate is good enough.
   - It fails when the structure changes during the measurement (printing while measuring), or
     when the object signal dominates the field and the structure moves relative to the camera.
   - The absolute offset of the result is then fixed afterwards by MIRA's levelling. MIRA's Otsu
     approach therefore already solves the absolute part.
2. **No fixed region per objective.** Every print has another structure; large structures reach the
   border of the field of view. A fixed mask per objective (decision O4 of 2026-11-16) or a border
   ring as fallback does not hold in general. The reference region must come **per capture**.
3. **Two sources for a per-capture mask**, which complement each other:
   - **A priori from the print job**: the expected footprint from `LayerTarget`, dilated by a safety
     margin, its complement is the reference region. Available before the reconstruction, so no
     second pass. Needs the mapping stage µm ↔ camera pixel (MI-41, MI-50) and a pose tolerance.
   - **A posteriori from MIRA**: Otsu detection on a first reconstruction. Exact, but a chicken-and-egg
     loop: reconstruct without mask → detect → re-align with mask. Works with the kept fields
     (`keep_fields`) but costs time and needs the fields to live until MIRA answers.
4. **The phase noise measure needs a mask.** The noise in a structure-free region is the honest
   quality number of a product (K8, N-study K13). Without a mask it stays NaN (today's behaviour).
5. **Stitching makes everything per tile.** Each tile of a large structure has its own structure
   coverage, so its own mask; some tiles may have almost no free substrate. Tiles get independent
   piston offsets; MI-21 / MI-40 adjust them in the overlaps. The levelling of the mosaic then
   happens once, globally, in MIRA.
6. **Averaging happens on two levels**, and the boundary must stay sharp:
   - within one position (same stage position, N frames, complex average, piston-aligned):
     OffAxisHolo;
   - across positions (speckle fusion, shifted captures): needs registration first, then complex
     averaging after equalising offset and tilt: MIRA (MI-30, M3).
7. **Background model and tiles.** The background compensates the fixed optics; it is valid for all
   tiles of one configuration (K7). A substrate tilt that varies across a large field is not covered
   by it and appears as a tile-dependent linear phase, which MI-21 / MI-40 must remove.
8. **Unwrapping across tiles** is open: unwrapping per tile before stitching can leave 2π steps
   between tiles; unwrapping after stitching needs the complex or wrapped mosaic (relates to O5,
   K11).

## 4. Problems to solve

- P-A: How does a per-capture reference mask reach the kernel (contract, format, coordinates)?
- P-B: When is the mask available relative to the reconstruction (before, or only after a first
  pass)? What does a second pass cost on the DHM-PC?
- P-C: What is the fallback without a mask, and how is the result flagged?
- P-D: Who removes which offset: per-frame alignment (OffAxisHolo), tile offsets (MIRA MI-21 /
  MI-40), absolute levelling (MIRA)? Is the result of OffAxisHolo levelled at all, or only aligned?
- P-E: How are the noise and quality numbers of a tile defined when its free area is small?
- P-F: Where does unwrapping happen for stitched products?

## 5. Decisions needed (to be taken in the planning session K18)

| ID | Decision | Options | Proposal (not decided) | Affects |
|---|---|---|---|---|
| SA1 | Source of the reference mask per capture | a priori from `LayerTarget`; a posteriori from MIRA (Otsu); both | Both: a priori by default, MIRA refines when time allows | K8, K18, T…, MI-50 |
| SA2 | Format and coordinates of the mask in the contract | pixel mask on the crop grid per capture; polygons / raster in stage µm | Stage µm in the contract (resolution independent), rasterised in OffAxisHolo with the calibrated pixel size | G2 / DL-A3, K9, MI-41 |
| SA3 | Two-pass flow (re-alignment after MIRA detection) | never; on request; always | On request only, with `keep_fields`; measure the cost in K14 first | K3, K18, DL-B4 |
| SA4 | Fallback without a mask | whole field (today); border ring; automatic (pixels with small phase change against the background) | Whole field for the alignment, noise = NaN and a flag in the result | K8 |
| SA5 | Split of offset handling | as in P-D | OffAxisHolo aligns frames only; MIRA levels and adjusts tiles | K8, MI-21, MI-40 |
| SA6 | Unwrapping of stitched products | per tile; after stitching | Decide with O5 / K11 after the first stitched data | O5, K11, MI-40 |
| SA7 | Replacement of O4 | – | O4 ("mask per objective") is superseded by SA1 / SA2 | K8 |

## 6. What OffAxisHolo can build once SA1, SA2 and SA4 are decided

- A per-measurement reference mask argument in both entry points (`reconstruct_frames`,
  `ReconstructionSession`), used for the piston estimate and the noise measure (K8).
- Optional re-alignment of the kept fields with a mask delivered later (SA3).
- Quality flags in the result: which region was used (mask source, whole field, none).

Nothing of this is built yet; the kernel keeps today's behaviour (whole field, noise NaN without a
region).
