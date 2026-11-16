# MIRA — instructions and todos

Status: draft v0.1, 2026-11-15, waiting for maintainer approval. Read `00_SOFTWARE_PLAN_OVERVIEW.md` first.
Repository: `MIRA` (package `mira`, "Microscopy Image Registration & Analysis"), branch `main`.

## 1. Role

MIRA takes the outputs of the hologram evaluation and relates them to place and to the design.

1. **Registration:** place products by their stage position and refine the placement in the image.
2. **Speckle fusion:** combine captures of the same structure at shifted positions into one product
   with less speckle noise.
3. **Stitching:** combine a raster of products into a mosaic of a structure larger than the DHM
   field of view.
4. **Target comparison:** compare a product with the target structure and return a
   `DeviationReport` to nanofactorysystem.

The existing structure analysis (`StructureProcessor`, detection, segmentation, alignment,
`mira.analysis`) stays and is reused. Out of scope: hologram reconstruction (`offaxisholo`),
hardware, network, stage control, decisions about corrections.

## 2. State of the repository (2026-11-15)

- Three commits. Ten modified files and the whole `src/mira/analysis/` folder are uncommitted.
- Many parallel copies (`*_old*.py`, `processor_old^1.py`) and one Seafile conflict copy of
  `visualization/plotter_2d.py`.
- `pyproject.toml` lists dependencies and also declares them dynamic from a `requirements.txt` that
  does not exist; the author field is a placeholder; the README still describes `image_processing`.
- `tests/_test/test_dhm.py` contains no test function. The test suite used during the DOE3 work was
  not found in this folder.
- `processing/registration.py` already has `RegistrationRefiner`, `ECCRefinement` and
  `PhaseCorrelationRefinement`; `processor.apply_affine_transform` is the single gateway for geometry.
- No `CLAUDE.md`, no todo file. Notes: `offene_punkte_und_zernike_psd_wrapper.txt`,
  `zusammenfassung_aenderungen_und_tests.txt` in `src/mira/`.

## 3. Rules

1. Nothing is deleted. Old copies move to `attic/` only after the maintainer approves the list (MI-01).
2. Fix problems at their source, not downstream. Prefer a validation error over a silent fallback.
3. Every geometric transform of a processor goes through `apply_affine_transform`; raw channels,
   masks and segments are transformed together.
4. Computation and plotting stay separate (`mira.analysis`, `mira.processing` against
   `mira.visualization`). `import mira` must not import matplotlib (MI-05).
5. New functions for products are pure: arrays and metadata in, arrays and metadata out. No file
   access, no global state, backend-agnostic signatures (NumPy first).
6. Units in names (`_um`, `_rad`); heights in metres only where the existing API requires it, and
   converted at the boundary.
7. Results of the publication analysis must not change. Run the regression of MI-04 after every
   non-trivial change.
8. One deliverable per session, with a short integration note for the maintainer at the end.

## 4. Open decisions

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| M1 | Baseline: commit the current working state as it is, and what happens to the `*_old*` files and the conflict copy. | Commit as is on a branch, then move old copies to `attic/` after review. | MI-01 |
| M2 | Data model for products: extend `StructureProcessor`, or add a light `Product` type with an adapter. | New `Product` type; `StructureProcessor.from_product()` as bridge. | MI-10 |
| M3 | Averaging domain for speckle fusion: complex field, intensity, or phase (open in F5). | Complex field after equalising offset and tilt; intensity separately. Confirm with data. | MI-30 |
| M4 | Scope of "speckle noise optimisation": fusion of shifted captures only, or also weighting, selection and the choice of the shift pattern. | Fusion first; the pattern is planned by nanofactorysystem (T85). | MI-30 |
| M5 | Comparison domain: expected phase computed from the target, or measured height from the phase. | Phase domain; height as derived output. Depends on G4 and G7. | MI-51 |
| M6 | Who rasterises the target: nanofactorysystem delivers a raster, or MIRA rasterises a height map or outline onto the product grid. | nanofactorysystem delivers a height map in µm; MIRA resamples it. | MI-50 |
| M7 | Acceptance tolerances for registration and comparison. | Set after the noise floor is measured; until then tests use synthetic data with stated tolerances. | MI-22, MI-52 |
| M8 | GPU use on the DHM-PC (G5). | CPU first; port single kernels only after profiling. | MI-61 |
| M9 | Python and NumPy baseline (`requires-python >= 3.10` today). | Python 3.12, NumPy 2, as the other packages. | MI-02 |

## 5. Todos

### Phase 0 — baseline

- [?] **MI-01 Secure the working state (S).** Show the maintainer `git status` and a diff of the
  conflict copy against `plotter_2d.py`; propose the commit(s) and the list of files for `attic/`.
  Needs M1. Do not change code in this todo.

- [ ] **MI-02 Packaging (S).** Remove the conflict between static and dynamic dependencies, set
  author and description, extras `viz`, `analysis`, `dev`; editable install works in a fresh
  environment. Needs M9 for the versions.

- [ ] **MI-03 Project files (S).** `CLAUDE.md` (commands, architecture, the rules above, work log
  and commit conventions as in nanofactorysystem), `TODO.md` with this plan, `WORKLOG.md`; README
  updated to the real package structure.

- [?] **MI-04 Test baseline (M).** Ask the maintainer where the test suite of the DOE3 work lives
  and bring it into `tests/`. Add a small regression on stored snapshots for the analysis results.
  Done when: `python -m pytest` runs and its result is recorded as baseline in `WORKLOG.md`.

- [ ] **MI-05 Import hygiene (S).** `import mira` loads no plotting library; visualization is
  imported on use. A test checks it.

### Phase 1 — products and geometry

- [ ] **MI-10 Product data model (M).** `Product`: complex field or phase plus intensity, pixel size
  (y, x) in µm, stage position, z, wavelength, medium, quality, ids, material snapshot. Constructors
  from a `ProductSet` message, from an OffAxisHolo `ReconstructionResult`, from `experiment.h5`
  (`h5py` only). Bridge to `StructureProcessor`.
  Needs M2, G2. Done when: round trip against the shared example messages passes.

- [ ] **MI-11 Stage-to-image geometry (M).** Model that maps stage micrometres to product pixels
  (rotation, scale, axis flips, offset) per objective; estimation from captures with known shifts;
  save and load. Look at `tools/Transform` in nanofactorysystem for the camera case.
  Partial result allowed: model and estimation on synthetic data; real calibration needs captures.

### Phase 2 — registration

- [ ] **MI-20 Registration by position and image (M–L).** Coarse placement from positions and
  MI-11, then sub-pixel refinement restricted to the overlap (reuse the refiners in
  `processing/registration.py`). Returns transform and a confidence value; low confidence falls
  back to the stage position and is flagged.

- [ ] **MI-21 Phase offset and tilt between products (M).** Robust estimate of piston and linear
  phase terms from the overlap of two products.

- [ ] **MI-22 Synthetic test fields (S–M).** Generator for fields with known shift, piston, tilt and
  speckle-like noise. Used by MI-20, MI-21, MI-30, MI-40. Needs M7 for the final tolerances.

### Phase 3 — speckle fusion

- [ ] **MI-30 Fusion of a speckle set (M–L).** `fuse(products, method=...)`: register, equalise,
  average. Returns the fused product, the transform of every member and the noise in the reference
  region before and after. Needs M3, M4. Partial result allowed: complex and intensity averaging
  implemented, default left open.

- [ ] **MI-31 Noise metrics (S).** Shared functions for noise in a reference region and its spectrum
  (reuse ensemble statistics and PSD tools), so that fusion, N-study and comparison report the same
  number.

### Phase 4 — stitching

- [ ] **MI-40 Stitching (L).** Pairwise registration in the overlaps, global least-squares
  adjustment of positions, phase offsets and tilts over the whole raster, blending with weights that
  fall off towards the tile edge. Returns mosaic, tiles and transforms. A tile may itself be a fused
  speckle set.

- [ ] **MI-41 Stage-to-camera calibration from stitching (S).** The residual shifts of MI-40 update
  the geometry model of MI-11 (rotation, scale).

- [ ] **MI-42 Size limits (S).** Document and test memory use for a raster of k × k tiles; float32
  and complex64 throughout.

### Phase 5 — target comparison

- [ ] **MI-50 Target on the product grid (M).** Resample a `LayerTarget` (height map and mask in µm,
  position) onto the pixel grid of a product, using MI-11 and the detected pose of the structure
  (existing detection and alignment). Needs M6.

- [ ] **MI-51 Forward model (M).** Expected phase from target height, wavelength and the index
  difference of the material state; inverse for the height output. Reuse `calculate_delta_n` for
  calibration. Needs M5, G4, G7. Partial result allowed with the index difference as explicit input.

- [ ] **MI-52 Comparison (M).** `compare(product, target, *, sigma, k) -> DeviationReport`:
  deviation map, RMSE, MAE, statistics per segment, share of the area above the threshold k·sigma,
  classification too low / too high, validity flags (registration confidence, coverage, product
  quality). Needs M7.

- [ ] **MI-53 Stable entry point and report format (S–M).** `mira.api.evaluate_layer(product,
  target, context, settings) -> DeviationReport`; report as header dictionary plus arrays with
  `schema_version`; test against the shared example. This is what nanofactorysystem T90 calls.
  Reports the computing time.

### Phase 6 — integration

- [ ] **MI-60 Plug-in for the DHM service (M).** `mira.plugins.dhmlink`: `fuse_group(members,
  params)` and `stitch_group(members, params)` returning a `GroupProduct`. Pure functions, no
  network. Called by `dhmlink` DL-B9. Needs G5.

- [?] **MI-61 GPU kernels (M).** Only after profiling shows the need. Needs M8.

- [ ] **MI-62 Offline use (M).** Run fusion, stitching and comparison on `experiment.h5` and the raw
  archive, and write results under `/analysis` through the access layer of nanofactorysystem
  (`ExperimentFile`, optional dependency).

## 6. Suggested order

MI-01 → MI-02, MI-03, MI-04, MI-05 → MI-10 → MI-22 → MI-50, MI-51, MI-52, MI-53 (needed first, for the
operating modes `monitor` and `shadow`) → MI-11, MI-20, MI-21 → MI-30, MI-31 → MI-40, MI-41, MI-42 →
MI-60, MI-62 → MI-61 if needed.
