# Handover from OffAxisHolo to NanoFactorySystem_automated (2026-11-16)

For: the maintainer and Claude Code sessions working in NanoFactorySystem_automated. Source: the
OffAxisHolo work of 2026-11-16 (todos K1, B5, B3, K2–K7, K18), branch stack ending at
`feat/k7-background` (commit `b70838b`), not yet merged into `Holo_HR`.

What to do with this file: update the master copy of the system plan (section 1), take the
information for `MeasurementRecipe`, `CaptureContext`, `LayerTarget` and `experiment.h5` into the
nanofactorysystem todos (sections 2–5), and prepare the planning session K18 (section 4). Nothing
here changes a nanofactorysystem file.

## 1. Updates for the master system plan (`docs/system/SOFTWARE_PLAN_OVERVIEW.md`)

The master copy lives in this repository. Proposed changes (then copy unchanged into the other three):

| Item | Change |
|---|---|
| G2 | **Decided 2026-11-16 (maintainer, as recommended):** JSON Schema plus example messages in `dhmconnect/schemas/`; the other repositories vendor the examples as test fixtures. |
| G9 | Partly decided 2026-11-16: the maintainer keeps `.git`, `.venv` and `build/` of OffAxisHolo inside the Seafile folder knowingly (risks known). Python 3.12 stays the recommendation. |
| DHM-PC facts | The driver supports CUDA 13 (OffAxisHolo uses `cupy-cuda13x`). Python is not installed on the DHM-PC yet; it was only reached through the C server so far. All Python packages there need a fresh install (Python 3.12). |
| New topic | Reference mask, averaging and stitching: decisions SA1–SA7 (OffAxisHolo `docs/stitching_and_averaging.md`), to be decided in a cross-repository planning session (OffAxisHolo K18). Proposal: add them to §5 as global decisions, because they touch all four repositories. |
| §6 milestones | S1 progress: OffAxisHolo product interface K2–K7 done (K8 blocked by K18). Proposal: add the planning session K18 before S3 / S4. |

## 2. Decisions and defaults that affect the measurement recipe

- **Frame selection (O3):** default policy `all` until the N-study (OffAxisHolo K13) on real data is
  evaluated. Available policies and parameters for `MeasurementRecipe.selection_policy`:
  - `all`: every frame is averaged;
  - `threshold`: fixed bounds per metric, e.g. `{"saturated_count": (None, 100)}`;
  - `robust`: median / MAD outliers on `robust_metrics` (default `sideband_fraction`,
    `phase_noise_rad`) with factor `robust_k` (default 3.5);
  - `best_k`: the `k` best frames ranked by `rank_by` (default `phase_noise_rad`, ascending);
  - `min_frames` (default 1) is always enforced and reported.
- **Frame metrics** are on by default (per frame: saturation and black counts, intensity quantiles,
  sideband energy fraction and peak position, piston phase, phase noise).
- **Requested products:** `phase` and `intensity` always, `field` (complex) on request.
- **Number of frames:** free; recommendation for the first experiments: record with `keep_raw = all` so
  the N-study can set the default N and the selection policy from real data.
- **Background (O6):** a background never expires; an optional maximum age only adds a warning to the
  result. When to record a new background is a recipe / service decision (proposal: once per
  objective and session, plus after optical changes). Every result names its `background_id`.
- **z:** the propagation distance comes from the request (`CaptureContext` z hint) or the parameter
  set. It is open which plane is the result plane (G4 / O1): the Zeiss 63x preset says 0.11 µm, about
  −2.6 µm was measured.

## 3. What a result contains (for `ProductSet` and `experiment.h5`)

Draft schema 0.3 of `ReconstructionResult` (OffAxisHolo `realtime/result.py`; final contract in
`dhmconnect/schemas/`, DL-A3):
- arrays: `phase_rad` (float32, wrapped), `intensity` (float32, linear), optional `field` (complex64);
- `pixel_pitch_um` (after the sideband crop, not the camera pitch), `z_um`, `wavelength_um`,
  `n_medium`;
- one `FrameRecord` per frame (`index`, `used`, `reject_reason`, `metrics`, `acquisition` passed through
  unchanged);
- `quality`: `n_obj`, `n_effective`, `n_bg`, `piston_std_rad`, selection summary, `background_id`,
  `background_created_utc`, `background_age_s`;
- `timings_s` (stages and total), `config_hash`, `config`, `versions`, `warnings`, `created_utc`,
  `schema_version`.

Proposals for `experiment.h5` (single writer: nanofactorysystem):
- store `schema_version`, `config_hash`, `background_id` and `warnings` with every product, so a
  product can be traced to its background and parameter set;
- store the frame records (metrics and selection) next to the product; they are the input for the
  N-study and for later re-selection.
- OffAxisHolo K12 will read `experiment.h5` read-only (hologram stacks, device parameters, products)
  with `h5py` only and without HDF5 file locking, following `docs/design/EXPERIMENT_STORAGE.md` of this
  repository. Please keep that layout document current; a change there is a change for K12.

## 4. Open topic for the planning session K18: reference mask, averaging, stitching

Details: OffAxisHolo `docs/stitching_and_averaging.md`. What it means for nanofactorysystem:

- **Piston phase:** every camera frame carries a random constant phase offset (vibration, path
  changes). OffAxisHolo aligns the frames of one measurement before averaging. For honest noise numbers
  and a robust alignment it needs a structure-free **reference region per capture**; a fixed region per
  objective (former decision O4) does not hold, because every print has another structure and large
  structures are stitched.
- **A priori mask from the print job (SA1, the maintainer finds it very sensible):** nanofactorysystem
  knows before the print where the structure is (`LayerTarget`: height map and mask in µm, position).
  The expected footprint, dilated by a safety margin, gives the reference region as its complement.
  Needed for it:
  - the footprint of the structure at the time of the capture (after layer n) in stage µm, plus the
    commanded and actual stage position of the capture (already planned in `CaptureContext`);
  - a pose tolerance / safety margin (who sets it: recipe or material / process data?);
  - the mapping stage µm ↔ camera pixel (MIRA MI-41, MI-50).
- **A posteriori mask from MIRA:** Otsu detection on a first reconstruction (as MIRA does today when it
  sets the background to 0). Exact, but needs a second pass (SA3).
- **Levelling:** proposal SA5: OffAxisHolo only aligns the frames of one measurement; the absolute
  offset of a product and the offsets between tiles are removed by MIRA (MI-21, MI-40). Products in
  `experiment.h5` would then be aligned but not levelled; the levelled version comes from MIRA.
- **Two averaging levels:** within one stage position (N frames) in OffAxisHolo; across positions
  (speckle fusion) after registration in MIRA (MI-30). The recipe's group kind decides which applies.
- **Stitching:** every tile has its own mask and offset; unwrapping across tiles is open (SA6).

Decisions to prepare from the nanofactorysystem side: SA1 (source of the mask), SA2 (format and
coordinates in the contract; proposal stage µm, carried in `CaptureContext` or the recipe), SA5 (who
levels), and whether the `LayerTarget` footprint is available at capture time in the service.

## 5. Other open decisions with a nanofactorysystem part

| ID | Topic | Status |
|---|---|---|
| G4 / O1 / O2 | Result plane, z per objective, medium index, sign convention phase → height | open; blocks OffAxisHolo K9, K10 and MIRA MI-51 |
| G7 | Material registry as single source of refractive indices; snapshot in `CaptureContext` | recommendation unchanged; OffAxisHolo presets stay an offline fallback (K9, P3) |
| O11 | Mode "print" of the height calculation (n_medium = liquid resin): what should it compute? | open (OffAxisHolo P5) |
| O9 | Stage control is not part of the OffAxisHolo GUI but of the system GUI (F15) | recommendation, not decided |
| G3 / O7 | GUI framework: PySide6 with pyqtgraph | recommendation, not decided |

## 6. Requests to nanofactorysystem

1. Update the master system plan (section 1) and copy it into the other repositories.
2. **T86 / T87** (former realtime items F3, F4): integration with the control script and the reference
   measurement, done together with the GPU validation K14 on the DHM-PC.
3. **T75 / G7:** material registry with `n_medium` for the result (today from the OffAxisHolo preset).
4. **`MeasurementRecipe` (T71 and contracts):** selection policy fields as in section 2; default `all`.
5. **`CaptureContext`:** keep the z hint and the commanded / actual position; consider the structure
   footprint for the reference mask (SA1 / SA2, after K18).
6. **`experiment.h5` layout:** store the fields of section 3; keep `docs/design/EXPERIMENT_STORAGE.md`
   current for OffAxisHolo K12.
7. **Planning session K18:** bring the `LayerTarget` view (availability at capture time, coordinate
   system, safety margin).

## 7. References in OffAxisHolo

- `docs/stitching_and_averaging.md` (SA1–SA7), `docs/reconstruction_order.md`.
- `src/offaxisholo/realtime/result.py`, `selection.py`, `metrics.py`, `background.py`, `api.py`.
- `TODO.md` (open decisions table; K8, K9, K12, K13, K14, K18), `WORKLOG.md` (entries of 2026-11-16).
- Companion note for DHMConnect: `docs/handover/to_dhmconnect_2026-11-16.md`.
