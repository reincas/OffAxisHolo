# Software plan — real-time feedback 2PP system: overview

Status: v0.4, 2026-11-18. v0.3 (2026-11-16) plus the results of the decision session of 2026-11-17/18
between the maintainer and Claude: all global decisions G3–G10 are taken, the distribution across the
PCs is new (G5-new with N1–N7), the group protocol and the stitched `LayerProduct` are defined, and the
target is **Release V1.0** (= everything decided in that session). Top criterion for V1.0: **works
reliably**; then fast computation (little idle time) and efficient use of the hardware.

This file exists as an identical copy in all four repositories under `docs/system/`.
**Master copy:** `NanoFactorySystem_automated/docs/system/SOFTWARE_PLAN_OVERVIEW.md`. Change the master
first, then copy it unchanged into the other three repositories (one `docs(system): …` commit each).
The full minutes of the session (reasons, numbers, answers to side questions) are kept outside Git in
the maintainer's `_system/DECISIONS_SESSION_2026-11-18.md`; everything that binds the code is in this
file and in the "Decided" tables of the four `TODO.md` files.

| Repository | Python package | Branch base | Todo IDs | Where the todos live |
|---|---|---|---|---|
| OffAxisHolo (upstream `reincas/OffAxisHolo`) | `offaxisholo` | `Holo_HR` | `K…`, `L…`, `P…`, carried over `B…`, `C…`, `H…` | `TODO.md` |
| DHMConnect | `dhmconnect` | `main` | `DL-…` | `TODO.md` |
| NanoFactorySystem_automated | `nanofactorysystem` | `main_HR` | `T…`, `F…` | `TODO.md`, `future_todo.md` |
| MIRA | `mira` | `main` | `MI-…` | `TODO.md` |

Decision IDs: global `G…`; repository decisions `M…` (MIRA), `O…` (OffAxisHolo), `D…` (DHMConnect);
design documents of nanofactorysystem: `SI1–SI4` (`DHM_SERVICE_INTEGRATION.md`, formerly D1–D4),
`OM1–OM5` (`OPERATING_MODES.md`, formerly O1–O5), `C1–C5` (`CAMERA_LIVE.md`), `P1–P3`
(`GUI_READINESS.md`); reference mask and stitching `SA1–SA7`; distribution `N1–N7`; clean-up `X-A…X-J`.

## 1. Packages and ownership

| Package | Runs on | Owns | Must not contain |
|---|---|---|---|
| `offaxisholo` | DHM-PC (inside the service), any PC (offline, GUI) | Everything about **one hologram stack**: frame metrics and selection, demodulation, averaging of the frames of one position, compensation, propagation, phase and intensity, quality numbers, timings → `ReconstructionResult` (wrapped phase). Phase ↔ height formulas (`fieldops`), rasteriser helper for reference masks, parameter-set schema and saveable configurations (O11). Offline analysis and the GUI (`offaxisholo.gui`, extra `[gui]`), a pure observer that takes only images as input (O8, O9). | Network server, hardware drivers, stage or position logic, stitching, fusion, levelling, target comparison. Operating modes such as "print" (dropped, O11). |
| `dhmconnect` | Service on the DHM-PC, client on the Work-PC and in GUIs | Protocol v1, client, DHM service process (state machine, jobs, hardware gateway on the Koala SDK, calibration procedures, groups, result store, raw and tile archive, live socket), slot for post-processing plug-ins, worker processes, **home of the shared contracts** (`dhmconnect/schemas/`, G2). | Reconstruction math (calls `offaxisholo`), registration / stitching math (calls `mira`). Never writes `experiment.h5`. Never moves the stage. |
| `nanofactorysystem` | Work-PC | Orchestration of the print, information model (material registry = master of refractive indices, substrate, experiment, structure), operating modes, measurement recipes, **capture-position plan generator** (`GroupSpec`), target rasteriser (`LayerTarget`), calibration procedure writing field ↔ camera ↔ DHM, focus determination from the corner markers, later the controller (F16), `experiment.h5` (single writer), camera live feed, system GUI incl. DHM + stage operation (F15). Today also the legacy DHM path (protocol v6). | Reconstruction, registration, stitching. |
| `mira` | DHM-PC (plug-in in the service: registration, speckle fusion, stitching, unwrapping per tile, detection, levelling) and Work-PC (`evaluate_layer`, offline analysis) | Product types `TileProduct` / `LayerProduct`, registration by stage position + image, piston/tilt adjustment, speckle fusion and stitching (one pipeline, two configurations), comparison with the target → `DeviationReport`, detection + fit of the calibration marks (MI-41). The existing statistical analysis of the DOE publication. | Hardware, network, hologram reconstruction. |

## 2. Machines and distribution (G5-new)

| Machine | Role | Runs |
|---|---|---|
| **DHM-PC** (HP Z2 G9, i9-14900K 32 threads, 64 GB, RTX A2000 12 GB, Windows 11, LyncéeTec DHM, Basler camera, OPL motor, lasers) | **Compute node** | `dhmconnect` service (acquisition, gateway, calibration), `offaxisholo` (one GPU owner thread), `mira` plug-in (CPU worker pool), store until `ack`, raw / tile archive |
| **Work-PC** (i5-10600K, Windows sees 5 CPUs because of INtime, 32 GB, no CUDA GPU, Windows 10, Aerotech A3200, Femtika laser, MatrixVision camera) | **Control node** | `nanofactorysystem` (orchestration, spec, slicing, A3200 axes/galvo/laser, camera feed and video, `experiment.h5`), `dhmconnect.client`, `mira.evaluate_layer` (G6), controller (F16), GUIs |
| **Pre-release PC** (office) | Development | editable installs of all four repositories in Seafile, tests with dummy backend / simulation / replay |

- **N1 – result per structure and layer:** the stitched `LayerProduct` (contract `GroupProduct`) is the
  default transfer: unwrapped phase (float32, 2π resolved through the overlaps) + `intensity_rel` (float32)
  of the whole structure on one common grid; the complex field of the same mosaic on request. Plus
  validity mask, grid (`origin_um` in stage coordinates, `pixel_um`, `shape`), stitching info per tile
  (planned / actual position, registration, piston/tilt, anchor, residual, discarded border band),
  quality (`reference_source` / `reference_fraction` per tile, noise, unwrapping quality), metadata (z,
  indices, background ids, config hashes, versions, timings). Tile products (`ProductSet`) and raw frames
  stay on the DHM-PC and are referenced. Sizes: one field of view 512² ≈ 2 MiB; a 300 µm structure ≈ 1725²
  px ≈ 23 MiB per channel; its raw data ≈ 780 MiB per layer.
- **N2 – inbound group protocol:** per structure and layer `open_group(GroupSpec)` with structure info
  (id, `occupied_um`, target reference, material snapshot), an ordered position list (ring, then
  inwards), parameters per position (frames, shift pattern, exposure) and stitching parameters. Per
  position the Work-PC moves the stage and sends `measure(CaptureContext with actual position, tile
  index)` and moves on right after `frames_acquired`. Pipeline: while the stage moves to i+1 the DHM-PC
  reconstructs and stitches tile i incrementally; `close_group` → `LayerProduct`. Background tiles are a
  group of their own before the print. The DHM-PC never moves the stage; the list is a plan.
- **N3:** MIRA runs on both PCs (see §1).
- **N4 – DHM-PC resources:** one GPU owner thread for reconstruction; CPU worker pool for MIRA
  (configurable, reserve for acquisition / Koala); store and working data on NVMe (C:); raw and tile
  archive on the HDD (D:, 1.8 TB free); check the GPU TDR events (LiveKernelEvent 141), test TCC mode,
  keep kernels short.
- **N5 – relieve the Work-PC:** nothing heavy in the print process; a compute-heavy controller can later
  be offloaded to the DHM-PC through `call` (DL-A5); Intel QuickSync for H.264; plan disk space (≈ 205 GB
  free); do not upgrade Windows 10 without consulting Aerotech / INtime; isolate the network.
- **N6 – network:** 1 GbE direct link (192.168.22.1 Work-PC ↔ 192.168.22.2 DHM-PC) is the bottleneck;
  uncritical with N1; DL-H3 measures; a 2.5/10 GbE card in the DHM-PC is optional later.
- **N7 – data on the DHM-PC:** store (until `ack`), archive (raw frames by `keep_raw`, tile products,
  fields at z = 0, orphan results), background tiles. Only acknowledged data of a closed experiment are
  deleted, after a retention period or manually; fill-level warnings; a tool lists and cleans per
  experiment.

## 3. Data flow (V1.0)

```
nanofactorysystem (Work-PC)                         dhmconnect service (DHM-PC)
  plan generator --open_group(GroupSpec)----------->  group state
  stage to pos i --measure(CaptureContext, i)------>  acquire frames --frames_acquired--> (stage to i+1)
                                                      offaxisholo: ReconstructionResult (tile i)
                                                      mira plug-in: register, unwrap, piston/tilt,
                                                                    fuse / stitch incrementally
                 --close_group--------------------->  LayerProduct (+ ProductSets in the tile archive)
  experiment.h5 <--result (LayerProduct), ack-------
  LayerTarget + LayerProduct --> mira.evaluate_layer --> DeviationReport --> experiment.h5
  controller (F16, later) --> CorrectionPlan --> next layer
```

Waiting rule (SI2 / OM2): the print waits for reconstruction + evaluation only when the next layer needs
the data (`correct`, or `wait_for_evaluation = true` per structure); otherwise the next layer prints after
`frames_acquired`. **Today** only the legacy path exists: `DhmClient` (protocol v6) → `DhmServ.exe` → raw
holograms → offline reconstruction with OffAxisHolo → offline analysis with MIRA.

## 4. Dependency rule

```
nanofactorysystem    --> dhmconnect.client, mira (both optional at import time)
dhmconnect.server    --> offaxisholo, mira (mira optional)
offaxisholo.gui      --> dhmconnect.client (optional extra, only as one "image source" adapter)
mira                 --> offaxisholo (optional; fieldops formulas imported or mirrored with a parity test)
offaxisholo core     --> none of the others
```

No cycles. The test suite of every package passes without the other three installed (optional imports,
fakes, vendored example messages).

## 5. Shared contracts (home: `dhmconnect/schemas/`, DL-A3)

| Contract | Produced by | Consumed by | Content (V1.0) |
|---|---|---|---|
| `ParameterSet` | offaxisholo (schema), loaded by dhmconnect `configure` | all | named, saveable JSON configurations (O11): camera (pixel pitch, ROI, bit depth, exposure as metadata), objective (magnification, NA, pixel pitch in the object plane — never from Koala), reconstruction (default z with its `propagation_index`, `OrderCalibration` = stored position of the first order, mask shape and size, crop size, `eps`), `phase_sign`, wavelength 0.674994 µm, unwrapping method, `optical_margin_um`; identified by id + `config_hash` |
| `CaptureContext` | nanofactorysystem | dhmconnect, stored with every product | experiment UUID and label, substrate, session, structure, phase, layer, capture id, image index, group id, tile index, commanded and actual position, `occupied_um` polygons (stage µm; `uint8` raster as alternative), `pose_tolerance_um`, material snapshot (indices per state with `wavelength_nm`), `n_surround`, `propagation_index`, z hint (`structure_top` optional) |
| `MeasurementRecipe` | nanofactorysystem | dhmconnect | frames per position, shift pattern, selection policy, exposure policy, `keep_raw` (V1.0 default `all`), `publication` flag (keep and transfer all raw data to the Work-PC after the print), requested products (complex field on request), `reference_exclusion` (`all_printed` default / `since_background`), unwrapping method, post-processing steps with a free parameter dict |
| `GroupSpec` | nanofactorysystem plan generator | dhmconnect, mira plug-in | structure info, ordered position list with per-position parameters, stitching parameters, jitter seed (N2) |
| `ReconstructionResult` | offaxisholo | dhmconnect, GUI, mira | wrapped phase, `intensity_rel` (= I_object / I_background, 1.0 = empty substrate), optional complex field, pixel pitch after crop, z + `propagation_index`, wavelength, frame records, quality (`reference_source`, `reference_fraction`, `phase_noise_rad` = `null` without a mask), timings, `config_hash` + parameter-set id, versions |
| `ProductSet` | dhmconnect | mira plug-in, tile archive | `ReconstructionResult` unchanged + `CaptureContext` + job ids + `background_id` + `raw_ref` (host, path, SHA-256, frame count); internal per-capture product on the DHM-PC, transferred on request only |
| `LayerProduct` / `GroupProduct` | dhmconnect (computed by the mira plug-in) | nanofactorysystem, mira, GUI | see N1; references its member `ProductSet`s by id |
| `LayerTarget` | nanofactorysystem | mira | expected geometry after layer n in µm on a grid `pixel_um = min(half hatch distance, product pitch)`, exact masks from the footprint polygons, position, material, `domain` (`phase` / `height`); sources: toolpath + voxel model, footprint only (with a warning), later `design_heightmap` (grey-value height map) |
| `DeviationReport` | mira | nanofactorysystem | deviation map, RMSE, MAE, sigma, validity, registration transform, `domain`, versions; stored under `/analysis/captures/<id>/deviation`, `source = "nanofactorysystem.feedback"` (OM5) |
| `CorrectionPlan` | nanofactorysystem controller | nanofactorysystem planning | later, see F16 |

Conventions for all four packages: English; units in names (`_um`, `_rad`, `_s`, `_us`, `_mw`,
`_nm`); float32 / complex64, C order, little-endian; image axis 0 = y, axis 1 = x; **pixel sizes and
positions at every contract boundary as `(y, x)` in µm** (X-G; MIRA converts to its internal `(x, y)` in m
in exactly one place, `StructureProcessor.from_product()`); timestamps UTC ISO 8601; every message carries
`schema_version`; every message carries `config_hash` + parameter-set id, the full configuration travels
once per hash and nanofactorysystem stores it once per hash in `experiment.h5` (DL-A3 a); never `pickle`
across a process or network boundary; **no silent defaults** for physical parameters (a missing index is
an error or a warning that names the source).

## 6. Global decisions

All global decisions are taken. Claude Code: do not decide the deferred items of §7. Ask the maintainer,
or implement only the part that does not depend on them (see the partial-work rule in §9).

| ID | Decision | Result |
|---|---|---|
| G1 | Home and name of the communication code | **Decided 2026-11-15:** own repository `DHMConnect`, package `dhmconnect`, todo IDs `DL-…`. |
| G2 | Home of the shared contracts | **Decided 2026-11-16:** JSON Schema + example messages in `dhmconnect/schemas/`; the others vendor the examples as test fixtures. |
| G3 | GUI framework | **Decided 2026-11-18:** PySide6 + pyqtgraph for all GUIs (system GUI, `offaxisholo.gui` with extra `[gui]`); migrate the PyQt5 legacy (`DHM_User_Interface`) when touched; napari optional for the offline analysis GUI (evaluated in L0). |
| G4 | Result plane, z, indices, sign | **Decided 2026-11-18:** separate parameters `propagation_index` and `n_surround`; z always stored with its `propagation_index`; `n_structure` from the material snapshot. In-situ only resin indices (surround = liquid resin, structure = cured in resin); developed: surround = air. **Focus:** after writing the corner markers a DHM capture, z from a z sweep with a sharpness metric (e.g. minimum of std/mean of the amplitude) → `/calibration/dhm/focus`, default z of the experiment; fallback z per objective from the parameter set, provisional 63x **−2.6 µm at n = 1.0** until the focus study K10. Per-capture autofocus is measured and reported in `quality`, not applied until validated. In-situ experiments keep the compensated field at z = 0 in the tile archive. Sign: more optical path → positive phase; n_structure > n_surround → positive height; `phase_sign` (±1) in the parameter set, verified once against Sensofar/AFM; publication numbers must not change. Comparison domain = OM4. |
| G5 | Where fusion and stitching run | **Decided 2026-11-18 (G5-new):** compute on the DHM-PC (worker processes, MIRA on the CPU first), control on the Work-PC; N1–N7 in §2. |
| G6 | Where the target comparison runs | **Decided 2026-11-18:** Work-PC. |
| G7 | Material data | **Decided 2026-11-18:** material registry in nanofactorysystem = master; snapshot in `CaptureContext` and `experiment.h5`; versioned JSON export for offline use, OffAxisHolo material presets are generated from it; states at least `uncured`, `cured_in_resin`, `cured_developed`, each with `wavelength_nm`; one DHM wavelength 0.674994 µm. SZ2080 at 674.994 nm (maintainer's measurement): uncured n = 1.508, cured n = 1.52 (Δn = 0.012, cured value checked against air). Publication scripts set n = 1.52 explicitly. |
| G8 | Hardware access | **Decided 2026-11-18:** V1.0 via the Koala SDK. Binding after a short test in DL-H1: (a) pythonnet in-process (`PYTHONNET_RUNTIME=coreclr`) if loading, login and grabbing are stable in a long run; otherwise (b) a local C# gateway built from DhmServ's Koala code (`legacy/DhmServer/…/Microscope.cs`), fixed (init loop, laser off on disconnect, no `0.0.0.0`), behind `HardwareGateway`. No pypylon in V1.0. |
| G9 | Python and installation | **Decided 2026-11-18:** development working copies in Seafile only on the pre-release PC; all other PCs install the four packages from GitHub (tagged releases such as `v1.0.x`, pinned by a lock file per PC, `uv lock` or pip-tools; venvs under short paths outside Seafile). DHM-PC: **Python 3.14** via the Python install manager, venv/lock with `uv`; Work-PC and pre-release PC follow after checks (mvIMPACT, NumPy 2). Repositories: `requires-python >= 3.12`, tests on 3.12 and 3.14. Mixed versions are safe (JSON + raw buffers). |
| G10 | Reference mask, averaging, stitching | **Decided 2026-11-18**, see SA1–SA7 below. |

**G10 in detail (SA1–SA7):**
- **SA1:** a-priori mask from the print program / 3D data (system coordinates, stored with every capture)
  is the default; MIRA detection is check and fallback.
- **SA2:** `occupied_um` polygons (all printed matter in the field of view) in stage µm in
  `CaptureContext`; margin = `pose_tolerance_um` (context, nanofactorysystem) + `optical_margin_um`
  (parameter set, OffAxisHolo default); the service rasterises with an OffAxisHolo helper. Calibration
  writing field ↔ camera ↔ DHM: print a cross + square; cross → offset of the centres, square →
  rotation / scale / shear; additionally the cross at known stage offsets (e.g. 3×3) for an affine
  least-squares fit; per objective, camera and DHM separately; result = versioned calibration file
  loaded with the parameter set, id + copy in `/calibration/dhm/mapping`; valid until the hardware is
  touched; manual recalibration at any time; optional quick check at the start of an experiment. Owners:
  nanofactorysystem = procedure, MIRA = detection + fit (MI-41). Until then: rotation 0, nominal pixel
  size, generous tolerance.
- **SA3:** no second alignment pass in v1; MIRA does detection, levelling and noise on the product.
- **SA4:** ladder a priori → MIRA detection → `whole_field` with `reference_fraction = 0`;
  `reference_source` ∈ {`mask_a_priori`, `mask_a_posteriori`, `border_ring`, `whole_field`},
  `reference_fraction`, `phase_noise_rad` = `null` without a mask, warning below
  `min_reference_fraction`.
- **SA5:** OffAxisHolo only aligns the frames of one measurement; the MIRA plug-in on the DHM-PC levels
  and adjusts tiles (offsets / transforms in the product, reversible).
- **SA6 – stitching algorithm (maintainer):** field of view ≈ 85–90 µm; from ≈ 160 µm some tiles have no
  substrate. First a ring of tiles around the structure with ≈ 50 % substrate (every ring tile anchored to
  the substrate), then inwards with large overlaps; the tile measured before is ground truth. Discard a
  border band of every tile after propagation (width to be studied; estimate |z|·tan θ_max with
  sin θ_max = NA/n, e.g. 63x NA 1.4 oil at z = −2.6 µm, n = 1.508: ≈ 6.5 µm ≈ 75 px). Minimum overlap = 2 ×
  border band + alignment strip. Stage x/y is the prior; image registration refines within the pose
  tolerance. Unwrapping per tile before stitching: piston from the complex overlap (angle of
  Σ U_new·conj(U_ref)), integer k·2π from the unwrapped difference, consistency check (constant difference,
  otherwise flag the tile / guided unwrapping). Finally a global least-squares adjustment of all overlaps
  with the ring tiles as absolute anchors. Option `reference_exclusion`.
- **SA7:** "mask per objective" (old O4) is dropped. Background tiles are measured at the same raster
  positions as the later object tiles, before the print (SI3).

**Decisions of 2026-11-18 that bind more than one repository:**

| ID | Result |
|---|---|
| Protocol v1 (DL-A2; D2, D3, D9) | Approved with four additions: stable `client_id` in `hello` (re-grant of the controller role after a reconnect; `ZMQ_ROUTING_ID` + `ZMQ_ROUTER_HANDOVER`); `ZMQ_ROUTER_MANDATORY=1`; separate live PUB socket on port **27185** with SNDHWM ≈ 2; ZMTP heartbeat on DEALER/ROUTER and a `result` timeout that depends on the announced size. Ports: 27183 commands, 27184 events, 27185 live; bind to the direct-link address only (tests on 127.0.0.1); no authentication on the closed link (D5). One controller + observers (D4). |
| SI1–SI4 (T71) | Recipe `off` never connects to the DHM. SI1 circuit breaker (reconnect with backoff at layer boundaries); in-situ: on connection loss reconnect, if that fails abort the print with a clear reason, resumable via `restart_experiment`; `continue` stays an explicit recipe option. SI2 waiting depends on the job (see §3); fallback: poll `job_status` if `frames_acquired` does not arrive within `n_frames × exposure + margin`. SI3 background per structure at the structure's own position before it is printed; `max_age_s` = 3600 s, warning only. SI4 orphan results are never deleted automatically; on restart / next connection `LayerProduct`s are fetched into `experiment.h5` (marked `recovered`) and acknowledged, tiles and raw data stay on the DHM-PC and `/dhm_orphans` holds references. |
| OM1–OM5 (T72) | OM1 toolpath + voxel model, else footprint only with a warning (`shadow` allowed with a footprint target). OM2 `monitor`/`shadow` do not wait; optional `wait_for_evaluation` per structure (always true in `correct`). OM3 warn only, stopping opt-in. OM4 flexible comparison domain (automatic, phase preferred; can be fixed to `phase` / `height`). OM5 reports under `/analysis/captures/<id>/deviation`. |
| C1–C5 (T73) | T102 is a hard prerequisite of T93. 5 fps default; MP4 `mp4v` default (documentation only), lossless `h5` opt-in, AVI/MJPG dropped, export to H.264 MP4 (or `avc1` directly if T94 shows it works; QuickSync on the Work-PC). C3–C5 as proposed. The camera overview before/after the print keeps running through the camera service, with an optional downscale factor. |
| P1–P3 (T74) | P1 runner in a worker thread (serialisable events). P2 `stop_now()` yes (no safety function); a layer stopped mid-program is never reprinted automatically, the structure counts as `stopped`, resuming is opt-in (`resume_stopped=True`). P3 pause between layers, warning threshold `max_pause_s`, optional interface / focus check after a long pause. |
| D6 | Calibration (exposure, OPL scan) runs in the service on the DHM-PC; the legacy procedures on the Work-PC (`optimage` / `motorscan`) stay selectable. |
| D7 | Store and forward with `ack`. V1.0 `keep_raw = all` (also training data); raw data stay on the DHM-PC (D:), `raw_ref` in `experiment.h5`; no automatic deletion in V1.0; warning at 80 % fill level, automatic downgrade to `selected` at 95 % with a clear message. Option `publication`: keep everything and transfer to the Work-PC after the print (background, never during the print; target configurable). After the studies the default becomes `selected`. |
| D8 | Group post-processing in worker processes on the DHM-PC. |
| D10 | Service start in V1.0 via Task Scheduler at logon of a lab user (interactive session for Koala), watchdog restart, single-instance lock, never together with legacy `DhmServ` at the hardware, manual start/stop scripts; disconnect RDP, never log off. Windows service only if DL-H1 shows Koala works in session 0. |
| DL-A3 a–c | a) `config_hash` + parameter-set id in every message, full config once per hash (also in `experiment.h5`); b) `intensity` → `intensity_rel`; c) `ProductSet` wraps `ReconstructionResult` unchanged (internal tile product on the DHM-PC); the stitched `LayerProduct` is the default transfer. |
| M2 | Immutable, backend-neutral types `TileProduct` (one capture, complex64 field) and `LayerProduct` (mosaic) in MIRA with `from_message` / `to_message`; complex field as a regular input. |
| M3, M4, T85 | Fusion in the complex field; stitching already gives speckle reduction → one plan generator in nanofactorysystem, one algorithm in MIRA (fusion and stitching are two configurations of one pipeline). Rule: **at least 9 positions per structure** with the DHM regardless of size; `single_shot` stays possible; no rigid pattern for large structures but a small random jitter (seed stored). Start values (configurable, replaced after the study): `min_positions` 9, `frames_per_position` 10, `min_coverage` 4, step for small structures ≈ 10 % of the field of view (≈ 9 µm at 63x), overlap ≥ 2 × border band + 10 µm, jitter ± 1 µm (63x) / ± 3 µm (20x). |
| M6 | nanofactorysystem rasterises at least as fine as the product (pure function); MIRA only shifts, transforms and downsamples (area-weighted), never upsamples; later target source `design_heightmap` with explicit grey → height mapping. |
| O5 | A single `ReconstructionResult` stays wrapped; unwrapping in the group step per tile with `fast2d`, k·2π through the overlaps; methods `fast2d`, `least_squares`, `none`, `temporal` per recipe; quality in the product. Printed structures are ≤ 15 µm high → in-situ with SZ2080 no 2π ambiguity is expected; `temporal` low priority (test plan K11-T in OffAxisHolo). |
| O8, O9 | Live view in the service (reduced images, binning, rate limit, live socket, below measurement jobs). The OffAxisHolo GUI runs on the Work-PC and the DHM-PC, takes only images ("image source" adapters: file/replay, dhmconnect live client, other devices) and never controls a stage; DHM + stage operation belongs to the system GUI (F15). |
| O11 | OffAxisHolo modes "print" / "developed" are dropped; explicit parameters and named configurations (`ParameterSet`) instead. Removing `mode` from the offline API is a BREAKING CHANGE; the golden reference is kept by a parameter set with the old values. |

## 7. Deferred points

| Point | When / by what |
|---|---|
| M7 tolerances for registration and comparison | after a sufficient data base (noise floor, effect of speckle reduction; M4 study with K13, MI-31); until then synthetic tests with tolerances in the test only, monitor thresholds empty |
| M10 ICC form, M11 ECC in the publication | during MI-04, with the paper |
| `pose_tolerance_um` for 20x / 63x | measure |
| Border band width, stitching details (overlap fraction, inner order) | study; the maintainer works out the details |
| Final defaults for positions / frames / offset | after the speckle / position study (M4) |
| G8 route (a) or (b); pythonnet on Python 3.14 | test in DL-H1 |
| D10 Windows service | if DL-H1 shows Koala in session 0, else after replacing Koala |
| Python upgrade of Work-PC / pre-release PC | after checking mvIMPACT / NumPy 2 |
| O5 least squares or `temporal` as default | after K11 / K11-T |
| Final default z per objective | focus study K10 |

## 8. Milestones towards Release V1.0

| Step | Content | Result | Hardware |
|---|---|---|---|
| S0 | Hygiene (done 2026-11-18: legacy/attic moves, translations), decisions (done 2026-11-18), master plan v0.4 in all repositories, contracts v1 (DL-A3, DL-A4), parameter-set schema (O11), material seed (G7). | Clean baselines, frozen contracts. | no |
| S1 | OffAxisHolo product API (K8, K9, K16), dhmconnect protocol + service on sim/replay incl. groups (DL-B1–B7, DL-C1/C2), nanofactorysystem adapter with fake (T82–T84), MIRA product types (MI-10). | Dry run of an experiment through the service without hardware. | no |
| S2 | Hardware: DL-H1 (Koala binding G8, Python 3.14, TDR check), real gateway (DL-H2), measurements (DL-H3), operation (DL-H4: Task Scheduler + watchdog), GPU validation (K14), service mode (T86, T87). | Masterplan stage R1. | yes |
| S3 | Information model and operating modes (T78–T81, T88–T92), calibration mapping, focus from the corner markers, target rasteriser, MIRA comparison (MI-50–MI-53). | Masterplan stage R2 (monitor, shadow). | partly |
| S4 | Plan generator (T85), registration, fusion, stitching, unwrapping per tile (MI-20–MI-42, DL-B9), speckle / position study (K13, MI-31), orphan recovery, raw transfer, archive tool. | Large structures, known noise floor. | partly |
| S5 | OffAxisHolo GUI (L0–L8), camera live feed (T93–T95), run control (T96, T97), system GUI (F15). | Usability. | partly |
| S6 | Process model and controller (F16). | Masterplan stage R3, after V1.0. | yes |

S3 can start in parallel to S2. S5 can start once S1 exists. The planning session P1 (K18) took place on
2026-11-17/18 (G10).

## 9. Working rules for Claude Code in all four repositories

1. The repository's own `CLAUDE.md` wins over this file.
2. Planning first: todos named "Design" end with a document and wait for maintainer approval (line
   `Status: approved <date>` written by the maintainer, or recorded by Claude when the maintainer approved
   it in a decision session).
3. A todo with an open or deferred decision is never guessed. Implement the independent part, keep the
   todo in "In Progress" and add a follow-up todo that names the missing part and the decision or
   measurement it needs.
4. All four `TODO.md` files use the same format (sections Open / In Progress / Blocked / Done, fields
   Priority, Size, Depends on, Done when; format rules in the comment at the top of each file) and keep a
   table "Decided" (ID, date, result) next to "Open decisions". Sizes: S (hours), M (about a day), L
   (several days).
5. The existing paths keep working: offline OffAxisHolo is the golden reference, the legacy DHM path of
   nanofactorysystem stays selectable, MIRA's publication analysis keeps its results.
6. No assumed performance numbers. Measure, record the number with date and machine, then optimise.
7. Everything is testable without hardware (dummy backend, simulated gateway, replay). Tests that need
   the lab are marked `hardware`.
8. Nothing is deleted without the maintainer's approval. **Attic procedure (X-B):** a file that leaves
   the active code must have been committed at its original path; in a `chore` commit it moves to the
   local, Git-ignored `attic/` folder (`git rm --cached`), and `docs/ATTIC_INDEX.md` records original
   path, last commit SHA, date and reason. Frozen code that is still needed (DHMConnect `legacy/`) stays
   versioned.
9. A change to a shared contract is made in `dhmconnect/schemas/` first; the other repositories update
   their vendored examples in their own commits and reference the DHMConnect commit.
10. No silent defaults for physical parameters (refractive indices, z, pixel size, wavelength).

## 10. Facts about the machines

| Date | Machine | Fact | Source |
|---|---|---|---|
| 2026-11-16 | DHM-PC | The GPU driver supports CUDA 13 (OffAxisHolo uses `cupy-cuda13x`). | OffAxisHolo handover 2026-11-16 |
| 2026-11-16 | DHM-PC | Python is not installed yet; the PC was only reached through `DhmServ.exe`. Target: Python 3.14 (G9). | OffAxisHolo handover 2026-11-16 |
| 2026-11-18 | DHM-PC | HP Z2 Tower G9; i9-14900K (32 threads); 64 GB; RTX A2000 12 GB (driver 32.0.15.8115) + Intel UHD 770; NVMe 1 TB (794 GB free) + HDD 2 TB (1.8 TB free); Intel I219-LM 1 GbE + Realtek 1 GbE; Windows 11 Pro build 26200, accessed via RDP; **repeated LiveKernelEvent 141 (GPU timeout / TDR)**. | DxDiag 2026-11-18 |
| 2026-11-18 | Work-PC | Custom build (Z490); i5-10600K (Windows sees 5 CPUs: INtime / Aerotech HyperWire); 32 GB; Intel UHD 630 (QuickSync), no CUDA GPU; NVMe 500 GB (205 GB free); Realtek 2.5 GbE + Intel I219-V 1 GbE + Broadcom 1 GbE; Windows 10 Pro 22H2 (end of support 2025-10-14); **repeated LiveKernelEvent 144 (USB3 driver)**. | DxDiag 2026-11-18 |

## 11. After V1.0 (notes, not todos)

Hardware access without Koala (pypylon for the camera; own OPL / laser control only with vendor approval
because of laser safety) and then the service as a Windows service at boot; MIRA on the GPU (CuPy, as in
OffAxisHolo); refractive index as a function of laser dose (link to the voxel database); rules for mask
sizes from a paper on DHM architectures (provided by the maintainer); more design inputs (grey-value height
maps ≤ 50 nm); speckle: selection of bad captures, adaptive capture pattern; offload the controller (F16) to
the DHM-PC; raise the Python version of all PCs together and the Work-PC to Windows 11 (after consulting
Aerotech / INtime); optional 2.5/10 GbE card; in-depth code optimisation once everything is in use.
