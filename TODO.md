# TODO

<!--
Format rules (for humans and Claude):
- IDs are never reused. Groups: B/C/H carried over from docs/realtime/todo_realtime.txt,
  K product interface, L GUI, P package backlog (from src/offaxisholo/ToDos.txt).
  Follow-ups reference their origin: "(follow-up of K3)" or "(found during K3)".
- Phase: every open todo carries "[phase: <n>]" (see "Phase plan"). Backlog todos carry "[phase: B]".
- Priority: high | medium | low
- Size: S (hours) | M (about a day) | L (several days)
- Depends on: IDs or "–". IDs of the other repositories are allowed: T…/F… (NanoFactorySystem_automated),
  DL-… (DHMConnect), MI-… (MIRA); G… are the global decisions in docs/system/SOFTWARE_PLAN_OVERVIEW.md,
  O… the decisions of this repository (table below).
- Done when: one line or sub-bullets; every criterion must be verifiable.
- "Decision:" marks a question to the maintainer (working rule 3).
- Sort "Open" by phase, then by blockers/dependencies, then priority.
- "Done" is a staging area: the status review routine moves entries to ARCHIVE.md.
-->

## Phase plan

Target state: the realtime kernel delivers a complete `ReconstructionResult` (phase, intensity, optional
field, per-frame metrics and selection, quality numbers, timings, parameter set, versions) through two
entry points that the DHM service (DHMConnect) and the GUI use; it is validated on the GPU of the DHM-PC;
a GUI covers offline analysis, the experiment browser and live operation.

| Phase | Topic | Todos | Gate before the dependent todos start |
|---|---|---|---|
| 0 | Scope, repository hygiene, dependencies | K1, B5, B3, K21 | – |
| 1 | Product interface (no hardware) | K4, K16, K5, K2, K3, K6, K17, K7, K18 (done), K8, K19 | DHMConnect DL-A3 for the final schema (K16) |
| 2 | Replay, N-study, public API | K12, K13, K15 | – |
| 3 | Validation on the DHM-PC (hardware) | K14 | access to the DHM-PC |
| 4 | Physics parameters and parameter sets | K9, K10, K11, K20 | – (O1, O2, O5, O11, G4, G7 decided 2026-11-18) |
| 5 | Optimisation after profiling | C13 | K14 numbers exist |
| 6 | GUI (observer only, O8 / O9) | L0, L1, L2, L3, L4, L5, L7, L8, L9 | L0 approved by the maintainer; L5 needs DHMConnect DL-C1, DL-B10 |
| B | Package backlog (offline library) | P1–P17 | – |

Phase 6 can start in parallel to phases 2–5 once K6 exists. Target: Release V1.0 as defined by the
decision session of 2026-11-17/18 (master plan v0.4). Since G5-new this package runs inside the DHM
service on the DHM-PC (compute node) and offline / in the GUI on any PC; it reconstructs **one hologram
stack** and leaves registration, unwrapping across tiles, fusion, stitching and levelling to MIRA.

### Working rules for Claude Code (in addition to CLAUDE.md)
1. Pick the first open todo of the lowest unfinished phase whose dependencies are done. Take phase B
   todos only when nothing else is available or the maintainer names one. K14 and other `hardware`
   todos are done together with the maintainer on the DHM-PC.
2. Design todos (L0) end with a document in `docs/`. After writing it, move the todo to "Blocked" with
   "Blocked by: waiting for maintainer approval". Dependent work starts only when the document contains
   the line `Status: approved <date>` written by the maintainer.
3. If a todo leaves a decision open ("Decision:" or a reference to the table below), propose an answer
   and ask the maintainer before implementing the part that depends on it.
4. The offline library is the golden reference; `tests/test_offline_golden.py` and
   `tests/test_import_hygiene.py` must pass after every todo.
5. Network, protocol, camera, service and archive code belong to DHMConnect; registration, fusion,
   stitching and target comparison to MIRA. Add a todo there instead of building it here.
6. At the end of each phase run `python -m pytest` and append a WORKLOG.md entry
   "[Phase <n>] Phase summary" with the finished todos, open follow-ups and anything the maintainer has
   to check on the DHM-PC.

## Open decisions (maintainer)

All decisions O1–O11 and the global decisions that block todos here were taken in the decision session of
2026-11-17/18 (master plan v0.4 §6). Still deferred and relevant here:

| Point | When / by what | Blocks |
|---|---|---|
| Least squares or `temporal` as the default unwrapping method (O5) | after K11 and the test plan K11-T | default of K11 |
| Final default z per objective (G4) | focus study K10 | parameter-set default of K9 |
| Border band width discarded after propagation (G10 SA6) | study (depends on z, effective NA of the sideband filter) | default of `optical_margin_um` / border band in K9 |
| `single` precision as GPU default | K14 on the DHM-PC | K14 |

## Decided

| ID | Date | Result |
|---|---|---|
| G2 | 2026-11-16 | JSON Schema plus example messages in `dhmconnect/schemas/`, vendored here under `tests/contracts/`. |
| O3 | 2026-11-16 | Default frame selection `all` until the N-study K13; frame metrics on by default in the entry points (re-evaluate after K14). |
| O6 | 2026-11-16 | A background never expires; a configurable maximum age only warns (DHMConnect default 3600 s). |
| O1, O2 (G4) | 2026-11-18 | Separate parameters `propagation_index` and `n_surround`; z always given and stored with its `propagation_index`; `n_structure` from the material snapshot; in-situ only resin indices (surround = liquid resin, structure = cured in resin), developed: surround = air; z of an experiment from the corner-marker focus (nanofactorysystem T107, sweep K10), fallback per objective from the parameter set, provisional 63x **−2.6 µm at n = 1.0**; per-capture autofocus only reported in `quality`; sign: more optical path → positive phase, n_structure > n_surround → positive height, `phase_sign` (±1) in the parameter set, verified once against Sensofar / AFM. |
| O4 | 2026-11-18 | Superseded by G10 (SA7): "mask per objective" is dropped; a-priori mask per capture from `occupied_um` (K19). |
| O5 | 2026-11-18 | A single `ReconstructionResult` stays wrapped; unwrapping happens in the group step per tile (MIRA plug-in) with `fast2d`; methods `fast2d`, `least_squares`, `none`, `temporal`; quality data in the product; least squares only after K11; test plan K11-T. Printed structures are ≤ 15 µm high → in-situ with SZ2080 the phase stays ≤ ≈ 1.7 rad, no 2π ambiguity; `temporal` low priority. |
| O7 (G3) | 2026-11-18 | PySide6 + pyqtgraph; `offaxisholo.gui` with extra `[gui]`; napari evaluated in L0 as an option for the offline analysis GUI. |
| O8 | 2026-11-18 | Live view computed in the service (reduced hologram / phase / intensity, binning, rate limit, live socket 27185, below measurement jobs); local reconstruction only on the pre-release PC or for replay. The GUI runs on the Work-PC and the DHM-PC and takes **only images** as input (L9 "image source" adapters). |
| O9 | 2026-11-18 | No stage control and no DHM operation in this GUI: it is a pure observer. DHM + stage operation is part of the nanofactorysystem system GUI (F15). L6 is dropped. |
| O10 (G9) | 2026-11-18 | DHM-PC: Python 3.14 (Python install manager, `uv`), `cupy-cuda13x` (CuPy 14 has cp314 wheels); repositories `requires-python >= 3.12`, tests on 3.12 and 3.14 (K21). |
| O11 | 2026-11-18 | The modes "print" / "developed" are dropped. Explicit parameters (`n_structure`, `n_surround`, `propagation_index`) and named, saveable JSON configurations (`ParameterSet`) for camera, objective (pixel pitch in the object plane never from Koala) and reconstruction (default z, stored `OrderCalibration` of the first order, mask shape and size, crop, `eps`), loaded by DHMConnect `configure`. Removing `mode` from the offline API is a BREAKING CHANGE; the golden reference is kept by a parameter set with the old values (K9). P5 is obsolete. |
| G7 | 2026-11-18 | Material registry in nanofactorysystem = master; its versioned JSON export is the source of the material presets here (K9); no silent defaults (a missing index raises or warns with the source). One DHM wavelength 0.674994 µm. |
| G10 | 2026-11-18 | SA1 a-priori mask (default) from `occupied_um` + `pose_tolerance_um` (context) + `optical_margin_um` (parameter set), rasterised by a helper here (K19); SA3 no second pass; SA4 `reference_source` ladder; SA5 this package only aligns the frames of one measurement. |
| DL-A3 b | 2026-11-18 | `intensity` → `intensity_rel` = I_object / I_background (dimensionless, 1.0 = empty substrate), because compensation divides the fields (K16). |

## Open

### Phase 0 — Scope, repository hygiene, dependencies

### Phase 1 — Product interface (no hardware)

- [ ] K17: Frame gating for the sliding and EMA modes (follow-up of K3) [phase: 1]
  - Priority: low
  - Size: M
  - Depends on: K3, K6; useful for DHMConnect DL-B10 (live mode)
  - Done when:
    - fixed thresholds are evaluated per frame on the device into a 0/1 weight (no host decision), and the
      sliding and EMA accumulators average with that weight (weighted count, `n_effective` correct);
    - the rejected frames are reported in the frame records; hot-path check and a CuPy test exist.

### Phase 2 — Replay, N-study, public API

- [ ] K12: Replay reader for experiment files and the raw archive [phase: 2]
  - Priority: high
  - Size: M
  - Depends on: K6
  - Done when:
    - `offaxisholo.io.experiment` reads hologram stacks, device parameters and products from
      `experiment.h5` (layout: `docs/design/EXPERIMENT_STORAGE.md` in NanoFactorySystem_automated) and
      from raw archive files of DHMConnect, with `h5py` only and without HDF5 file locking;
    - `tools/replay_experiment.py` reconstructs captures from it;
    - a test builds a small file with the documented layout and replays it.

- [ ] K13: N-study [phase: 2]
  - Priority: medium
  - Size: S
  - Depends on: K2, K3, K12
  - Done when:
    - `tools/noise_vs_n.py` reports the noise in the reference region against the number of frames for
      subsets 1, 2, 4, … N and for each selection policy, from a replayed capture;
    - the evaluation on real data (captures with `keep_raw = all`) is recorded in WORKLOG.md and gives
      the input for O3. Partial result allowed while no such captures exist.
  - Note (2026-11-19, MIRA MI-31): report the noise with the shared definition `mira.noise.reference_noise`
    (optional import) so fusion, comparison and this study give the same number; `mira.noise.fit_noise_model`
    fits σ² = σt²/(Nf·Np) + σcoh²/Np (M4).

- [ ] K15: Public API and documentation [phase: 2]
  - Priority: medium
  - Size: S
  - Depends on: K6
  - Done when:
    - `offaxisholo.realtime` exports the entry points and result classes; README has a section on the
      product interface;
    - `offaxisholo.__version__` and a helper that returns the git hash for the result metadata exist
      (covers old G2 "version in every result").

### Phase 3 — Validation on the DHM-PC (hardware)

- [ ] K14: GPU validation on the DHM-PC [phase: 3]
  - Priority: high
  - Size: S
  - Depends on: access to the DHM-PC (with the maintainer)
  - Done when:
    - `pytest tests/test_backend.py -v`, `pytest tests/realtime -m gpu`, `tests/benchmark_backend.py` and
      `tests/benchmark_realtime.py` (with and without `--metrics`) have run on the DHM-PC;
    - single against double precision is compared on real holograms (phase and height maps), and
      the maintainer decides from it whether `"single"` becomes the GPU default;
    - GPU, memory, CUDA version, driver, operating system, Python version (3.14, G9) and all numbers are
      recorded in `docs/realtime/benchmarks.md`; kernels stay short (the DHM-PC shows repeated GPU TDR
      events, LiveKernelEvent 141; checked together with DHMConnect DL-H1, TCC-mode test);
    - closes B6 (CUDA run of the existing tests) and H4 (validate single precision); gives the input
      for O10.

### Phase 4 — Physics parameters and parameter sets

- [ ] K9: Parameter sets and saveable configurations (O11) [phase: 4]
  - Priority: high
  - Size: L
  - Depends on: DHMConnect DL-A3 (schema home); nanofactorysystem T108 (material JSON export)
  - Done when:
    - a `ParameterSet` schema with named, saveable and loadable JSON configurations exists for camera
      (pixel pitch, ROI, bit depth, exposure as metadata), objective (magnification, NA, pixel pitch in
      the object plane — never from Koala) and reconstruction (default z with its `propagation_index`,
      `OrderCalibration` = stored position of the first order so that `locate_order` does not run for
      every capture, recalibration on request, freely definable mask shape and size, crop size, `eps`,
      unwrapping method, `optical_margin_um`), plus `phase_sign`, wavelength 0.674994 µm, `n_surround`;
      name, id and hash go into every result;
    - the modes "print" / "developed" are removed from the offline API (BREAKING CHANGE); the golden
      reference test passes with a parameter set that holds the old values;
    - material presets are generated from the registry JSON export of nanofactorysystem (G7); no silent
      default for any index (missing → error or warning naming the source);
    - the preset JSON inconsistencies (duplicate keys `xPixelSizeUm` / `pixelPitch`, `wavelengthUm` /
      `wavelength`, name `"Zeiss 63X"` against key `"Zeiss 63x"`) are cleaned without breaking the
      offline loader;
    - values that are not measured yet (63x pixel size, z) are marked `unverified`.
  - `ParameterSet` draft 0.1 and `parameter_set_hash` exist in DHMConnect `fc85f5c` (DL-A3); K9 finalises
    the schema there and must reproduce the hash of the vendored example.

- [ ] K10: z handling and focus sweep (G4) [phase: 4]
  - Priority: high
  - Size: M
  - Depends on: –
  - Done when:
    - z from the request (always with its `propagation_index`), a list of z values (one FFT, several
      transfer functions) and a z sweep with a sharpness metric of the reconstructed amplitude (default:
      minimum of std / mean) exist and are tested on synthetic defocused fields; used by
      nanofactorysystem T107 (focus from the corner markers);
    - per-capture autofocus can be computed behind a flag and is only reported in `quality`, never
      applied, until validated;
    - the focus study (real data, 20x / 63x) gives the final default z per objective; until then 63x
      uses −2.6 µm at n = 1.0 (recorded in the parameter set as `unverified`).

- [ ] K11: Unwrapping per tile and test plan K11-T (O5) [phase: 4]
  - Priority: medium
  - Size: M
  - Depends on: –
  - Done when:
    - a single `ReconstructionResult` stays wrapped; a function unwraps one tile with `fast2d`,
      `least_squares` (congruent and not) or `none` and returns quality data (method, residue count,
      consistency flag), for MIRA's group step (MI-40);
    - `temporal` (φₙ = φₙ₋₁ + wrap(φₙ − φₙ₋₁), existing `TemporalUnwrapper`) is offered with optional
      re-anchoring every N layers (low priority: no 2π ambiguity expected in-situ with SZ2080 ≤ 15 µm);
    - the test plan K11-T in `docs/realtime/unwrapping_comparison.md` §"K11-T" is run on synthetic data;
      the comparison decides whether least squares or `temporal` may become a default (criterion there).

### Phase 5 — Optimisation after profiling

- [ ] C13: Optimisations after profiling (carried over) [phase: 5]
  - Priority: low
  - Size: L
  - Depends on: K14
  - Done when:
    - only optimisations that the K14 profile justifies are implemented (fused kernels for window,
      shift, mask, piston and accumulation; CUDA graphs; streams to overlap upload, compute and download;
      pinned buffers; `rfft2`), each with before / after numbers in `docs/realtime/benchmarks.md` and
      parity tests unchanged.

### Phase 6 — GUI (package `offaxisholo.gui`, entry point `offaxisholo-gui`, extras `[gui]` and `[live]`)

Legacy to review, not to extend: repository `DHM_User_Interface` (PyQt5; `Reconstruction/` with
`front_end.py` and `panels.py` for offline reconstruction, `DHM_GUI/` for capture with stage control,
wish list in its `todo.txt`).

- [ ] L0: Design the GUI [phase: 6]
  - Priority: medium
  - Size: M
  - Depends on: – (G3 decided: PySide6 + pyqtgraph)
  - Done when:
    - `docs/gui/DESIGN.md` covers users and use cases, screens, architecture (models without Qt,
      views, worker threads, no computation in the UI thread), the "image source" abstraction (L9) as the
      only input, the observer-only rule (O9: no stage, no DHM operation), the two target PCs (Work-PC
      and DHM-PC), what is taken over from the legacy PyQt5 GUI and what is dropped, and an evaluation of
      napari for the offline analysis part;
    - the maintainer has approved it (`Status: approved <date>`); L1–L8 start only afterwards.

- [ ] L1: Application skeleton [phase: 6]
  - Priority: medium
  - Size: M
  - Depends on: L0, B3
  - Done when:
    - a main window with dock layout, persistent settings, log pane, worker infrastructure and operator
      name starts; an import hygiene test shows that the core never imports the GUI.

- [ ] L2: Image views [phase: 6]
  - Priority: medium
  - Size: L
  - Depends on: L1
  - Done when:
    - hologram, spectrum (log magnitude) with sideband marker and mask overlay, phase and intensity are
      shown with linked zoom and pan, colour maps, colour bar with units, histogram, ROI statistics,
      line profile and two views side by side.

- [ ] L3: Offline analysis [phase: 6]
  - Priority: medium
  - Size: L
  - Depends on: L2, K6
  - Done when:
    - hologram and background files (tif, png, zdc) can be opened; a parameter panel bound to the
      configuration classes covers parameter set, order calibration, crop, mask, regularisation, a z
      slider that re-propagates from the cached compensated field, and the unwrapping method;
    - images and arrays are exported with a JSON metadata file that contains the configuration hash.

- [ ] L4: Experiment browser [phase: 6]
  - Priority: medium
  - Size: L
  - Depends on: L3, K12
  - Done when:
    - `experiment.h5` opens read-only through K12 as a tree substrate → experiment → structure →
      capture with phase and layer; stored products are shown; captures can be reconstructed again from
      stored holograms or the raw archive with other frame counts and selection policies; batch export.

- [ ] L5: Live mode [phase: 6]
  - Priority: medium
  - Size: L
  - Depends on: L2, L9, DL-B10, DL-C1 (observer role, live socket 27185)
  - Done when:
    - a connection dialog and status bar (service state, queue, dropped frames, background age) work
      through `dhmconnect.client`; live hologram, spectrum, phase and intensity are shown with frame rate
      and latency; a warning appears when no service is reachable and the simulated service is used.

- [ ] L9: "Image source" input abstraction (O8) [phase: 6]
  - Priority: medium
  - Size: M
  - Depends on: L1
  - Done when:
    - the GUI reads images only through an `ImageSource` interface (frames + metadata, no control
      commands); adapters: file / replay (raw archive, `experiment.h5`), `dhmconnect` live client as an
      optional extra (`[live]`), and a documented pattern for other devices of the same architecture;
    - a test drives the views with a synthetic source, without Qt event loop where possible.

- [ ] L7: GUI tests [phase: 6]
  - Priority: medium
  - Size: M
  - Depends on: L1
  - Done when:
    - model tests run without Qt; smoke tests run with `pytest-qt` in offscreen mode and are skipped
      without the `[gui]` extra; the live mode is tested against `dhmconnect.testing`.

- [ ] L8: User guide [phase: 6]
  - Priority: low
  - Size: S
  - Depends on: L3
  - Done when: a short guide with screenshots exists in `docs/gui/`.

### Phase B — Package backlog (offline library; from `src/offaxisholo/ToDos.txt`)

Each item keeps the offline behaviour unchanged unless the todo says otherwise; the golden test decides.

- [ ] P1: Data loader: array input and container comparison [phase: B]
  - Priority: low | Size: M | Depends on: –
  - Done when: `DataLoader` is checked for sufficiency, accepts `np.ndarray` input, and compares `.zdc`
    containers as the old docker script did; tests on small fixtures.

- [ ] P2: Use the scidatacontainer `StructureContainer` in the library [phase: B]
  - Priority: low | Size: M | Depends on: –
  - Done when: the error in the metadata generation is found and fixed; when a structure container is
    used, the DHM parameters stored inside it are used.

- [ ] P3: Refractive index out of the DHM preset [phase: B]
  - Priority: low | Size: S | Depends on: K9
  - Done when: the refractive index leaves the DHM preset; material indices come from the generated
    material presets (G7, K9); old presets still load with a warning.

- [ ] P4: Clean up `HologramProcessor` [phase: B]
  - Priority: low | Size: M | Depends on: –
  - Done when: `DHMPlotter` is a member of the processor, `set_dhm_dict` and `set_material_dict` exist,
    and the image filters that belong to MIRA are removed (after the maintainer confirms no caller needs
    them).

- [ ] P6: Rework plotting [phase: B]
  - Priority: low | Size: M | Depends on: –
  - Done when: single-image and height plots are reworked in the style of the visualization scripts, and
    images can be exported directly as PNG / TIFF data without matplotlib (usable as data points like a
    captured hologram).

- [ ] P7: Rework the logger [phase: B]
  - Priority: low | Size: S | Depends on: –
  - Done when: a new logger factory (rotating files, levels, sensible names) exists next to
    `utils.get_logger`, whose current behaviour (console + file, DEBUG always, name = log file path)
    stays for old callers.

- [ ] P8: Further propagation algorithms [phase: B]
  - Priority: low | Size: M | Depends on: –
  - Done when: Fresnel propagation (and further methods from the SPIE Photonics West talk on
    error-free angular spectrum propagation) are implemented on the array backend and compared with
    `angularSpectrum` on analytic cases.

- [ ] P9: Decide on the "full reconstruction" flag of `Hologram` [phase: B]
  - Priority: low | Size: S | Depends on: –
  - Done when: it is documented whether the flag is needed; if not, it is deprecated (not removed).

- [ ] P10: Further GPU vendors [phase: B]
  - Priority: low | Size: L | Depends on: K14
  - Done when: a PyTorch backend (AMD ROCm, Apple MPS; dispatch for pad, ndimage filters and DCT; MPS has
    no complex128) and optionally JAX are evaluated and, if worthwhile, implemented behind the backend
    interface.

- [ ] P11: GPU-capable exact unwrapping algorithms [phase: B]
  - Priority: low | Size: M | Depends on: O5
  - Done when: branch cut, graph cut or PUMA on the GPU are evaluated against "Fast 2D" with
    `tools/compare_unwrapping.py`; report in `docs/realtime/unwrapping_comparison.md`.

- [ ] P12: Batch processing on the GPU [phase: B]
  - Priority: low | Size: M | Depends on: K10
  - Done when: layer stacks and z-scans over many propagation distances run batched on the GPU with
    measured speed-up.

- [ ] P13: Extended documentation of the library [phase: B]
  - Priority: low | Size: M | Depends on: K15
  - Done when: an API documentation (e.g. Sphinx; old configuration in `attic/conf.py`) and usage
    examples for the offline library exist under `docs/`, including the batch workflow
    `ReconstructionPipeline` (called "docker" in the maintainer's old notes, formerly `DockerImageFile`).

- [ ] P14: LaTeX export of functions [phase: B]
  - Priority: low | Size: S | Depends on: –
  - Done when: the maintainer's idea "latexify_py for LaTeX export of Python functions" is evaluated for
    the core formulas (paper use).

- [ ] P15: Zernike decomposition and radial PSD as pure functions (requested by MIRA MI-74, MI-75) [phase: B]
  - Priority: medium | Size: M | Depends on: –
  - Done when: `offaxisholo` offers a Zernike decomposition (named modes, masked aperture) and a radially
    averaged PSD (frequency axis in 1/µm) as pure, backend-agnostic functions with tests on synthetic
    data; the old script `attic/legacy_tests/test_zernike.py` is reviewed for reuse; MIRA is told the final
    signatures.

- [ ] P16: Compare compensation before propagation with propagating both fields (found during B5) [phase: B]
  - Priority: low | Size: M | Depends on: real data of a known reference structure at several z
  - Done when: `P_z{E_obj · conj(E_ref) / |E_ref|²}` (current order) and `P_z{E_obj} / P_z{E_ref}` are
    compared on a known reference structure (object and background at several z) with a tool in
    `tools/`; result appended to `docs/reconstruction_order.md`; the offline default stays unchanged.

- [ ] P17: Numerical aberration compensation (found during B5) [phase: B]
  - Priority: low | Size: M | Depends on: P15
  - Done when: a numerical parametric lens / Zernike fit for use without a background hologram
    (Colomb et al., J. Opt. Soc. Am. A 23, 3177 (2006)) is evaluated on synthetic and real data against
    background compensation; source: `docs/reconstruction_order.md` §6.

## Items of `docs/realtime/todo_realtime.txt` and where they went

| Old item | Status / new place |
|---|---|
| B0, B1, B2, B4 | done 2026-11-13 (see "Done") |
| B3 (partial) | B3 above |
| B5 (partial) | B5 above |
| B6 | part of K14 |
| C1–C12, C14 | done 2026-11-13 (see "Done") |
| C13 | C13 above |
| D1–D4 (camera) | DHMConnect DL-B1 (gateway incl. simulated and replay camera), DL-B2 (acquisition, ring buffer), DL-H2 (real driver) |
| E1 (protocol) | DHMConnect DL-A2, DL-A4 |
| E2 (state machine) | DHMConnect DL-B3 |
| E3 (GPU worker) | DHMConnect DL-B4 (uses K6) |
| E4 (network layer) | DHMConnect DL-B5 |
| E5 (live mode) | DHMConnect DL-B10 |
| E6 (archive) | DHMConnect DL-B7 |
| E7 (logging) | DHMConnect DL-B11 (service logs); P7 here (library logger) |
| E8 (service tests) | DHMConnect, part of every DL-B todo |
| F1, F2 (client, examples) | DHMConnect DL-C1, DL-C3 |
| F3 (integration with the control script) | NanoFactorySystem_automated T86 |
| F4 (reference measurement) | NanoFactorySystem_automated T87 (with K14) |
| G1, G3, G4 (autostart, runbook, long run) | DHMConnect DL-H4 |
| G2 (version in every result) | K15 here, DL-B11 in DHMConnect |
| G5 (bind address, firewall, authentication) | DHMConnect DL-H5 and decision D5 |
| G1.8 (unwrapping comparison) | first report done (`docs/realtime/unwrapping_comparison.md`); continues in O5, K11, P11 |
| H1, H3 | decision O1 (G4) |
| H2 | decision O2 (G4) |
| H4 | part of K14 |
| H5 | decision O5, K11 |
| H6 | decision O11, P5 |
| Q1 camera / SDK | known: Basler acA1920-155um, today reached through Koala; access decided in DHMConnect (DL-H1, D1) |
| Q2 GPU / CUDA | K14, O10 |
| Q3 N, n, frame rate, latency | DHMConnect DL-H3 (measurement) |
| Q4, Q5 plane, z, medium, sign | O1, O2 (G4) |
| Q6 unwrapping in the service | O5 |
| Q7 network, authentication, clients | DHMConnect D4, D5 |
| Q8 background lifetime | O6 |
| Q9 live stream for control | NanoFactorySystem_automated T72 (operating modes), O8 |

## In Progress
<!-- Claude Code moves a todo here when starting work. -->



## Blocked
<!-- Format: todo as above, plus the line "Blocked by: <reason or ID>". -->

- [ ] B3: Split the dependencies (carried over, partial) [phase: 0]
  - Priority: medium
  - Size: S
  - Depends on: K21, DL-C1, L1
  - Done when:
    - `pyproject.toml` has the extras `gpu`, `dev`, `gui` (PySide6, pyqtgraph; G3 decided) and `live`
      (`dhmconnect` client; after DL-C1); the default install stays as today (offline library works);
    - the minimum versions are installable on the DHM-PC Python (recorded in WORKLOG.md);
    - `[project.scripts]` entry `offaxisholo-gui` is added together with L1.
  - Already done: pytest configuration and markers (`04067ea`); `gpu` extra switched to `cupy-cuda13x`,
    `requires-python` ≥ 3.11 (`2c8abb5`).
  - Blocked by: Python 3.14 is not yet installed on the DHM-PC (install check with DL-H1 / K14); the
    `live` extra waits for DHMConnect DL-C1, the script entry for L1. The `gui` extra may be added now.

## Done

- [x] K8: Quality numbers — 2026-11-19 — SA4 reference ladder (a-priori mask via `set_reference_mask` / `reconstruct_frames(reference_mask=...)` → border ring → whole field) for piston, per-frame noise and `quality`; `phase_noise_rad` after piston + plane removal (null for the whole field), warning below `min_reference_fraction`; new per-frame metric and `quality.sideband_snr` (sideband vs Nyquist-corner noise block) — commits: `0c4a3b8`.
- [x] K19: Reference-mask rasteriser helper (G10 SA2) — 2026-11-18 — `offaxisholo.realtime.reference_mask`: pure NumPy function from `occupied_um` polygons (or a `uint8` raster), actual position, `pose_tolerance_um` + `optical_margin_um`, pixel size (y, x) and an optional image → stage matrix to a boolean mask (True = structure-free); tested for exact area, margin, rotation and raster parity — commits: `2c2666d`.
- [x] K20: Phase ↔ height in `fieldops` with explicit indices (G4) — 2026-11-18 — `phase_to_height_um` / `height_to_phase_rad` with keyword-only `n_structure`, `n_surround`, `wavelength_um`, `phase_sign`, no defaults, documented as the owner of the formulas for MIRA MI-51; tested on an analytic 15 µm step (SZ2080 Δn 0.012 in-situ, 0.52 developed) — commits: `906f9b8`.
- [x] K4: `ReconstructionResult` and `FrameRecord` — 2026-11-18 — Dataclasses, `to_message()` / `from_message()` and round trips since `43b772f`; the remaining criterion (validation against the vendored examples) was met by K16 — commits: `43b772f`, `84bb939`.
- [x] K16: Validate `ReconstructionResult` against the shared contract — 2026-11-18 — Result follows contract 1.0 of DHMConnect `fc85f5c`: `intensity_rel`, `pixel_um` (y, x), `propagation_index`, `parameter_set_id`, D13 reference keys, `created_utc` with `Z`, `schema_version` 1.0 (BREAKING CHANGE); schema and examples vendored in `tests/contracts/`, `tests/realtime/test_contract.py` checks them — commits: `adae46d`, `84bb939`, `5d01b8e`.
- [x] K21: Python baseline 3.12 / 3.14 (G9) — 2026-11-18 — `requires-python >= 3.12` with classifiers 3.12 / 3.14; the suite passed on CPython 3.12.3 and 3.14.10 (187 passed, 12 skipped each); `cupy-cuda13x` 14.2.0 resolves as a cp314 Windows wheel — commits: `dbfee72`.
- [x] K18: Cross-repository planning: reference mask, averaging and stitching — 2026-11-18 — Decision session 2026-11-17/18: SA1–SA7 decided (G10), recorded in master plan v0.4 (`e994707`) and the "Decided" tables; resulting todos K8 (unblocked), K19, MIRA MI-40/41, nanofactorysystem T104, DHMConnect DL-B4/B9.
- [x] P5: Mode "print" of the e-field calculation — 2026-11-18 — Dropped: O11 removes the modes; explicit parameters and parameter sets instead (K9).
- [x] L6: DHM operation panel — 2026-11-18 — Dropped: O9 makes the GUI a pure observer; DHM + stage operation moves to the nanofactorysystem system GUI (F15).
<!-- Claude Code adds: - [x] <ID>: title — YYYY-MM-DD — 1–2 sentences on what changed — commits: `<sha>`, … -->
- [x] P18: Translate the remaining German documentation in active code — 2026-11-17 — comment in `utils.get_logger` and docstring of `Hologram._locate_order` in English; the maintainer's own notes in `io/structure.py`, the verbatim reference skeleton and `docs/realtime/*.txt` stay German on purpose (found during the cross-repository review of 2026-11-17). — commits: `ca07542`
- [x] K7: Background model — 2026-11-16 — `BackgroundModel` with Ub, K, quality, id, creation time and a background hash of the relevant configuration parts; save / load as `.npz` without pickle with id check; validity check and age warning (O6) tested; usable in both entry points. — commits: `d2f5342`
- [x] K6: Entry points — 2026-11-16 — `realtime/api.py`: `reconstruct_frames` and `ReconstructionSession` over `OffAxisPipeline` with metrics on and selection `all` by default, acquisition info passed through; the import surface (entry points plus configuration classes) is tested in a fresh interpreter. — commits: `4cac2bd`
- [x] K3: Frame selection — 2026-11-16 — `realtime/selection.py`: policies `all` (default, O3), `threshold`, `robust`, `best_k` with enforced and reported `min_frames`; block mode keeps the fields (`keep_fields`, `FieldStore`) and `apply_selection` rebuilds the average; mask and reasons go into the frame records. Sliding / EMA gating is K17. — commits: `3a7c520`
- [x] K2: Frame metrics — 2026-11-16 — `realtime/metrics.py`: eleven per-frame metrics computed in `push` from the demodulation FFT without host synchronisation, kept in a device ring buffer and turned into `FrameRecord`s; opt-in via `MetricsConfig.enabled`; tested with saturation, shifted carrier and reduced fringe contrast, CuPy parity test skipped without CUDA. — commits: `aab56d6`, `efdda6c`
- [x] K5: Timings — 2026-11-16 — `realtime/timing.py` `StageTimer` fills `ReconstructionResult.timings_s` (seven stages plus total, stage sum ≤ total tested); device synchronisation only with `OffAxisPipeline(..., profile=True)`; caller values pass through unchanged. — commits: `c66838f`
- [x] Port of the offline library to the array backend (before the realtime plan) — 2026-11-13 — NumPy/CuPy backend with precision control, core reconstruction, DCT least-squares unwrapping, simulation, conversion at plot/save boundaries, backend tests and benchmark, GPU documentation. — commits: `27c0a23`, `bb0bd1b`, `63481b2`, `1c7a012`, `c4b8ada`, `38f49d0`, `e2e56ef`
- [x] B0: pytest configuration and golden reference of the offline library — 2026-11-13 — commits: `04067ea`
- [x] B1: Lazy imports (`offaxisholo`, `reconstruction`, `phaseUnwrapping`) with `tests/test_import_hygiene.py` — 2026-11-13 — commits: `2ba3aff`
- [x] B2: tkinter only in the interactive dialog branch; broken `__main__` block removed — 2026-11-13 — commits: `2b64026`
- [x] B4: Thread-safe explicit backend namespace, thread rule documented — 2026-11-13 — commits: `59d3908`
- [x] Specification and reference skeleton moved to `docs/realtime/` — 2026-11-13 — commits: `e57eed4`
- [x] C1–C12, C14: Realtime kernel (`realtime/`: configuration, calibration, demodulation with crop, piston correction, background and compensation factor, accumulators, compensation and propagation per result, outputs, unwrapping options, tests of gates G1.1–G1.6 on the CPU, benchmark, noise-vs-N tool) — 2026-11-13 — commits: `a446653`, `cfdb038`, `c6ab76e`, `d47eab8`, `885d513`, `a9e5466`, `0656e4f`, `c63d328`, `5a435c9`, `bd44a73`, `f36035a`
- [x] G1.8 (first part): Unwrapping comparison tool and first report — 2026-11-13 — Least Squares deviates 0.65–2.3 rad RMS from Fast 2D on developed structures; Fast 2D stays the default. — commits: `3790c7e`
- [x] B5: Repository hygiene — 2026-11-16 — Legacy scripts, old todo notes and planning inputs moved to `attic/` (with `git mv`; `attic/` stays gitignored), `console.log` deleted, README tree current, `Notes_reconstruction_order.txt` versioned as `docs/reconstruction_order.md`; the maintainer keeps `.git`, `.venv` and `build/` in the Seafile folder knowingly (G9). — commits: `ed74765`, `04e5030`, `5c18e66`, `083fcd4`, `8223404`, `8755f70`
- [x] K1: Finish the scope change in the documents — 2026-11-16 — Project files (`CLAUDE.md`, `TODO.md`, `WORKLOG.md`, system plan) created; `todo_realtime.txt` marked historical and `anweisung_realtime.txt` §4, §7, §8, §10 point to DHMConnect; no todo here asks for network or camera code. — commits: `b247181`, `ee39f40`
- [x] B5-X: Remove the B5 attic files from Git (X-B) — 2026-11-18 — The 28 files that stayed versioned in `attic/` were removed with `git rm --cached` and kept locally; `docs/ATTIC_INDEX.md` records path, original path and last commit. — commits: `22fc2c5`
