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
| 0 | Scope, repository hygiene, dependencies | K1, B5, B3 | – |
| 1 | Product interface (no hardware) | K4, K16, K5, K2, K3, K6, K7, K8 | G2 for the final schema of K4 |
| 2 | Replay, N-study, public API | K12, K13, K15 | – |
| 3 | Validation on the DHM-PC (hardware) | K14 | access to the DHM-PC |
| 4 | Physics parameters (decision driven) | K9, K10, K11 | decisions O1, O2, O5 (G4, G7) |
| 5 | Optimisation after profiling | C13 | K14 numbers exist |
| 6 | GUI | L0, L1, L2, L3, L4, L5, L6, L7, L8 | L0 approved by the maintainer; L5/L6 need DHMConnect DL-C1, DL-B8, DL-B10 |
| B | Package backlog (offline library) | P1–P17 | – |

Phase 6 can start in parallel to phases 2–5 once K6 exists. Phase 4 todos may be started for their
independent part (structure, hashing) and kept "In Progress" until the decisions arrive.

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

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| O1 | Result plane and z per objective (old H1, H3): preset 0.11 µm against measured about −2.6 µm. Same as G4. | z comes with the request; default from the parameter set. | K9, K10 |
| O2 | Medium index and sign convention phase to height (old H2). Same as G4. | – | K9 |
| O3 | Default frame selection policy and thresholds. | Ship `all` until the N-study (K13) is evaluated. | K3 |
| O4 | Reference region for piston phase and noise: fixed mask per objective, or automatic border ring. | Mask in the parameter set, border ring as fallback. | K8 |
| O5 | Unwrapping in the product (old H5). First report: Least Squares deviates strongly on developed structures; in-situ examples need no unwrapping. | Wrapped phase by default, unwrapping optional, `fast2d` stays the default method. Least Squares may become a default only after a systematic comparison with Fast 2D (simulated and real holograms, noise, phase residues, steps / discontinuities, height error, `congruent=True/False`; maintainer note in `attic/ToDos.txt`). | K11 |
| O6 | Lifetime of a background and drift monitoring (old Q8). | Measure first (Masterplan V4). | K7 |
| O7 | GUI framework (G3) and whether the GUI lives in this repository. | PySide6 + pyqtgraph; subpackage `offaxisholo.gui` with extra `[gui]`. | L0 |
| O8 | Live view: reconstruction in the service on the DHM-PC, or locally in the GUI from raw preview frames. | Service side; local reconstruction as fallback. | L5 |
| O9 | Stage control in this GUI (the legacy GUI has a positioning widget). | No. Stage control belongs to the system GUI of nanofactorysystem (F15). | L6 |
| O10 | Minimum versions and CuPy package for the DHM-PC. Partly decided 2026-11-16: the driver supports CUDA 13 → `cupy-cuda13x`; Python is installed there from scratch; `requires-python` aligned to ≥ 3.11 (numpy 2.4.5 / scipy 1.17.1 have no Python 3.10 wheels). Open: confirm the install on the DHM-PC. | Install Python 3.12 (as on the development machine); keep the minimums unless the install fails. | B3, K14 |
| O11 | Mode "print" of the e-field / height calculation (old H6, maintainer note "not good"): what should it compute (n_medium = liquid resin)? | – | P5 |
| G2 | Home of the shared contracts. | **Decided 2026-11-16 (maintainer, as recommended):** JSON Schema plus example messages in `dhmconnect/schemas/`, vendored here under `tests/contracts/`. To be recorded in the master system plan. | K16 |
| G7 | Single source of truth for material data. | Material registry in nanofactorysystem; presets here stay a fallback for offline use. | K9, P3 |

## Open

### Phase 0 — Scope, repository hygiene, dependencies

### Phase 1 — Product interface (no hardware)

- [ ] K16: Validate `ReconstructionResult` against the shared contract (follow-up of K4) [phase: 1]
  - Priority: high
  - Size: S
  - Depends on: G2, DHMConnect DL-A3
  - Done when:
    - the example messages of `ReconstructionResult` and `FrameRecord` from `dhmconnect/schemas/` are
      vendored under `tests/contracts/` and a test validates the header of `to_message()` against them;
    - field names, units and `schema_version` of `realtime/result.py` match the contract (bump to 1.0).

- [ ] K2: Frame metrics [phase: 1]
  - Priority: high
  - Size: M
  - Depends on: K4
  - Done when:
    - new module `realtime/metrics.py` computes per frame: saturated and black pixel counts, median and
      5 % / 95 % quantiles, sideband peak weight and position (from the FFT that the demodulation
      computes anyway), piston phase and phase noise in the reference region; values stay on the device
      until the result is built (hot-path rules);
    - tests with synthetic frames (saturation, shifted carrier, reduced fringe contrast) pass on NumPy;
      a CuPy parity test exists and is skipped without CUDA.

- [ ] K3: Frame selection [phase: 1]
  - Priority: high
  - Size: M
  - Depends on: K2; O3 for the default policy
  - Done when:
    - new module `realtime/selection.py`: a pure function from a metrics table to a mask and a reject
      reason per frame; policies `all`, `threshold`, `robust` (median / MAD), `best_k`; `min_frames` is
      always enforced and reported;
    - design note in the module docstring: block mode keeps the N demodulated fields until selection
      (memory need documented); sliding and EMA modes can only use fixed thresholds;
    - every policy is tested; the mask is part of the result.

- [ ] K6: Entry points [phase: 1]
  - Priority: high
  - Size: M
  - Depends on: K3, K4, K5
  - Done when:
    - `reconstruct_frames(frames, background, config, *, selection, outputs, frame_info=None) ->
      ReconstructionResult` and a stateful `ReconstructionSession` (`set_background`, `push`, `result`,
      `reset`, `warmup`) exist as thin layers over `OffAxisPipeline`;
    - service (DHMConnect DL-B4) and GUI need no other import than these two names plus the
      configuration classes.

- [ ] K7: Background model [phase: 1]
  - Priority: medium
  - Size: M
  - Depends on: K6; O6 for expiry rules (mechanism now, limits configurable)
  - Done when:
    - `BackgroundModel` holds the averaged background field, the precomputed compensation factor,
      quality, id (hash), creation time and configuration hash;
    - save and load (`.npz` or HDF5) and a validity check against the active configuration are tested.

- [ ] K8: Quality numbers [phase: 1]
  - Priority: medium
  - Size: S
  - Depends on: K4; O4 for the reference region
  - Done when:
    - phase noise in the reference region after plane removal, number of frames used, piston standard
      deviation and sideband signal-to-noise ratio are in every result;
    - tested on synthetic data with known noise.

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
      `tests/benchmark_realtime.py` have run on the DHM-PC;
    - single against double precision is compared on real holograms (phase and height maps), and
      the maintainer decides from it whether `"single"` becomes the GPU default;
    - GPU, memory, CUDA version, driver, operating system, Python version and all numbers are recorded
      in `docs/realtime/benchmarks.md`;
    - closes B6 (CUDA run of the existing tests) and H4 (validate single precision); gives the input
      for O10.

### Phase 4 — Physics parameters (decision driven)

- [ ] K9: Parameter sets per objective [phase: 4]
  - Priority: high
  - Size: M
  - Depends on: O1, O2, G7 and a calibrated pixel size for 63x
  - Done when:
    - a named parameter set per objective holds explicit pixel size, wavelength, order calibration,
      crop, z, medium and reference region; name and hash go into every result;
    - the preset JSON inconsistencies (duplicate keys `xPixelSizeUm` / `pixelPitch`, `wavelengthUm` /
      `wavelength`, name `"Zeiss 63X"` against key `"Zeiss 63x"`) are cleaned without breaking the
      offline loader (golden test unchanged);
    - partial result allowed: structure and hashing done, values marked `unverified`.

- [ ] K10: z handling [phase: 4]
  - Priority: medium
  - Size: M
  - Depends on: O1
  - Done when:
    - z from the request, a list of z values (one FFT, several transfer functions), and an interface
      for numeric autofocus (metric scan over cached transfer functions) behind a flag exist and are
      tested; autofocus metric and range stay open until measured data exists.

- [ ] K11: Unwrapping in the product [phase: 4]
  - Priority: low
  - Size: S
  - Depends on: O5
  - Done when:
    - the product field `phase_unwrapped` is produced only on request; the method is recorded in the
      metadata; the default stays wrapped phase.

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
  - Depends on: O7 (G3)
  - Done when:
    - `docs/gui/DESIGN.md` covers users and use cases, screens, architecture (models without Qt,
      views, worker threads, no computation in the UI thread), data sources (file, experiment, live),
      what is taken over from the legacy GUI and what is dropped;
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
  - Depends on: L2, DL-B10, DL-C1; O8 and DHMConnect D4 (observer role while an experiment runs)
  - Done when:
    - a connection dialog and status bar (service state, queue, dropped frames, background age) work
      through `dhmconnect.client`; live hologram, spectrum, phase and intensity are shown with frame rate
      and latency; a warning appears when no service is reachable and the simulated service is used.

- [ ] L6: DHM operation panel [phase: 6]
  - Priority: medium
  - Size: L
  - Depends on: L5, DL-B8; O9
  - Done when:
    - objective and parameter set, exposure manual and automatic, OPL scan with contrast curve,
      background acquisition with quality, laser on / off with state, a fringe contrast meter for
      alignment and a snapshot (measure N frames and save with operator, notes and a name pattern) work
      against the simulated service.

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
  - Priority: low | Size: S | Depends on: G7, K9
  - Done when: the relevance of the refractive index in the DHM preset is checked and it moves to the
    material / substrate presets where needed, without breaking old presets.

- [ ] P4: Clean up `HologramProcessor` [phase: B]
  - Priority: low | Size: M | Depends on: –
  - Done when: `DHMPlotter` is a member of the processor, `set_dhm_dict` and `set_material_dict` exist,
    and the image filters that belong to MIRA are removed (after the maintainer confirms no caller needs
    them).

- [ ] P5: Mode "print" of the e-field calculation [phase: B]
  - Priority: low | Size: S | Depends on: O11
  - Done when: the mode computes what O11 decides, with a test; until then documented as unvalidated.

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

- [ ] K4: `ReconstructionResult` and `FrameRecord` [phase: 1]
  - Priority: high
  - Size: M
  - Depends on: G2 (final schema); start with the fields below
  - Done when:
    - dataclasses wrap `RealtimeResult` with: phase, intensity, optional complex field, pixel pitch
      after the sideband crop, z, wavelength, medium, a `FrameRecord` per frame (metrics, used or
      rejected, reason), quality numbers, timings, configuration hash, versions, `schema_version`;
    - `to_message()` returns a JSON-able header and a list of named NumPy buffers; `from_message()`
      reverses it;
    - a round-trip test passes and the header validates against the shared example messages
      (vendored under `tests/contracts/`).
  - Status 2026-11-16: dataclasses, `to_message()` / `from_message()` and round-trip tests done
    (`realtime/result.py`, draft `schema_version` 0.1, `43b772f`); open: validation against the vendored
    examples, see K16.


## Blocked
<!-- Format: todo as above, plus the line "Blocked by: <reason or ID>". -->

- [ ] B3: Split the dependencies (carried over, partial) [phase: 0]
  - Priority: medium
  - Size: S
  - Depends on: O10, O7, DL-C1, L1
  - Done when:
    - `pyproject.toml` has the extras `gpu`, `dev`, `gui` (PySide6, pyqtgraph; after O7) and `live`
      (`dhmconnect` client; after DL-C1); the default install stays as today (offline library works);
    - the minimum versions are installable on the DHM-PC Python (recorded in WORKLOG.md);
    - `[project.scripts]` entry `offaxisholo-gui` is added together with L1.
  - Already done: pytest configuration and markers (`04067ea`); `gpu` extra switched to `cupy-cuda13x`,
    `requires-python` ≥ 3.11 (`2c8abb5`).
  - Blocked by: Python is not yet installed on the DHM-PC (install check); `gui` extra waits for O7 / L0,
    `live` extra for DHMConnect DL-C1, the script entry for L1.

## Done
<!-- Claude Code adds: - [x] <ID>: title — YYYY-MM-DD — 1–2 sentences on what changed — commits: `<sha>`, … -->
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
