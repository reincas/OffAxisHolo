# OffAxisHolo — instructions and todos

Status: draft v0.1, 2026-11-15, waiting for maintainer approval. Read `00_SOFTWARE_PLAN_OVERVIEW.md` first.
Repository: `OffAxisHolo`, branch `Holo_HR`. Existing specification: `docs/realtime/anweisung_realtime.txt`;
existing todo list: `docs/realtime/todo_realtime.txt` (groups A–J). This plan adds groups K and L and
changes the scope of the existing list (see K1).

## 1. Role

OffAxisHolo does everything about evaluating holograms and nothing else.

- **Input:** N recorded object holograms (`uint8` / `uint16`, N × H × W or pushed one by one), a
  background (frames or a prepared background model), a configuration, optional acquisition
  information per frame (exposure, camera and host time, grab duration).
- **Output:** one `ReconstructionResult`: phase and intensity image, optional complex field, pixel
  pitch after the sideband crop, z, wavelength, medium, a record per frame (metrics, used or rejected,
  reason), quality numbers, timings (acquisition values passed through, reconstruction per stage and
  in total), configuration hash and versions.
- **Second product:** a GUI for hologram analysis, for loading experiments and for live operation of
  the DHM (group L).

Out of scope: network server, camera and other hardware drivers, service state machine, raw archive
(all `dhmlink`); registration by position, speckle fusion, stitching, target comparison (all `mira`).

## 2. Rules

1. The offline library stays unchanged and is the golden reference (`tests/test_offline_golden.py`).
   New code goes into new modules.
2. `import offaxisholo.realtime` stays lightweight (`tests/test_import_hygiene.py`). The GUI is an
   optional extra and is never imported by the core.
3. All kernels are backend-agnostic (NumPy / CuPy through `backend.py`). Exactly one thread owns the
   CUDA context. No host synchronisation in the hot path.
4. Average complex fields, never phases. Compensation before propagation, unwrapping last.
5. The pixel size never comes from Koala (B17). It is part of the parameter set.
6. Test before optimising; record measured numbers in `docs/realtime/benchmarks.md` with date, machine,
   backend and precision.
7. Create a `CLAUDE.md` for this repository as part of K1 (commands, architecture, the rules above,
   work log and commit conventions as in nanofactorysystem).

## 3. Open decisions

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| O1 | Result plane and z per objective (H1, H3): preset 0.11 µm against measured about −2.6 µm. | z comes with the request; default from the parameter set. | K9, K10 |
| O2 | Medium index and sign convention phase to height (H2). | – | K9 |
| O3 | Default frame selection policy and thresholds. | Ship `all` until the N-study (K13) is evaluated. | K3 |
| O4 | Reference region for piston phase and noise: fixed mask per objective, or automatic border ring. | Mask in the parameter set, border ring as fallback. | K8 |
| O5 | Unwrapping in the product (H5). First report: Least Squares deviates strongly on developed structures; in-situ examples need no unwrapping. | Wrapped phase by default, unwrapping optional, `fast2d` stays the default method. | K11 |
| O6 | Lifetime of a background and drift monitoring (Q8). | Measure first (Masterplan V4). | K7 |
| O7 | GUI framework (G3) and whether the GUI lives in this repository. | PySide6 + pyqtgraph; subpackage `offaxisholo.gui` with extra `[gui]`. | L0 |
| O8 | Live view: reconstruction in the service on the DHM-PC, or locally in the GUI from raw preview frames. | Service side; local reconstruction as fallback. | L5 |
| O9 | Stage control in this GUI. The legacy GUI has a positioning widget. | No. Stage control belongs to the system GUI of nanofactorysystem. | L6 |
| O10 | Minimum versions in `pyproject.toml` are very recent (numpy ≥ 2.4.5, scipy ≥ 1.17.1); CUDA version of the DHM-PC is unknown (`cupy-cuda12x` or `13x`). | Decide after B6. | B3, K14 |

## 4. Todos carried over from `todo_realtime.txt`

Keep their IDs and text. Still open: B3 `[~]` dependency split (add the extras `gui` and `live`),
B5 `[~]` repository hygiene, B6 CUDA run on the DHM-PC, C13 optimisation after profiling, H1–H6.
B5 now also covers the repository root: `todo.txt` carries no status (the maintained list is
`docs/realtime/todo_realtime.txt`), `anweisung.txt` is identical to `docs/realtime/anweisung_realtime.txt`,
and four files are untracked. Ask the maintainer before removing or moving any of them.

## 5. Group K — product interface

- [ ] **K1 Update scope and documents (S).** Mark groups D (camera), E (service), F (client) and G
  (operation) in `todo_realtime.txt` as moved to `dhmlink` with the new IDs (`DL-B…`, `DL-C…`, `DL-H…`).
  Add pointers in `anweisung_realtime.txt` §4, §7, §8, §10. Write `CLAUDE.md`.
  Done when: no todo in this repository asks for network or camera code.

- [ ] **K2 Frame metrics (M).** New module `realtime/metrics.py`: saturated and black pixel counts,
  median and 5 % / 95 % quantiles, sideband peak weight and position (from the FFT that demodulation
  computes anyway), piston phase and phase noise in the reference region. Values stay on the device
  until the result is built.
  Done when: tests with synthetic frames (saturation, shifted carrier, reduced fringe contrast) pass
  on NumPy; CuPy parity test exists and is skipped without CUDA.

- [ ] **K3 Frame selection (M).** New module `realtime/selection.py`: pure function from a metrics
  table to a mask and a reject reason per frame. Policies `all`, `threshold`, `robust` (median / MAD),
  `best_k`; `min_frames` is always enforced and reported.
  Design note: block mode keeps the N demodulated fields until selection (document the memory need);
  sliding and EMA modes can only use fixed thresholds.
  Needs decision O3 for the default. Done when: every policy is tested, the mask is part of the result.

- [ ] **K4 `ReconstructionResult` and `FrameRecord` (M).** Dataclasses wrapping `RealtimeResult` with
  the fields of section 1. `to_message()` returns a JSON-able header and a list of named NumPy
  buffers; `from_message()` reverses it. Carries `schema_version`.
  Needs G2. Done when: round-trip test passes and the header validates against the shared example
  messages (vendored under `tests/contracts/`).

- [ ] **K5 Timings (S).** Stage timers for upload, demodulation, averaging, compensation and
  propagation, download and total. GPU-synchronised timing only with `profile=True`. Acquisition
  values supplied by the caller are passed through unchanged.
  Done when: `result.timings_s` is filled and a test checks that the stage sum does not exceed the total.

- [ ] **K6 Entry points (S–M).** `reconstruct_frames(frames, background, config, *, selection,
  outputs, frame_info=None) -> ReconstructionResult` and a stateful `ReconstructionSession`
  (`set_background`, `push`, `result`, `reset`, `warmup`). Both are thin layers over `OffAxisPipeline`.
  Done when: service and GUI need no other import than these two names plus the configuration classes.

- [ ] **K7 Background model (S–M).** `BackgroundModel`: averaged background field, the precomputed
  compensation factor, quality, id (hash), creation time, configuration hash. Save and load (`.npz`
  or HDF5), validity check against the active configuration.
  Needs O6 for expiry rules; implement the mechanism now, leave the limits configurable.

- [ ] **K8 Quality numbers (S).** Phase noise in the reference region after plane removal, number of
  frames used, piston standard deviation, sideband signal-to-noise ratio.
  Needs O4. Done when: values are in every result and tested on synthetic data with known noise.

- [ ] **K9 Parameter sets (M).** Named parameter set per objective with explicit pixel size,
  wavelength, order calibration, crop, z, medium, reference region; name and hash go into every
  result. Clean the preset JSON (duplicate keys `xPixelSizeUm` / `pixelPitch`, `wavelengthUm` /
  `wavelength`, name `"Zeiss 63X"` against key `"Zeiss 63x"`) without breaking the offline loader.
  Needs O1, O2, G7 and a calibrated pixel size for 63x. Partial result allowed: structure and
  hashing done, values marked `unverified`.

- [ ] **K10 z handling (S–M).** z from the request, a list of z values, and an interface for numeric
  autofocus (metric scan over cached transfer functions) behind a flag.
  Needs O1. Autofocus metric and range stay open until measured data exists.

- [ ] **K11 Unwrapping in the product (S).** Product field `phase_unwrapped` only on request; method
  recorded in the metadata. Needs O5.

- [ ] **K12 Replay reader (M).** `offaxisholo.io.experiment`: read hologram stacks, device
  parameters and products from `experiment.h5` and from raw archive files of `dhmlink`, with `h5py`
  only (schema: `docs/design/EXPERIMENT_STORAGE.md` in nanofactorysystem; read without HDF5 file
  locking). `tools/replay_experiment.py` reconstructs captures from it.
  Done when: a test builds a small file with the documented layout and replays it.

- [ ] **K13 N-study (S).** Extend `tools/noise_vs_n.py`: noise in the reference region against the
  number of frames for subsets and for each selection policy, from a replayed capture.
  Depends on K2, K3, K12. The evaluation on real data needs captures with `keep_raw = all`.

- [ ] **K14 GPU validation on the DHM-PC (S, hardware).** Run `pytest tests/test_backend.py -v`,
  `tests/realtime`, both benchmarks; compare single against double precision on real holograms.
  Record GPU, memory, CUDA version and all numbers. Closes B6 and H4.

- [ ] **K15 Public API and documentation (S).** Exports of `offaxisholo.realtime`, README section,
  `__version__` and a helper that returns the git hash for the result metadata.

## 6. Group L — GUI

Package `offaxisholo.gui`, entry point `offaxisholo-gui`, extras `[gui]` and `[live]`.
Legacy to review, not to extend: repository `DHM_User_Interface` (PyQt5; `Reconstruction/` with
`front_end.py` and `panels.py` for offline reconstruction, `DHM_GUI/` for capture with stage control,
wish list in its `todo.txt`).

- [?] **L0 Design document (M).** `docs/gui/DESIGN.md`: users and use cases, screens, architecture
  (models without Qt, views, worker threads, no computation in the UI thread), data sources (file,
  experiment, live), what is taken over from the legacy GUI and what is dropped.
  Needs G3 / O7. Ends with the maintainer's approval; L1–L8 start only afterwards.

- [ ] **L1 Application skeleton (M).** Main window with dock layout, persistent settings, log pane,
  worker infrastructure, operator name, import hygiene test for the core.

- [ ] **L2 Image views (M–L).** Hologram, spectrum (log magnitude) with sideband marker and mask
  overlay, phase, intensity. Linked zoom and pan, colour maps, colour bar with units, histogram, ROI
  statistics, line profile, two views side by side.

- [ ] **L3 Offline analysis (L).** Open hologram and background files (tif, png, zdc). Parameter
  panel bound to the configuration classes: parameter set, order calibration, crop, mask,
  regularisation, z slider that re-propagates from the cached compensated field, unwrapping method.
  Export images and arrays with a JSON metadata file that contains the configuration hash.

- [ ] **L4 Experiment browser (L).** Open `experiment.h5` read-only through K12. Tree substrate →
  experiment → structure → capture with phase and layer. Show stored products, reconstruct again from
  stored holograms or from the raw archive, compare frame counts and selection policies, batch export.

- [ ] **L5 Live mode (L).** Connection dialog and status bar (service state, queue, dropped frames,
  background age) through `dhmlink.client`. Live hologram, spectrum, phase and intensity with frame
  rate and latency display. A warning when no service is reachable and the simulated service is used.
  Depends on DL-B10, DL-C1. Needs O8 and D4 of the dhmlink plan (observer role while an experiment runs).

- [ ] **L6 DHM operation panel (M–L).** Objective and parameter set, exposure manual and automatic,
  OPL scan with contrast curve, background acquisition with quality, laser on / off with state,
  fringe contrast meter for alignment, snapshot (measure N frames and save with operator, notes and a
  name pattern). Depends on DL-B8, DL-C1. Needs O9.

- [ ] **L7 Tests (M).** Model tests without Qt; smoke tests with `pytest-qt` in offscreen mode; the
  live mode is tested against `dhmlink.testing`.

- [ ] **L8 User guide (S).** Short guide with screenshots in `docs/gui/`.

## 7. Suggested order

K1 → K4, K5 → K2, K3 → K6, K7, K8 → K12, K13 → K14 (needs the DHM-PC) → K9–K11 as decisions arrive → K15.
GUI: L0 → approval → L1, L2 → L3 → L4 → L5, L6 (need `dhmlink`) → L7, L8.
