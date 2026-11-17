# Work Log

Append-only log of completed and partial tasks. Newest entry at the bottom.
Format and rules: see "Work log (mandatory)" in CLAUDE.md. Work before 2026-11-15 is documented in
`docs/realtime/todo_realtime.txt` (status block) and in the git history of `Holo_HR`.

### 2026-11-15 23:15 CET — [K1] Project files: CLAUDE.md, TODO.md, system plan
- **Status:** partial (the pointers in `docs/realtime/*.txt` remain open in K1)
- **Context:** Maintainer request: create a `CLAUDE.md` in the style of NanoFactorySystem_automated and
  sort all todos into one list. Inputs: `todo_claude/01_OFFAXISHOLO_PLAN.md`,
  `00_SOFTWARE_PLAN_OVERVIEW.md`, `anweisung.txt` (= `docs/realtime/anweisung_realtime.txt`), `todo.txt`
  (older copy of `docs/realtime/todo_realtime.txt` without the status block), `uebergabe_echtzeit_dhm.md`,
  `Notes_guteHandhabung.txt`, `OffAxisHolo_projectStructure_18052026.txt`, `src/offaxisholo/ToDos.txt`,
  plus the as-is analysis of the DHM interface and the DHM service architecture draft.
- **Decisions taken with the maintainer:** the communication repository is DHMConnect with the package
  `dhmconnect` (G1); the system plan is copied into every repository, master copy in
  NanoFactorySystem_automated; all four repositories use the TODO.md format of nanofactorysystem.
- **Changes:**
  - `CLAUDE.md` (new): overview, commands, architecture (backend, offline library, realtime kernel),
    system context, repository rules (golden reference, import hygiene, hot path, layer separation),
    maintainer instructions (English, work log, Definition of Done, todos, commits, tests).
  - `TODO.md` (new): phases 0–6 and B with K1–K15, L0–L8, carried-over B3, B5, C13, new package backlog
    P1–P15 (from `ToDos.txt`, plus P15 requested by MIRA), decisions O1–O11, mapping of every item of
    `todo_realtime.txt` (groups D–G moved to DHMConnect), done items with commits.
  - `docs/system/SOFTWARE_PLAN_OVERVIEW.md` (new): copy of the system plan v0.2.
  - `WORKLOG.md` (new): this file.
- **Tests:** none (documentation only).
- **Commits:** uncommitted — proposed messages (one per topic, on a branch `docs/project-files` from `Holo_HR`):
  1. `docs(system): add the plan for the four cooperating repositories [K1]`
     Files: `docs/system/SOFTWARE_PLAN_OVERVIEW.md` (new)
  2. `docs: add CLAUDE.md with architecture, rules and commit conventions [K1]`
     Files: `CLAUDE.md` (new)
  3. `docs(todo): collect all open todos in TODO.md and map the old realtime list [K1]`
     Files: `TODO.md` (new), `WORKLOG.md` (new)
- **Follow-ups:** K1 (pointers in `docs/realtime/`), B5 (untracked `Claude outputs/` and `todo_claude/`,
  legacy tests, `ToDos.txt` inside the package).

### 2026-11-16 09:26 CET — [K1] Finish the scope change in the realtime documents
- **Status:** done
- **Context:** The project files of 2026-11-15 were committed by the maintainer in `b247181`
  ("Organizing todos"); the commits proposed in the previous entry are therefore not needed.
  Work branch: `docs/k1-scope` from `Holo_HR`.
- **Changes:**
  - `docs/realtime/todo_realtime.txt`: English note at the top: historical, `TODO.md` is the
    maintained list with the mapping table, groups D/E/F/G moved to DHMConnect (DL-B…, DL-C…, DL-H…).
    CRLF line endings kept; German text unchanged.
  - `docs/realtime/anweisung_realtime.txt`: English pointer notes at the start of §4 (components:
    camera/service/client → DL-B1, B2, B3, B4, B5, B7, DL-C1, DL-H2), §7 (state machine → DL-B3,
    DL-B10), §8 (protocol → DL-A2, DL-A4, DL-C1, D4, D5; contracts G2) and §10 (camera abstraction →
    DL-B1, DL-H2; kernel tests use `tests/realtime/synthetic_dhm.py`). No other change.
  - `TODO.md`: K1 moved to Done.
  - `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 90 passed, 8 skipped (GPU, no CUDA). Documentation-only change.
  Checked `TODO.md`: no todo asks for network or camera code (L5 only uses `dhmconnect.client`).
- **Commits:** `ee39f40` docs(realtime): mark the old todo list historical, point to DHMConnect [K1];
  todo bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** none. Next todo of phase 0: B5 (needs maintainer decisions), B3 (needs O10).

### 2026-11-16 14:24 CET — [K16, K18] Handover notes for DHMConnect and nanofactorysystem
- **Status:** done
- **Context:** Maintainer request: one document each for nanofactorysystem and DHMConnect with all
  decisions, information and new features of this session that concern them.
- **Changes:**
  - `pyproject.toml`: `gpu` extra `cupy-cuda13x>=13.6` (first CuPy for CUDA 13); `requires-python`
    `>=3.11` (numpy 2.4.5 / scipy 1.17.1 have no 3.10 wheels); ruff `target-version` py311.
  - `README.md`: GPU install note (CUDA 13, Python 3.11+).
  - `CLAUDE.md`: Python version, gpu extra, DHM-PC facts (CUDA 13, Python not installed yet).
  - `TODO.md`: B3 moved to Blocked with the remaining items; O10 updated (partly decided).
  - `WORKLOG.md`: this entry.
- **Tests:** `pip download --only-binary=:all: --platform win_amd64` of all minimum versions plus
  `cupy-cuda13x==13.6.0` succeeds for Python 3.11, 3.12 and 3.13; `python -m pytest`: 90 passed,
  8 skipped; `ruff check .` clean.
- **Commits:** `2c8abb5` build: target CUDA 13 and Python 3.11+ for the DHM-PC [B3]; todo bookkeeping in
  the following `docs(todo)` commit.
- **Follow-ups:** after Python is installed on the DHM-PC: `pip install -e ".[gpu,dev]"` and record the
  result here (closes the install item of B3, input for K14). Found: the local black 26.5 and the
  pre-commit black 24.10 format multi-line f-strings differently (`tests/test_import_hygiene.py`), and
  black 26 warns about Python 3.15 targets; the pre-commit hook is the reference, nothing changed.

### 2026-11-16 11:49 CET — [Phase 0] Phase summary
- **Finished:** K1 (scope change of the realtime documents), B5 (repository hygiene).
- **Blocked:** B3 (install check needs Python on the DHM-PC; `gui` extra waits for O7, `live` for
  DHMConnect DL-C1, script entry for L1). CUDA 13 and `requires-python` ≥ 3.11 are done (`2c8abb5`).
- **Tests:** `python -m pytest`: 90 passed, 8 skipped (GPU) at the end of phase 0.
- **Branches (stacked, merge in this order into `Holo_HR`):** `docs/k1-scope` → `chore/b5-hygiene` →
  `build/b3-dependencies` → `feat/k4-result`.
- **For the maintainer on the DHM-PC:** install Python 3.12, then `pip install -e ".[gpu,dev]"` and
  `python -m pytest -m gpu`; report the result (B3, K14).
- **Decisions still open:** G2 (home of the contracts) for K4 / K16; record G9 "kept knowingly" in the
  master system plan.

### 2026-11-16 11:49 CET — [K4] ReconstructionResult and FrameRecord
- **Status:** partial (validation against the vendored contract examples open: K16, needs G2 / DL-A3)
- **Context:** Phase 1 started. DHMConnect has no `schemas/` yet (only `TODO.md` with DL-A3); G2 is still
  a recommendation. Independent part implemented with the fields listed in K4 and the conventions of
  `docs/system/SOFTWARE_PLAN_OVERVIEW.md` §4. Branch `feat/k4-result` from `build/b3-dependencies`.
- **Design decisions:**
  - Units in the names and µm in the result (`pixel_pitch_um`, `z_um`, `wavelength_um`) as the contract
    conventions require; the kernel config keeps metres.
  - Phase and linear intensity as float32, field complex64 only with `include_field=True`.
  - Metadata (quality, timings, versions, config) is stored in JSON form, so result and round trip
    compare equal; NaN becomes `null`.
  - Wire format: JSON header with `message_type`, `schema_version` (draft `0.1`) and buffer descriptors
    (name, dtype, shape, byte order); buffers C-contiguous little-endian. `from_message` accepts arrays,
    raw bytes or (name, payload) pairs, rejects other major versions, ignores unknown keys.
  - `frames` stays empty unless given: the entry points (K6) fill one record per frame; K2/K3/K5/K8
    fill metrics, selection, timings and quality.
- **Changes:**
  - `src/offaxisholo/realtime/result.py` (new): `ReconstructionResult`, `FrameRecord`, `default_versions`.
  - `src/offaxisholo/realtime/__init__.py`: lazy exports `ReconstructionResult`, `FrameRecord`.
  - `tests/realtime/test_result.py` (new): 15 tests plus a CuPy parity test (skipped without CUDA).
  - `tests/test_import_hygiene.py`: `offaxisholo.realtime.result` added to the light modules.
  - `TODO.md`: K4 to In Progress with status; new K16 (follow-up of K4); phase table.
  - `WORKLOG.md`: this entry and the phase 0 summary.
- **Tests:** `python -m pytest`: 106 passed, 9 skipped (GPU); `ruff check .` clean.
- **Commits:** `43b772f` feat(realtime): add ReconstructionResult and FrameRecord [K4]; todo bookkeeping
  in the following `docs(todo)` commit.
- **Follow-ups:** K16; decision G2. Found: the local black 26.5 and the pre-commit black 24.10 format
  some constructs differently; the pre-commit hook is the reference.

### 2026-11-16 12:11 CET — [K5] Stage timings
- **Status:** done
- **Context:** Maintainer decisions of 2026-11-16: G2 as recommended (JSON Schema plus example messages
  in `dhmconnect/schemas/`); continue with phase 1, ask explicitly when a decision is due. Branch
  `feat/k5-timings` from `feat/k4-result`.
- **Design decisions:**
  - Seven cumulative stages (upload, demodulation, piston, averaging, compensation, propagation, download)
    plus `total` = first push to last timed step. The total includes the gaps between frames, so the
    stage sum cannot exceed it. "download" includes deriving phase and intensity.
  - `profile` is a constructor argument of `OffAxisPipeline`, not configuration (does not change the
    result or the config hash). Without it the timer never touches the device; GPU stage times are then
    enqueue times. With it, the current CUDA stream is synchronised before every clock reading.
  - The flag travels as `ReconstructionResult.timings_profiled` (not inside `timings_s`, which holds
    seconds only); draft `schema_version` 0.1 → 0.2.
  - Caller values: per frame in `FrameRecord.acquisition`, per measurement via
    `from_realtime(timings_s=...)`; both pass through unchanged (tested through a JSON round trip).
- **Changes:**
  - `src/offaxisholo/realtime/timing.py` (new): `StageTimer`, `STAGES`.
  - `src/offaxisholo/realtime/core.py`: `OffAxisPipeline.timer`, `profile` argument, timed `push` and
    `results`, `reset` clears the timings.
  - `src/offaxisholo/realtime/result.py`: download timing, pipeline timings plus caller values,
    `timings_profiled`, schema 0.2.
  - `tests/realtime/test_timing.py` (new): 11 tests plus a GPU test (skipped without CUDA).
  - `tests/test_import_hygiene.py`: `offaxisholo.realtime.timing` added.
  - `CLAUDE.md`: realtime modules `result.py`, `timing.py`; G2 decided.
  - `TODO.md`: G2 decided; K5 to Done.
  - `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 118 passed, 10 skipped (GPU); `ruff check .` clean; static hot-path
  check of `push` passes.
- **Commits:** `c66838f` feat(realtime): add stage timings to the kernel and the result [K5];
  `388bdc0` docs: describe result.py and timing.py and the G2 decision in CLAUDE.md [K5]; todo
  bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** record G2 in the master system plan (NanoFactorySystem_automated); tell DHMConnect
  (DL-A3) that `realtime/result.py` (schema 0.2) is the proposed starting point for the
  `ReconstructionResult` contract.

### 2026-11-16 12:18 CET — [K2] Frame metrics
- **Status:** done
- **Context:** Branch `feat/k2-metrics` from `feat/k5-timings`.
- **Design decisions:**
  - Eleven metrics per frame (`METRIC_NAMES`): saturated / black counts, 5 / 50 / 95 % intensity
    quantiles (camera counts), sideband energy fraction and peak fraction (relative to the total spectral
    energy incl. DC via Parseval, so incoherent light lowers it), peak position relative to the calibrated
    order (px), piston phase against the background, phase noise in the reference region.
  - Phase noise uses `compensation.ref_border` and is NaN without it; the default reference region is
    decision O4 (K8).
  - Quantiles via `xp.quantile`; `cupy.bincount` would read the maximum on the host. Optimisation only
    after the K14 profile (C13).
  - `MetricsConfig` (enabled, saturation_level 255, black_level 0, capacity) is off by default; the new
    config part changes every config hash (no stored hashes exist).
  - Device ring buffer (capacity default `max(n_obj, window)`); one host copy per result.
  - Metrics do not change the field (bit-identical test with metrics on / off).
- **Changes:**
  - `src/offaxisholo/realtime/metrics.py` (new): metric functions, `MetricsRecorder`, `METRIC_NAMES`.
  - `src/offaxisholo/realtime/config.py`: `MetricsConfig` and its validation.
  - `src/offaxisholo/realtime/core.py`: `_demodulate_parts`, metrics in `push`, `frame_metrics()`, reset.
  - `src/offaxisholo/realtime/result.py`: `FrameRecord`s from the metrics unless the caller passes some.
  - `src/offaxisholo/realtime/timing.py`: stage `metrics`.
  - `src/offaxisholo/realtime/__init__.py`: lazy exports `MetricsConfig`, `MetricsRecorder`, `METRIC_NAMES`.
  - `tests/realtime/test_metrics.py` (new): 17 tests plus CuPy parity with `allow_synchronize(False)`.
  - `tests/test_import_hygiene.py`: `offaxisholo.realtime.metrics` added.
  - `tests/benchmark_realtime.py`: `--metrics` option (smoke run on the laptop only, numbers not recorded).
  - `CLAUDE.md`: module `metrics.py` described.
  - `TODO.md`: K2 to Done; K14 runs the benchmark with and without `--metrics`.
  - `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 136 passed, 11 skipped (GPU); `ruff check .` clean; static hot-path
  check of the metric functions passes.
- **Commits:** `aab56d6` feat(realtime): add per-frame metrics from the demodulation FFT [K2];
  `efdda6c` test(realtime): add a --metrics option to the realtime benchmark [K2];
  `25b6303` docs: describe the frame metrics module in CLAUDE.md [K2]; todo bookkeeping in the
  following `docs(todo)` commit.
- **Follow-ups:** decisions O3 (K3), O4 (K8), O6 (K7) and "metrics on by default in the entry points"
  (K6) asked from the maintainer.

### 2026-11-16 12:29 CET — [K3] Frame selection
- **Status:** done
- **Context:** Maintainer decisions of 2026-11-16 (asked explicitly): O3 default policy `all` until K13;
  frame metrics on by default in the K6 entry points; O4 mask per objective with border-ring fallback;
  O6 background age only as a warning (default: no limit). Recorded in `TODO.md` (`794a799`). Branch
  `feat/k3-selection` from `feat/k2-metrics`.
- **Design decisions:**
  - `select_frames` is a pure host function run once per measurement on `frame_metrics()`.
  - `threshold`: NaN in a bounded metric rejects ("not available"), so a missing reference region is
    visible. `robust`: `|x − median| > k · 1.4826 · MAD` (k = 3.5); constant (MAD = 0) or missing metrics
    are not tested. `best_k`: by `rank_by` (default `phase_noise_rad`, lower is better), NaN last, ties by
    index.
  - `min_frames` re-adds the best rejected frames; their reason keeps the original text with the prefix
    "forced by min_frames: "; the quality summary reports policy, totals, `min_frames` and whether it was
    enforced.
  - Block mode: `OffAxisPipeline(..., keep_fields=True)` stores every corrected field in a fixed device
    stack (`FieldStore`, capacity `n_obj`, allocated once); `apply_selection(mask)` rebuilds the average
    and may be called again. Memory documented in the module docstring. Sliding / EMA are refused
    (follow-up K17: device-side gating).
- **Changes:**
  - `src/offaxisholo/realtime/selection.py` (new): `SelectionPolicy`, `Selection`, `select_frames`.
  - `src/offaxisholo/realtime/accumulators.py`: `FieldStore`.
  - `src/offaxisholo/realtime/core.py`: `keep_fields`, `apply_selection`, field store in `push` / `reset`.
  - `src/offaxisholo/realtime/result.py`: `from_realtime(selection=...)`.
  - `src/offaxisholo/realtime/__init__.py`: lazy exports `Selection`, `SelectionPolicy`, `select_frames`.
  - `tests/realtime/test_selection.py` (new): 22 tests.
  - `tests/test_import_hygiene.py`: `offaxisholo.realtime.selection` added.
  - `CLAUDE.md`: module `selection.py` described.
  - `TODO.md`: O3, O4, O6 decided; K3 to Done; new K17 (follow-up of K3).
  - `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 159 passed, 11 skipped (GPU); `ruff check .` clean. Two test expectations
  were wrong at first (test data with MAD = 0; tie order) and were corrected, not the code.
- **Commits:** `794a799` docs(todo): record the maintainer decisions O3, O4 and O6 [K3];
  `3a7c520` feat(realtime): add frame selection with policies and a field store [K3];
  `4af6182` docs: describe the frame selection module in CLAUDE.md [K3]; todo bookkeeping in the
  following `docs(todo)` commit.
- **Follow-ups:** K17. Next: K6 (entry points), then K7, K8.

### 2026-11-16 12:32 CET — [K6] Entry points
- **Status:** done
- **Context:** Branch `feat/k6-entry-points` from `feat/k3-selection`.
- **Design decisions (no open decision touched):**
  - `outputs` always contains phase and intensity; `"field"` adds the complex field (K11 will add
    `phase_unwrapped`).
  - `metrics=True` by default (decision 2026-11-16) overrides `config.metrics.enabled`; the stored
    config shows the effective value. Policy `all` by default (O3); other policies need metrics and the
    block mode (fields kept), sliding / EMA gating is K17.
  - `reconstruct_frames` sets the averaging mode to block and `n_obj` to the number of frames (one
    measurement); the stored config shows it.
  - Acquisition info per frame is passed through unchanged; the session keeps it for the last
    `max(n_obj, window)` frames, like the metrics.
- **Changes:**
  - `src/offaxisholo/realtime/api.py` (new): `reconstruct_frames`, `ReconstructionSession`.
  - `src/offaxisholo/realtime/__init__.py`: lazy exports and docstring pointer.
  - `tests/realtime/test_api.py` (new): 14 tests plus a GPU test.
  - `tests/test_import_hygiene.py`: `offaxisholo.realtime.api` added.
  - `CLAUDE.md`: module `api.py` described.
  - `TODO.md`: K6 to Done.
  - `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 174 passed, 12 skipped (GPU); `ruff check .` clean.
- **Commits:** `4cac2bd` feat(realtime): add the entry points reconstruct_frames and
  ReconstructionSession [K6]; `90cdd5a` docs: describe the entry points module in CLAUDE.md [K6]; todo
  bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** tell DHMConnect that DL-B4 can build on `ReconstructionSession`. Next: K7, K8.

### 2026-11-16 12:38 CET — [K7] Background model
- **Status:** done
- **Context:** Branch `feat/k7-background` from `feat/k6-entry-points`. Decision O6 (2026-11-16): no
  expiry, optional maximum age only as a warning.
- **Design decisions:**
  - Validity uses a separate `RealtimeConfig.background_hash()` (dhm, demod, eps_rel, piston,
    precision), not the full config hash: changing z, averaging, metrics, the object reference border or
    the backend keeps a background valid.
  - Content id = first 16 hex digits of sha256(background hash, creation time, field); checked on every
    construction, so a changed model or file is rejected.
  - `.npz` with a JSON meta text, `allow_pickle=False` on save and load (no code execution from files).
  - Results carry `background_id`, `background_created_utc`, `background_age_s` in quality and the new
    `warnings` list (draft schema 0.3).
- **Changes:**
  - `src/offaxisholo/realtime/background.py` (new): `BackgroundModel`, `model_id`, `utc_now`.
  - `src/offaxisholo/io/background.py` (new): `save_background`, `load_background`.
  - `src/offaxisholo/realtime/config.py`: `background_hash()`.
  - `src/offaxisholo/realtime/core.py`: `background_model()`, `use_background()`.
  - `src/offaxisholo/realtime/result.py`: `warnings`, background quality, schema 0.3, `to_jsonable`
    public.
  - `src/offaxisholo/realtime/api.py`: `use_background`, `background_model`, `background_max_age_s`,
    `BackgroundModel` accepted as background.
  - `src/offaxisholo/realtime/__init__.py`: lazy export `BackgroundModel`.
  - `tests/realtime/test_background.py` (new): 11 tests (all files under `tmp_path`).
  - `tests/test_import_hygiene.py`: `realtime.background`, `io.background` added.
  - `CLAUDE.md`: module described. `TODO.md`: K7 to Done. `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 187 passed, 12 skipped (GPU); `ruff check .` clean.
- **Commits:** `d2f5342` feat(realtime): add a reusable background model with save, load and checks
  [K7]; `a8c9a9e` docs: describe the background model in CLAUDE.md [K7]; todo bookkeeping in the
  following `docs(todo)` commit.
- **Follow-ups:** K8 needs two details of O4 from the maintainer (asked): format of the reference mask
  and whether the border ring also becomes the default region of the piston correction.

### 2026-11-16 13:19 CET — [K18] Stitching and averaging problem document; K8 blocked
- **Status:** partial (K18 is a planning todo; the document is written, the decisions are open)
- **Context:** Before K8 the maintainer asked what "piston phase" and "reference mask" mean. After the
  explanation: structures differ from print to print, large structures will be stitched, MIRA detects
  structures with Otsu and sets the background to 0 (this already levels the absolute offset). The
  maintainer finds the a priori mask from the print job very sensible and the a posteriori mask from MIRA
  useful; detection, combination of holograms and stitching will be worked on intensively later. A fixed
  mask per objective (O4 of the same morning) therefore does not hold.
- **Findings (details in the document):** per-frame alignment before averaging (OffAxisHolo) and levelling
  of the final map / tile offsets (MIRA) are different steps; for a static structure the whole-field
  alignment is good enough, it fails when the structure changes during the measurement; the noise measure
  needs a per-capture mask; averaging within a position belongs here, across positions to MIRA (MI-30).
- **Changes:**
  - `docs/stitching_and_averaging.md` (new): terms, current state of OffAxisHolo / MIRA / nanofactorysystem,
    findings, problems P-A–P-F, decisions SA1–SA7 with proposals (not decided).
  - `CLAUDE.md`: pointer to the document.
  - `TODO.md`: O4 reopened and superseded by SA1–SA7; new K18 (planning session, found during K8); K8
    moved to Blocked (blocked by K18) with a note on what is already in every result; K4 status shows
    schema 0.3; phase table.
  - `WORKLOG.md`: this entry and the session summary below.
- **Tests:** `python -m pytest`: 187 passed, 12 skipped (GPU); `ruff check .` clean. No code changed.
- **Commits:** `5243359` docs: collect the reference mask, averaging and stitching problems [K18];
  todo bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** planning session K18 with access to all four repositories (start Claude Code in the
  common parent folder or use `/add-dir`).

### 2026-11-16 13:19 CET — Session summary (end of session)
- **Done today:** K1, B5, K5, K2, K3, K6, K7. **In Progress:** K4 (contract validation, K16).
  **Blocked:** B3 (Python on the DHM-PC, O7, DL-C1, L1), K8 (K18). **New:** K16, K17, K18, P16, P17.
- **Decisions of the maintainer today:** B5 items (attic, notes versioned, G9 kept knowingly, `attic/`
  stays ignored); CUDA 13 on the DHM-PC, Python to be installed there; G2 as recommended; O3 `all`;
  metrics on in the entry points; O6 warning only; O4 decided and reopened the same day (→ K18).
- **Branches (stacked, merge in this order into `Holo_HR`; nothing pushed):** `docs/k1-scope` →
  `chore/b5-hygiene` → `build/b3-dependencies` → `feat/k4-result` → `feat/k5-timings` →
  `feat/k2-metrics` → `feat/k3-selection` → `feat/k6-entry-points` → `feat/k7-background`.
  The last branch contains all commits of the session.
- **Tests at the end:** `python -m pytest`: 187 passed, 12 skipped (GPU tests, no CUDA on this machine).
- **For the maintainer:**
  - record G2 (decided) and G9 (kept knowingly) in the master system plan (NanoFactorySystem_automated);
  - tell DHMConnect that `realtime/result.py` (draft schema 0.3) is the proposed starting point for the
    `ReconstructionResult` contract (DL-A3) and that DL-B4 can build on `ReconstructionSession`;
  - on the DHM-PC: install Python 3.12, `pip install -e ".[gpu,dev]"`, `python -m pytest -m gpu`,
    `python tests/benchmark_realtime.py 1024 --metrics` and without `--metrics` (B3, K14);
  - local black 26.5 and the pre-commit black 24.10 format some constructs differently; the hook is the
    reference.
- **Next todos:** K18 (planning session), then K8; phase 2 (K12 replay reader, K15 public API) can start
  independently.

### 2026-11-17 15:58 CET — [P18] Translate the remaining German documentation in active code
- **Status:** done
- **Changes:**
  - `src/offaxisholo/utils.py`: comment in `get_logger` translated.
  - `src/offaxisholo/reconstruction/hologram_class.py`: docstring of `_locate_order` translated and made
    descriptive.
  - `TODO.md`: P18 added directly to Done.
- **Tests:** `python -m pytest`: 187 passed, 12 skipped (GPU / optional), unchanged; pre-commit black and
  ruff passed.
- **Commits:** `ca07542` docs(utils): translate the last German comment and docstring [P18]; bookkeeping in
  the following `docs(todo)` commit.
- **Follow-ups:** German that stays on purpose: the maintainer's notes in `src/offaxisholo/io/structure.py`
  (lines 6, 9), the commented-out `print("Gleich")` in `hologram_class.py`,
  `docs/realtime/reference/holo_gpu_pipeline.py` (verbatim), `docs/realtime/anweisung_realtime.txt` and
  `todo_realtime.txt`, the attic. The cross-repository overview of this session lives in `../_system/`.
