# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`offaxisholo` is a Python package (src layout, Python ≥ 3.11, developed with 3.12) for the reconstruction and simulation of off-axis holograms recorded with a transmissive digital holographic microscope (DHM, LyncéeTec, 674.99 nm laser). It has two parts:

- the **offline library** (`Hologram`, `HologramProcessor`, `ReconstructionPipeline`, `reconstruct_hologram`, …): CPU or GPU, the **golden reference** for everything new;
- the **realtime kernel** `offaxisholo.realtime` (`OffAxisPipeline`): reconstructs a stream of camera frames on the GPU (CuPy) or CPU (NumPy) and averages several object and background holograms. It becomes the computational core of the DHM service on the DHM-PC (service and protocol live in DHMConnect).

Planned (see `TODO.md`): a product interface around the realtime kernel (frame metrics, frame selection, `ReconstructionResult`, timings, background model, parameter sets, replay reader; group K) and a GUI for hologram analysis and live DHM operation (group L).

Repository: upstream `github.com/reincas/OffAxisHolo` (Reinhard Caspary, MIT licence). The maintainer works on the branch `Holo_HR`; `main` belongs to the upstream author.

Project files:
- `TODO.md`: the only maintained task list (sections Open / In Progress / Blocked / Done; format rules and phase plan at the top). It also maps every item of the earlier list `docs/realtime/todo_realtime.txt` to its new place.
- `WORKLOG.md`: append-only work log (see "Work log").
- `docs/system/SOFTWARE_PLAN_OVERVIEW.md`: the plan for the four cooperating repositories (copy; master in NanoFactorySystem_automated).
- `docs/realtime/anweisung_realtime.txt` (German): specification of the realtime kernel and the originally planned service — the WHAT and WHY, including the mathematical processing chain (§3), hot-path rules (§6) and gates (§11). Still valid for the kernel; its service, camera and protocol parts moved to DHMConnect.
- `docs/realtime/todo_realtime.txt` (German): the change list of 2026-11-13 (groups A–J) with its status. Historical: do not maintain it, use `TODO.md`.
- `docs/realtime/unwrapping_comparison.md`: first report Fast 2D vs. Least Squares (G1.8).
- `docs/realtime/reference/holo_gpu_pipeline.py`: CPU-validated reference skeleton, kept verbatim (excluded from black and ruff).
- `attic/`: archived material (B5): `legacy_tests/` (the former `tests/test/` scripts and their data), `ToDos.txt` (the maintainer's old todo notes; their items are in `TODO.md`, group P), `planning_2026-11-15/` (the planning inputs of `CLAUDE.md`, `TODO.md` and `docs/system/`). `attic/` is in `.gitignore`: files moved there with `git mv` stay versioned, new files there stay local.
- `docs/stitching_and_averaging.md`: open problem collection on reference masks, piston phase, averaging within and across positions and stitching; decisions SA1–SA7, **decided 2026-11-18** (G10; results in master plan v0.4 §6 and the "Decided" table of `TODO.md`).
- `docs/reconstruction_order.md`: why compensation happens in the hologram plane before propagation, the padding trick of the propagation, numerical tests and the focus measured on real data (formerly the local `Notes_reconstruction_order.txt`; original in `attic/notes/`).

## Commands

```bash
python -m pip install -e ".[dev]"     # editable install with pytest, black, ruff
python -m pip install -e ".[gpu]"     # CuPy for CUDA 13 (cupy-cuda13x, DHM-PC); with a CUDA 12 driver install cupy-cuda12x instead
python -m pytest                      # all tests; GPU tests are skipped without CUDA
python -m pytest tests/realtime       # realtime kernel (gates G1.1–G1.6 on the CPU)
python -m pytest tests/test_offline_golden.py   # golden reference of the offline library
python -m pytest -m gpu               # GPU tests, only on a machine with CUDA (DHM-PC)
python tests/benchmark_backend.py 1024 2048      # single-step timings of the offline library
python tests/benchmark_realtime.py 1024 2048     # realtime kernel throughput and stage timings
python tools/compare_unwrapping.py    # unwrapping comparison report
python tools/noise_vs_n.py            # phase noise against the number of frames (C14)
black . && ruff check --fix .         # formatting and linting (line length 88)
```

- pytest configuration is in `pyproject.toml`: `testpaths = ["tests"]`; `tests/golden/` is not collected. Markers: `gpu` (CuPy and a CUDA device), `slow`, `hardware` (DHM-PC with camera).
- `tests/golden/golden_offline.npz` is produced by `tests/golden/make_golden_offline.py`. Regenerate it only when a todo says so, and give the reason in `WORKLOG.md`.
- `tests/realtime/synthetic_dhm.py` and `tests/realtime/conftest.py` provide synthetic holograms (carrier fringes, piston phase, noise) for kernel tests.
- Measurement and comparison scripts live in `tools/`, because `scripts/` is gitignored (only `scripts/clean.py` lives there, locally). `data/` (example holograms and results) is gitignored as well; test data must be small and live under `tests/`.
- `.pre-commit-config.yaml` (black, ruff) is local and gitignored.

## Architecture

**Backend** (`backend.py`)
- NumPy (default) or CuPy, precision `"double"` (default, reproduces the CPU results) or `"single"` (float32/complex64; much faster on consumer GPUs, not yet validated on real hardware).
- `set_backend`, `use_backend(...)` and `set_precision` change a module-global state (`_state`) and are **not thread-safe**. `namespace(name)` returns an array namespace without touching the global state; the realtime kernel receives its `xp` explicitly.
- `to_numpy()` converts results at plot and save boundaries.

**Offline library (golden reference, do not change its behaviour)**
- `reconstruction/hologram_class.py`: `Hologram`, `ReferenceHologram`. `locate_order` searches the −1st order on the left half of the spectrum, which fixes the sign convention (field ∝ O·conj(R)).
- `reconstruction/processor_class.py`: `HologramProcessor.run`: spatial filtering of object and background → compensation in the hologram plane (E_obj · conj(E_ref) / |E_ref|²) → propagation of the compensated field → unwrapping → height `h = λ0 φ / (2π (n_structure − n_medium))` (mode "developed": air; mode "print": liquid resin).
- `reconstruction/numericalPropagation.py`: band-limited angular spectrum method, split into `transfer_function` / `apply_transfer_function` (shared with the realtime kernel) plus the optional padding trick (`border_mean`, `pad_centered`, `crop_centered`).
- `reconstruction/phaseUnwrapping.py`: `"Fast 2D"` (scikit-image, CPU, default) and `"Least Squares"` (DCT, GPU-capable, not validated on DHM data), plus older variants.
- `reconstruction/fieldops.py`: pure functions `phase`, `intensity`, `phase_to_height`.
- `reconstruction/plotter_class.py`: `DHMPlotter` (matplotlib). `reconstruction/base_class.py`: `HologramCore`.
- `pipeline/reconstruction_pipeline.py`: `ReconstructionPipeline` (batch workflow; tkinter dialogs only in the interactive branch).
- `single_functions/functions.py`: `reconstruct_hologram`, `reconstruct_single_hologram`.
- `io/loader.py` (`DataLoader`), `io/structure.py` (scidatacontainer containers), `simulation/simulate.py` (synthetic holograms), `utils.py` (`get_logger`, `get_hologram`).
- `config/`: presets as JSON (`DHM_preset/Zeiss_20x.json`, `Zeiss_63x.json`; `Background_preset/SZ2080.json`) loaded by `config/presets.py`. Known inconsistencies, documented but not fixed (K9): duplicate keys (`xPixelSizeUm`/`yPixelSizeUm` vs `pixelPitch`, `wavelengthUm` vs `wavelength`) and `"name": "Zeiss 63X"` against the preset key `"Zeiss 63x"`.

**Lazy imports**
- `offaxisholo/__init__.py`, `reconstruction/__init__.py` and `realtime/__init__.py` load their heavy members on first access (PEP 562). `import offaxisholo.realtime` must load neither tkinter, matplotlib, scikit-image, OpenCV nor scidatacontainer (`tests/test_import_hygiene.py`). `calibrate_order` and `"fast2d"` unwrapping load OpenCV / scikit-image on first use.

**Realtime kernel** (`realtime/`)
- `config.py`: typed dataclasses `DHMConfig` (from a preset), `DemodConfig` (size, order, crop, radius, mask, windows), `CompensationConfig` (`eps_rel`), `AveragingConfig` (`block` / `sliding` / `ema`), `PropagationConfig` (z values, padding, band limit), `RealtimeConfig` with JSON round trip and `config_hash`.
- `calibration.py`: `calibrate_order` → `OrderCalibration` (order position and radius, once, offline; never per frame).
- `core.py`: `OffAxisPipeline` with `set_background(frames)`, `push(frame)`, `result()` → `RealtimeResult`. Chain: upload raw frame → FFT → crop the sideband n×n around the order by index arithmetic → IFFT n×n (pixel pitch dx' = dx·N/n, `config.crop_pitch`) → piston correction against a reference region (float64 reduction, no host sync) → accumulate complex fields. Per result only: compensation `E = mean(U_obj) · K` with `K = conj(Ub) / (|Ub|² + eps_rel·max|Ub|²)` (computed once per background, `eps_rel` start value 1e-3) → propagation with a cached transfer function H(z).
- Parity mode (`crop=None`, `mask="circle"`, `eps_rel="machine"`, `piston=False`, `precision="double"`) reproduces `HologramProcessor` up to round-off.
- `accumulators.py`: `BlockAccumulator`, `SlidingAccumulator` (ring buffer, periodic re-summation), `EMAAccumulator` (N_eff = (2 − α)/α).
- `unwrapping.py`: `unwrap(phase, method)` and `TemporalUnwrapper` (live tracking only).
- `result.py` (K4): `ReconstructionResult` (phase float32, linear intensity, optional complex64 field, `pixel_pitch_um` after the crop, `z_um`, `wavelength_um`, `n_medium`, `FrameRecord` list, quality, `timings_s`, config and hash, versions, `created_utc`, draft `schema_version`) with `from_realtime` (the only device → host copy), `to_message` (JSON header + named little-endian buffers) and `from_message`. The final contract comes from `dhmconnect/schemas/` (G2, decided 2026-11-16; K16).
- `metrics.py` (K2): per-frame metrics computed in `push` from the demodulation FFT (saturated / black counts, intensity quantiles, sideband energy fraction and peak position, piston phase, phase noise in the reference region); device ring buffer `MetricsRecorder`, host copy via `OffAxisPipeline.frame_metrics()`. Switched on by `MetricsConfig.enabled` (off by default, so parity and benchmarks are unchanged).
- `selection.py` (K3): `select_frames(frame_metrics, SelectionPolicy)` → `Selection` (mask, reasons, quality summary); policies `all` (default, O3), `threshold`, `robust` (median / MAD), `best_k`; `min_frames` always enforced. Block mode keeps the fields with `OffAxisPipeline(..., keep_fields=True)` (`FieldStore`, memory `n_obj · n²` complex values) and `apply_selection(mask)` rebuilds the average; sliding / EMA only allow device-side threshold gating (K17, not built).
- `api.py` (K6): the entry points `reconstruct_frames(frames, background, config, *, selection, outputs, frame_info, metrics, z)` and `ReconstructionSession` (`warmup`, `set_background`, `push(frame, info)`, `result`, `reset`) → `ReconstructionResult`. The service (DHMConnect DL-B4) and the GUI import only these plus the configuration classes. Metrics on by default, selection `all` by default (decisions 2026-11-16).
- `background.py` (K7): `BackgroundModel` (Ub, K, quality, frames, creation time, content id, `RealtimeConfig.background_hash()` over dhm, demod, eps_rel, piston, precision); `OffAxisPipeline.background_model()` / `use_background(model)`; save and load in `io/background.py` (`.npz`, no pickle). No expiry; `background_max_age_s` only adds a warning to the result (O6).
- `timing.py` (K5): `StageTimer`, cumulative seconds per stage (upload, demodulation, piston, averaging, compensation, propagation, download) and `total`; `OffAxisPipeline.timer`. Without `profile=True` it never synchronises the device (GPU numbers are then enqueue times).
- Noise model: σ² ≈ σ_obj²/N_obj + σ_bg²/N_bg (background floor).

**Facts from the hardware analysis that matter here**
- Camera: Basler acA1920-155um, ROI 1024 × 1024, 8 bit (1 MiB per frame), exposure about 16.8 ms. Zeiss 63x preset: pixel pitch 0.0869 µm in the object plane, `dcRadius` 304, `propagationDistance` 1.1e-7 m, while a focus of about −2.6 µm was measured (open decision G4 / O1).
- The pixel size reported by Koala configuration 180 belongs to a 20x objective (finding B17). The pixel size always comes from the parameter set, never from the device.
- The DHM-PC driver supports CUDA 13 (maintainer, 2026-11-16). Python is not installed there yet (the PC was only reached through the C server so far); target: **Python 3.14** with `uv` (G9, 2026-11-18). DxDiag 2026-11-18: NVIDIA RTX A2000 12 GB, Windows 11 Pro, i9-14900K, 64 GB; repeated GPU TDR events (LiveKernelEvent 141) → keep kernels short (master plan v0.4 §10).

## System context

This repository is one of four that together build a 2PP system with real-time feedback from the DHM (details: `docs/system/SOFTWARE_PLAN_OVERVIEW.md`).

| Package | Repository | Owns |
|---|---|---|
| `offaxisholo` (this one) | OffAxisHolo | Everything about one hologram stack: frame metrics and selection, demodulation, averaging, compensation, propagation, phase and intensity, quality numbers, timings. Analysis and live GUI. |
| `dhmconnect` | DHMConnect | Protocol v1, client, the DHM service on the DHM-PC (state machine, jobs, camera and hardware gateway, calibration procedures, result store, raw archive). It calls this package. |
| `nanofactorysystem` | NanoFactorySystem_automated | Orchestration of the print, `experiment.h5` (single writer), measurement recipes, operating modes. |
| `mira` | MIRA | Registration by stage position, speckle fusion, stitching, comparison with the target structure. |

Rules that follow from this:
- No network server, protocol, camera or hardware driver, service state machine or raw archive in this repository (moved to DHMConnect: groups D, E, F, G of `todo_realtime.txt`). No registration, fusion, stitching or target comparison (MIRA).
- The core of `offaxisholo` imports none of the other three packages. Only the optional GUI (`offaxisholo.gui`, extras `[gui]` / `[live]`) may import `dhmconnect.client`.
- Shared contracts (`ReconstructionResult`, `FrameRecord`, `ProductSet`, …) are defined in DHMConnect as JSON Schema plus example messages (`dhmconnect/schemas/`, decision G2, decided 2026-11-16); this repository vendors the example messages under `tests/contracts/`. Conventions: units in names (`_um`, `_rad`, `_s`), float32 / complex64, axis 0 = y, axis 1 = x, `schema_version` in every message, never `pickle` across a process or network boundary.
- Todo IDs of the other repositories may be referenced: `T…`/`F…` (NanoFactorySystem_automated), `DL-…` (DHMConnect), `MI-…` (MIRA).

## Repository rules

1. The offline library stays unchanged in behaviour and is the golden reference (`tests/test_offline_golden.py`). New code goes into new modules. An existing file is changed only when a todo says so, and its default behaviour stays the same.
2. `import offaxisholo.realtime` stays lightweight (`tests/test_import_hygiene.py`). The GUI is an optional extra and is never imported by the core.
3. Kernels are backend-agnostic (NumPy / CuPy, `xp` passed explicitly). Exactly one thread owns the CUDA context.
4. Hot path (per frame): no host synchronisation (`.item()`, `.get()`, `bool()` of a device array, boolean indexing, `xp.nonzero`, Python decisions on device values), no allocation after warm-up, no logging above debug level and no f-strings in the loop.
5. Average complex fields, never phases. Compensation before propagation, unwrapping last.
6. No invented physics: sign convention (`phase_sign`), propagation distance with its `propagation_index`, `n_surround` and `n_structure` are explicit parameters of the parameter set or the request (G4, O11, decided 2026-11-18), never hard-wired assumptions and never silent defaults.
7. Scope after G5-new (2026-11-18): this package reconstructs **one hologram stack** (frames of one position) and stays wrapped; unwrapping across tiles, registration, fusion, stitching and levelling belong to MIRA. The GUI is a pure observer that takes only images as input (O8, O9); no operating modes such as "print" (O11).
8. The pixel size never comes from Koala (B17). It is part of the parameter set.
9. Test before optimising. Record measured numbers in `docs/realtime/benchmarks.md` with date, machine, GPU, backend and precision. No performance number without a measurement.
10. Strict layer separation (maintainer note): DATA (configuration, results, records), PROCESSING (pure computation, no file IO, plotting, logging or network), IO (loading and saving), PIPELINE (orchestration). Plotting stays in the plotting modules.
11. Nothing is deleted. Legacy material lives in `attic/` (B5); anything else is moved there only after the maintainer approves.

## Maintainer Instructions (Hannes)

### Language
Everything is written in English: variable, function, class and module names, docstrings,
comments, README and other documentation, design documents, todo entries, commit messages,
pull request texts and log entries. If you encounter German identifiers or comments within
the scope of your current task, translate them. Outside that scope, add a todo instead of
renaming them unasked. German material handed over by the maintainer (for example
`docs/realtime/anweisung_realtime.txt`) is translated when it is turned into documentation or
todos; do not rewrite it in place unasked.

### Session start
1. Read `## Current Status` and `## Status Update Anchor` at the end of this file. As long as
   they hold no status yet, go by the phase plan in `TODO.md`.
2. Read `TODO.md` and pick a task (working rule 1 at the top of `TODO.md`).
3. Read the most recent entries in `WORKLOG.md`.
4. For work on the realtime kernel, read the relevant sections of
   `docs/realtime/anweisung_realtime.txt`; for anything that crosses repository borders, read
   `docs/system/SOFTWARE_PLAN_OVERVIEW.md`.

### Work log (mandatory)
After every completed task, and before ending a session even if the task is unfinished,
append an entry to `WORKLOG.md`.

- Get the timestamp from the system: `TZ=Europe/Berlin date "+%Y-%m-%d %H:%M %Z"`. Never guess it.
  (In Git Bash on Windows without tzdata, use PowerShell `Get-Date -Format "yyyy-MM-dd HH:mm"`.)
- The log is append-only. Put the newest entry at the bottom. Never edit past entries.
- Write one entry per task. List **every** file that was created, modified, deleted or renamed.

Format:
```
### <YYYY-MM-DD HH:MM TZ> — [<Todo ID>] <Short task title>
- **Status:** done | partial | blocked
- **Changes:**
  - `path/to/file.py`: what changed and why
  - `path/to/new_file.py` (new): purpose
  - `path/to/old_file.py` (deleted | renamed → `new/path.py`)
- **Tests:** what was run and the result, or "none" with the reason
- **Commits:** `<short SHA>` <commit subject line>, … (or "uncommitted — proposed message:" followed by the full message)
- **Follow-ups:** open points or new todos created
```

### Definition of Done
A task counts as done only if all of the following hold:
- All acceptance criteria stated in the todo are met.
- The code runs, and existing tests pass, in particular `tests/test_offline_golden.py` and
  `tests/test_import_hygiene.py`. New logic has tests where that is feasible.
- Public functions and classes have English docstrings (NumPy style).
- No new `TODO`/`FIXME` related to the task remains in the code.
- The change is committed and logged in `WORKLOG.md`.

If a task does not meet this, mark it `partial` in the log and describe the remaining gap.

### Todos
- Follow the format rules at the top of `TODO.md`.
- When you start a todo, move it to "In Progress". If it cannot proceed, move it to "Blocked"
  and state the reason.
- When a todo meets the Definition of Done, move it to "Done" with the date, 1–2 sentences
  and the commit SHAs. Never move entries to `ARCHIVE.md`; the status review routine does that.
- If a todo is only partially done, keep it in "In Progress" and add a follow-up todo that
  describes the remaining gap and the decision or measurement it needs.
- A todo that needs an open decision (table "Open decisions" in `TODO.md`) is never guessed:
  implement the independent part and ask.
- If you discover new problems, add them to "Open" and tag them with `(found during <ID>)`.
- A todo that belongs to another repository (network, camera, registration, …) is added to that
  repository's `TODO.md` or reported to the maintainer, not implemented here.

### Commits (mandatory)
- Work on a feature branch created from `Holo_HR` (`feat/…`, `fix/…`, `docs/…`); the maintainer
  merges into `Holo_HR`. Never commit to `main` (upstream).
- Commit after every completed task and after every coherent sub-step of larger tasks. A task
  that ends without a commit is not done.
- Split commits by topic: one logical change per commit. Do not mix refactoring, formatting,
  feature work, tests of unrelated code and documentation of other tasks. Todo bookkeeping
  (`TODO.md`, `WORKLOG.md`) goes into its own `docs(todo): …` commit at the end of a task.
- Write every message for a human who reads `git log` months later without the session: the
  subject says what the change does (not "update files", "fixes", "WIP"), the body says why it
  was needed and what a reader must know (decisions, side effects, limits, measured numbers).
  Example of a good subject: `feat(realtime): add frame metrics from the demodulation FFT [K2]`.
- Always write a complete commit message in this format (Conventional Commits):
```
  <type>(<scope>): <imperative summary, max. 72 chars> [<Todo ID>]

  <Body: what changed and why, wrapped at 72 chars. Mention design
  decisions and side effects. No restatement of the diff.>

  Files:
  - path/to/file.py: short description of the change
  - path/to/new_file.py (new): purpose

  Refs: <Todo ID>
```
  - `type` is one of: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `build`, `chore`.
  - `scope` is the affected module or package, e.g. `realtime`, `propagation`, `unwrapping`,
    `config`, `gui`, `io`, `backend`.
  - Add `BREAKING CHANGE: <description>` as a footer if a public API changes.
  - Earlier commits on `Holo_HR` use plain imperative subjects ("Add …"); new commits use the
    format above.
- If you cannot or should not commit, do not commit. This applies to failing tests, unrelated
  uncommitted changes in the working tree, and cases where the maintainer said not to commit.
  Still generate the full commit message: print it at the end of your response and record it
  in WORKLOG.md.
- When working on a feature branch, also generate a pull request title and description:
  Summary, Changes (with files), How it was tested, Linked todos, Open points.
- Do not push, amend, rebase or force-push unless explicitly instructed.
- Do not edit `## Status Update Anchor` or `## Current Status`. These sections are maintained
  exclusively by the status review routine.

### Testing conventions
- `python -m pytest` must pass on a CPU-only machine, without CuPy, without Qt and without
  `dhmconnect`, `mira` or `nanofactorysystem`. GPU tests carry the marker `gpu` and are skipped
  without CUDA; tests that need the DHM-PC camera carry `hardware`.
- Every new kernel is tested against the offline library (CPU, double) or an analytic case
  before it is optimised; CuPy parity tests exist for every kernel and are skipped without CUDA.
- Tolerances are stated in the test (CPU double ≈ 1e-9 rad, GPU single ≤ 1e-3 rad unless a
  todo says otherwise).
- Never make a test pass by weakening or deleting assertions without justification.
  If a test reveals a real bug, fix it in a separate `fix:` commit if the fix is small and clear.
  Otherwise, mark the test `xfail(reason=..., strict=True)` and add a todo.
- Tests write only to `tmp_path`.
- Never modify the legacy scripts in `attic/legacy_tests/`; they are not collected.

## Current Status
<!-- Maintained exclusively by the status review routine. Claude Code reads it, never edits it. -->
No status written yet (file created 2026-11-15).

## Status Update Anchor
<!-- Maintained exclusively by the status review routine. Claude Code reads it, never edits it. -->
No anchor set yet.
