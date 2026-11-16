# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`nanofactorysystem` is a Python 3 (3.12) package that drives the Femtika Laser Nanofactory, a two-photon-polymerization (2PP) laser direct-writing system, as a 3D printer. The hardware is an Aerotech A3200 motion controller (stages X/Y/Z and galvo scanner A/B) with a laser attenuator, a MatrixVision camera and a LyncéeTec DHM (digital holographic microscope). Most code only produces real results on the lab PC connected to that hardware.

Comments, docs and TODOs are often in German. `TODO.md` is the task list with the sections Open / In Progress / Blocked / Done; its format rules are in the comment at the top of the file. `ARCHIVE.md` holds archived completed tasks.


## Commands

```bash
python -m pip install ".[test]"   # install package with all subpackages and test dependencies (pyproject.toml)
python clean.py                # remove build/, dist/, *.egg-info, __pycache__
python -m pytest               # all tests without hardware (unit + dummy backend); hardware tests are skipped
python -m pytest test/devices/test_aerotech.py::test_zmax_safety   # a single test
python -m pytest -m "not slow"                  # skip the ~10 s experiment dry run
python -m pytest -m hardware --run-hardware     # hardware tests, only on the lab PC
python -m pytest test/test_aerobasic/test_golden_programs.py --update-golden   # re-record golden programs
```

- `test/README.md` describes every test module, the categories (unit, dummy integration, hardware, manual), the fixtures and how to use the dummy backend. Keep it up to date when adding, moving or removing tests.
- pytest configuration is in `pyproject.toml` (`testpaths = ["test"]`; `test/manual/`, `_programs` and legacy directories are excluded). Markers: `hardware` (skipped without `--run-hardware`), `slow`. Shared fixtures are in `test/conftest.py`: `test_config`, `dummy_backend`, `dummy_controller`, `dummy_system`, `no_sleep`, `tmp_program_dir`, `golden`, `lab_user`.
- `python -m pytest` must pass without hardware, without `~/nanofactory.json` and without `mvIMPACT`/`OffAxisHolo`. The environment needs NumPy-2-compatible builds (`opencv-python>=4.10.0.84`, pinned in `pyproject.toml`); an older OpenCV fails with `numpy.core.multiarray failed to import`.
- Generated program references for golden tests are `test/golden/*.txt`; other test output goes to `tmp_path`, or `test/_programs/` (gitignored) for manual inspection.
- Experiment entry scripts in `mains/` import siblings as `from Experiments.… import …`, so run them with `mains/` as the working directory (e.g. `cd mains && python main.py`).
- Optional hardware and local dependencies (`mvIMPACT`, `OffAxisHolo`, `PlotFont`) are not installable from PyPI; `mvIMPACT` is imported on first use of the real camera. The slicer also uses `trimesh`, `shapely` and `h5py`.

## Configuration

- `nanofactorysystem/config.py` loads `sysConfig`, which has `user:<name>` and `objective:<name>` sections (e.g. `"Zeiss 20x"`, `"Zeiss 63x"`). Lookup order: `use_config(path_or_dict)` (context manager, swaps the content in place), `$NANOFACTORY_CONFIG`, `~/nanofactory.json`, then the built-in `DEFAULT_CONFIG` (devices and both objectives, no users, no calibration file) with a warning.
- Class defaults of `System`, `A3200`, `Attenuator`, `Camera`, `Dhm` are `ConfigDefaults(section, {...})` descriptors, resolved on access (deep copy), so importing the package never reads config sections.
- The YAML files in `config/` (Hydra layout) describe the same data but are not currently read by any Python code. `config/voxel_seed.csv` is the seed of the voxel database.
- Data locations: `user:<name>.dataRoot` (experiment data root, default `~/Documents/Femtika_Experiment/<user>`), `system.voxelDatabase` (default `~/Documents/Femtika_Experiment/voxels.sqlite`), `system.syncFolderNames` (folder names refused as data location, default Seafile/OneDrive/Dropbox); `user:<name>.initials` for substrate labels.
- Runtime parameters are passed as nested dicts (`sys_args` with sections `attenuator`, `controller`, `sample`, `focus`, `layer`, `plane`, `dhm`, `camera`). Each component takes its own section via `popargs`, and `Parameter` subclasses merge these into their `_defaults`. `Parameter` works on a copy of each section, so a `sys_args` dict can be reused.

## Architecture

**Hardware layer**
- `system.py` `System` aggregates `Camera`, `Dhm` (skipped when `sys_args["dhm"]["usage"]` is False) and the `A3200` controller + attenuator (`devices/`). `System.controller` and `System.a3200_new` are the same object.
- One controller class since T20: `devices/a3200.py` `A3200(Parameter, AerotechController)`.
  - All commands go through one `aerobasic/ascii.py` `AerotechAsciiInterface` (`.api`, TCP 127.0.0.1:8000), which records the command history (`save_log()` → `A3200.log`).
  - Config-based µm helpers used by `System` and `tools/`: `moveabs`, `moveinc`, `position`, `wait`, `power`, `pulse`, `laseron/off`, `zline`, `home(axes)`, `container`; `run()` sends a raw command.
  - Program tasks from `devices/aerotech.AerotechController` used by `Experiment`: `run_program_as_task` (loads `.pgm` files as controller tasks), `xyz`, `axis_status`, `save_log`.
  - Safety: the helpers check `zMax` (and close the connection when exceeded); in addition, `api.z_limit` refuses immediate absolute Z moves beyond `zMax` sent through `.api` (maintainer decision). Programs run as tasks are not checked.
  - `devices/aerotech.Aerotech3200` is the same task handling without user/config (e.g. for scripts and `test_femtika`), with an optional `z_max` and its own `home()` sequence.
  - `AerotechError` is a `RuntimeError`; `TaskFailedError` (a failed or stalled task) is also a `ValueError`.
- `tools/` holds measurement algorithms built on the camera and DHM: `Focus`, `Layer` (resin interface detection), `Plane`, `Grid`, `Stitch`, and `Transform` (camera↔stage pixel/µm transforms).

**Hardware backends** (`backends/`, design in `docs/design/DUMMY_BACKEND.md`)
- `System(..., backend=...)` and `Experiment(..., backend=...)` select the backend explicitly: `None`/`"real"` (default, lab hardware) or `"dummy"`/a `DummyBackend` object. There is no environment variable. The backend only replaces the seams that touch hardware; all facades stay the real classes. The device classes take matching keyword-only injection parameters: `A3200(transport=, program_dir=)`, `AerotechAsciiInterface(transport_factory=)`, `Aerotech3200(transport_factory=, program_dir=)`, `Camera(driver=)`, `Dhm(driver=)`.
- `backends/protocols.py` defines the role protocols (what `System`/`tools`/`Experiment` use) and seam protocols (what a backend provides). Extend them when a consumer starts using a new device method.
- `backends/dummy/`: one deterministic `SimulatedWorld` (stage, laser, exposures, tasks, virtual clock, seeded RNG, `CallLog`) shared by `FakeA3200Transport` (socket-level ASCII controller with a small AeroBasic interpreter for program tasks), `DummyCameraDriver`, `DummyDhmClient` and a synthetic attenuator calibration file. Plane detection is not simulated: dry runs pass a known plane via `Experiment.plane_fit(plane=...)`. New AeroBasic commands used by generated programs must be added to `FakeA3200Transport.execute()` (the dry-run test checks that every statement is simulated).

**AeroBasic program generation** (`aerobasic/`)
- `AeroBasicAPI` (in `aerobasic/__init__.py`) exposes AeroBasic commands as Python methods. `AeroBasicProgram` (in `aerobasic/programs/__init__.py`) collects these commands as text lines and writes a `.pgm` file.
- `programs/setups.py` provides `DefaultSetup` (stages plus galvo, IFOV off) and `SetupIFOV` (Aerotech IFOV mode, objective-specific). The chosen setup changes which axes (`"XYZ"` vs `"ABZ"`) and accelerations are valid.
- `programs/drawings/` defines `DrawableObject`, the abstract base of every printable structure. A structure implements `iterate_layers(coordinate_system)`, which yields one `DrawableAeroBasicProgram` per layer. Each layer's coordinates pass through `devices/coordinate_system.CoordinateSystem`, which maps local µm coordinates to stage coordinates using an offset and a `ZFunction` (`StaticOffset`/`Plane`/`PlaneFit` from plane fitting). `DropDirection` (UP for 20x, DOWN for 63x) never changes z coordinates; it reverses the layer print order and selects the plane interface, the resin layer in the scanner and the QR pixel-line direction (all places are listed in its docstring). The tools get it as `sys_args["layer"]["dropDirection"]`.
- `DrawableObject.to_json()` serializes the constructor arguments (with `__module__`; enums, points and calibrations are encoded reversibly), and `drawings.base.structure_from_json()` rebuilds the structure. A parameter stored under another attribute name is declared in `_json_attributes`; a parameter that cannot be stored is reported (warning, `__missing__`).
- Structures that set the laser power in mW (`IFOV_Lines` and the IFOV structures built on it) convert it with a `devices/power_calibration.PowerCalibration`: an explicit `calibration=` argument, else the one activated with `power_calibration(...)` (`Experiment.build_programs()` activates the running system's attenuator calibration), else `attenuator.calibrationFile` from the config.
- Structure families:
  - `lines.py` (`Rectangle3D`, `Stair`, `Corner`, …)
  - `height_function_structures/` (gratings, lenses, DOEs built from `HeightFunctions` + slicer)
  - `hatch_generator.py`, `laser_segments.py`, `apertures.py` (reusable components)
  - `tile_manager.py` (stitching of structures larger than the FOV)
- The directories `old_to-delete/` and `new/` and the files `*_old.py` are legacy or experimental code, not the active path.
- `aerobasic/slicer/` is a separate mesh/heightmap → toolpath pipeline (`pipeline.slice_geometry`). It is a self-contained port of `tpp_slicer`, and its layout is described in `tree_overview_slicer.txt`.

**Voxel database** (`voxel/`, design in `docs/design/VOXEL_DATABASE.md`)
- `VoxelDatabase` (SQLite, schema version in `PRAGMA user_version`, migrations in `voxel/migrations.py`) stores measured voxel widths/heights per material, objective, setup, power and velocity; `voxel_size()` returns an exact hit, a 2-D linear interpolation in (ln P, ln v) inside the convex hull, or a 1-D dose fallback for collinear data, else None (no extrapolation). CSV import/export; `*.sqlite` is not versioned.
- Voxel-aware slicing (T54): `voxel/model.py` has the voxel models `NoVoxelData`, `FixedVoxelModel` and `DatabaseVoxelModel` (material, objective, setup); the slicer only knows the protocol `aerobasic/slicer/voxel.VoxelModel`. `slice_geometry(..., voxel_model=)` / `Model3D_Slicer(voxel_model=, power=)` / `ExperimentSpec(voxel_material=)` compensate with the voxel size of the job's power and velocity (`slicer/voxel.compensate`): contour offset by half the width, first/last slice half a height inside the part (`SlicingParameters.voxel_height_um`), and spacing per `SlicingParameters.spacing_mode` — `"voxel_overlap"` (default, maintainer decision: spacing = voxel size × (1 − `voxel_overlap`)) or `"static_hatching"` (given spacing, gaps only warned). Strategies get a `voxel=` `VoxelContext`. The values used are in `job.meta["voxel"]`, `Model3D_Slicer.to_json()["voxel"]` and the summary. Without a model or data nothing changes (golden files).

**Experiment orchestration** (`experiment.py`)
- `Experiment` is a context manager that owns a `System`. It lays out a grid of structures inside the resin-drop bounds, adding corner markers plus a QR code (the experiment UUID) unless `skip_corner=True`.
- Typical flow:
  1. `plot_experiment()`
  2. `plane_fit()`, which measures the substrate surface at sample points (or uses a given `plane=`); the points follow `plane_fit_mode` (`PlaneFitMode.GRID`/`CORNERS`/`BORDER`, old integers accepted), and `plane_fitting.py` (`sample_points`, `measure_plane`) also runs a single plane fit without `Experiment`. A warning is logged if the substrate height under a structure varies by more than `tilt_warning_um`
  3. `opl_scan()`, only when the DHM is used
  4. `add_structure(StructureType, name, axes, power, structure=DrawableObject)`; `layer_power=` (list or function of the layer id) makes every layer program set its own power (T31, stored as `layer_powers_mw`). `Model3D_Slicer` prints toolpath elements with a `power_mw` override (or from `power_map(z_um, role)`) in their own IFOV blocks
  5. `build_programs()`, which writes per-layer `.pgm` files and `structures.json`; it estimates and logs the printing time (`time_estimate.py`: path length / speed of every layer program plus `layer_overhead_s`, default 5 s; stored in `/time_estimate`), and the summary shows it next to the actual time (T28)
  6. `print_experiment()`, which runs each layer as an A3200 task and tracks progress; `measure()` takes DHM holograms (with DHM) and camera images only with `Experiment(camera_capture=True)` (default off); with `overview_capture=True` (default off) it also takes a stitched camera mosaic of the experiment area before and after printing (`overview.py`, `Experiment.capture_overview()`; with corners: rectangle incl. corners and QR code, with `skip_corner`: the structure grid; stored in `/overview/<phase>` and `overview_<phase>.png`, single images only with `overview_single_images`; T28)
- `restart_experiment()` and `mains/restart_experiment.py` resume from the progress in `experiment.h5`: every layer printed in an earlier session is skipped, so any number of aborts works. `parameters_from_dictionary()` reads the file (or the JSON of an old folder) and sets `resume=True`; an old JSON-only folder is imported into a new experiment file (`storage/legacy.py`), keeping its UUID.
- `StructureType.REPEAT` prints the most recent grid structure again in the next grid cell as `<name>_rep<n>` (`repeat_of` in the file); a duplicate name gets `_(<i>)`.
- Experiment outputs go into the `path` passed in, which is usually a lab-PC path. Without `resume`, a folder that already holds an experiment is refused.

**Experiment storage** (`storage/`, design in `docs/design/EXPERIMENT_STORAGE.md`)
- `ExperimentStore` writes everything of an experiment into `<path>/experiment.h5` (schema version, metadata, calibration, plane fit, OPL scan, layout, structures with layer programs and progress, camera/DHM captures with commanded and actual positions, DHM products, and per session the controller command log and the console log under `/logs`). Every write opens and closes the file; a lock file `experiment.lock` allows one writing process, and each run is a session.
- `experiment_dictionary.json`, `structures.json` and `print_progress.json` (schema `nanofactory.print_progress/2`, documented in `json_copies.print_progress`) are copies exported from the file (`storage/json_copies.py`); all stored paths are relative to the experiment folder, so a moved folder can be restarted; captures are no longer written as `.zdc` files (`export_capture()` writes one on request).
- Substrates (`storage/substrate_store.py`): `Experiment(path=None, substrate=<label>)` creates `<data root>/<substrate>/<label>-A_<YYYYMMDD-HHMM>/`; the data root is `dataRoot` of the user in the config or `~/Documents/Femtika_Experiment/<user>` and must not lie in a Seafile/OneDrive/Dropbox folder. `substrate.json` holds the substrate record and the index of its experiments; `find_experiments()` searches it. An explicit `path` still works; `substrate_information` (free dict) is stored in the experiment file.
- The summary (`storage/summary.py`) lists every user structure with slice, hatch, power, velocity, IFOV, DHM, camera and status; it is written after `build_programs()` and after every structure into `/summary` and `experiment_summary.json`, and logged as table.
- `storage/records.py` holds the plain data records (`ExperimentRecord`, `StructureRecord`, `CaptureRecord`, …); `ExperimentStore.read()` rebuilds all metadata from the file alone.

**Analysis tools** (`analysis/`, not part of the installed package; README there)
- `analysis/experiment_analysis/` (on the pytest `pythonpath`): `unpack_experiment(h5, target)` mirrors the experiment file into common formats (PNG for images and hologram series, CSV for arrays and tables, TXT for programs and logs, JSON for attributes, stored `.zdc`/`.png` blobs; plus summary, JSON copies and `tables/*.csv`; a snapshot copy while a process holds the lock); `ExperimentFile` reads at any time without HDF5 file locking (retries) and stores results under `/analysis/...` while holding `experiment.lock` for one write (refused while an experiment process holds it; design §5.9).
- `analysis/unpack_experiment.py` and `analysis/analysis_template.py` are the scripts to copy/run (settings at the top); `python -m experiment_analysis unpack|info|tree` from `analysis/`.

**Experiment scripts** (`mains/`)
- `mains/Experiments/**` holds one script per experiment in use. Each describes its experiment in `experiment_spec(objective, absolute_center, ...)` (objective values: FOV, zMax, drop direction, margins, corners) and keeps its former entry function (`print_file`/`testprint`, with `backend=` and `plane=` for dry runs), which runs it with `run_experiment()`. `test/integration/test_ported_scripts.py` checks all of them.
- `mains/Experiments/focal_points/z_line_focal_points.py` exposes z-lines at random points with camera images before and after (one `FocusDetect` container each, training data for a classifier; T36). It only calls `tools.focus.Focus.run`.
- `mains/Experiments/historical/` holds the scripts of earlier experiments that are not ported (not tested; see its README).
- Several experiments on one substrate: `mains/substrate_main.py` is the template (copy per substrate). It defines a `SubstrateSpec` (label, user, objective, resin edges, material) and a list of `SubstrateExperiment`s (an `ExperimentSpec` plus an optional folder); `nanofactorysystem/substrate_plan.run_substrate()` checks all experiment areas (grid, margin, corner/QR clearance) for overlap with each other, with experiments already in the substrate index and with the resin drop before anything is printed, then runs them one after another into the substrate index. The drop is the ellipse through the four edge points (`resin_drop.py`, T62; also the center check of `Experiment(resin_edges=)`); dip-in has no drop boundary and is not checked.
- `mains/lab_checks/new_features_check.py` is a substrate main that exercises the new features on the lab PC (two small experiments: overview, camera captures, power per layer, repetition, slicer block; files unpacked with `analysis/`) and prints a checklist (T68).
- New experiments are described by an `ExperimentSpec` (`nanofactorysystem/experiment_spec.py`: objective defaults for FOV, drop direction, zMax, margins and corners; structures as `StructureSpec`; `ProgramSource.DRAWING` or `.SLICER`) and run with `run_experiment()`; `mains/Experiments/experiment_template.py` is the template (dry run: `main(backend="dummy", plane=...)`).
- `mains/Experiments/default_exp_file.py` is the former template; its `binary_testprint(..., backend=, plane=)` can be dry-run on the dummy backend (see `test/integration/`).
- `mains/main.py` selects one of these functions by import and supplies the resin edge coordinates and center for the current substrate.

## Maintainer Instructions (Hannes)

### Language
Everything is written in English: variable, function and class names, docstrings, comments,
commit messages, documentation, and log entries. If you encounter German identifiers or
comments within the scope of your current task, translate them. Outside that scope,
add a todo instead of renaming them unasked.

### Session start
1. Read `## Current Status` and `## Status Update Anchor` in this file.
2. Read `TODO.md` and pick a task. Prefer the one named as the next task in `## Current Status`.
3. Read the most recent entries in `WORKLOG.md`.

### Work log (mandatory)
After every completed task, and before ending a session even if the task is unfinished,
append an entry to `WORKLOG.md`.

- Get the timestamp from the system: `TZ=Europe/Berlin date "+%Y-%m-%d %H:%M %Z"`. Never guess it.
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
- The code runs, and existing tests pass. New logic has tests where that is feasible.
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
  describes the remaining gap.
- If you discover new problems, add them to "Open" and tag them with `(found during T<n>)`.

### Commits (mandatory)
- Commit after every completed task and after every coherent sub-step of larger tasks.
- One logical change per commit. Do not mix refactoring, formatting and feature work.
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
  - `scope` is the affected module or package, e.g. `slicer`, `tiling`, `aerobasic`.
  - Add `BREAKING CHANGE: <description>` as a footer if a public API changes.
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
- `python -m pytest` must pass without hardware, without `~/nanofactory.json`, and without
  `mvImpact` or `OffAxisHolo`. Tests that need the lab are marked `hardware` and only run
  with `--run-hardware`.
- Never make a test pass by weakening or deleting assertions without justification.
  If a test reveals a real bug, fix it in a separate `fix:` commit if the fix is small and clear.
  Otherwise, mark the test `xfail(reason=..., strict=True)` and add a todo.
- Every moved, converted or removed test is justified in WORKLOG.md and reflected in
  `test/README.md`.
- Tests write only to `tmp_path` or `test/_programs/`.
- Keep existing `unittest.TestCase` classes. New tests may use plain pytest style with fixtures.
- Never modify legacy code (`old_to-delete/`, `new/`, `*_old.py`). Exclude it from collection.
- Real hardware is always the default backend. The dummy backend is opt-in.