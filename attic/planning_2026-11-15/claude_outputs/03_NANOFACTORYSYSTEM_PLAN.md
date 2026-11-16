# NanoFactorySystem_automated — instructions and todos

Status: draft v0.1, 2026-11-15, waiting for maintainer approval. Read `00_SOFTWARE_PLAN_OVERVIEW.md` first.

This file is a proposal in the formats of `TODO.md` and `future_todo.md`. `CLAUDE.md` of the
repository stays in force without change (English, work log, commits, Definition of Done, real
hardware as default backend, golden files, dummy-backend test for every behaviour change).
Nothing here is scheduled until the maintainer approves it and the entries are copied into `TODO.md`
(phase plan and "Open") and `future_todo.md`. IDs continue after T69 and F14; renumber if they are
taken by then.

## 1. Role

nanofactorysystem orchestrates the print, owns the information model and is the only writer of
`experiment.h5`. Five work areas:

1. Information model: material → substrate → experiment (project) → structure.
2. Variable communication with the DHM-PC through measurement recipes and the `dhmlink` client.
3. Operating modes, integrated now, per experiment with an override per structure.
4. Camera usable during the whole process (live feed, recording).
5. A core that a GUI can drive. The GUI itself is F15; its details follow separately.

Out of scope: hologram reconstruction (`offaxisholo`), registration, stitching, fusion and target
comparison (`mira`), the DHM service (`dhmlink`).

## 2. Additional working rules for these phases

1. Phases 5–10 below extend the phase plan. Design todos (T70–T74) end with a document in
   `docs/design/` and move to "Blocked" until the document carries `Status: approved <date>`.
2. `dhmlink` and `mira` are optional dependencies. `python -m pytest` keeps passing without them.
3. The legacy DHM path (`Dhm`, `DhmClient`, protocol v6) and today's hologram counts stay the default
   until the maintainer switches. Golden command logs must not change in the legacy mode.
4. A line starting with "Decision:" is a question to the maintainer (working rule 3 of `TODO.md`).

## 3. Proposed extension of the phase plan

| Phase | Topic | Todos | Gate before the dependent todos start |
|---|---|---|---|
| 5 | Concept for phases 6–10 | T70, T71, T72, T73, T74 | each design approved by the maintainer |
| 6 | Information model | T75, T76, T77, T78, T79, T80, T81 | T70 approved |
| 7 | DHM service integration | T82, T83, T84, T85, T86, T87 | T71 approved; T84 needs `dhmlink` DL-C1, DL-C2 |
| 8 | Operating modes | T88, T89, T90, T91, T92 | T72 approved; T90 needs `mira` MI-53 for the real evaluator |
| 9 | Camera live feed | T93, T94, T95 | T73 approved |
| 10 | Core ready for a GUI | T96, T97 | T74 approved |

Phases 6, 9 and 10 are independent of each other. Phase 7 can start with T82 and T83 before `dhmlink`
exists. Phase 8 needs T83 and T85.

## 4. Todos for `TODO.md` ("Open")

### Phase 5 — Concept

- [ ] T70: Design the information model v2 (material, substrate, experiment, structure) [phase: 5]
  - Priority: high
  - Depends on: –
  - Done when:
    - `docs/design/INFORMATION_MODEL.md` defines the four levels, their fields, where each is stored
      (config, SQLite, `substrate.json`, `experiment.h5`) and how old data is read;
    - it defines which settings live on the structure (setup IFOV on / off, program source, slicing,
      hatching, power, velocity, voxel model, DHM recipe, operating mode) with experiment defaults;
    - it defines the serialised form of an experiment (project file) and of a structure;
    - Decision: material store. Proposal: extend the voxel SQLite database (`material` plus a
      `material_property` table with key, value, unit, wavelength, state, source, date) so that
      details can be added at any time.
    - Decision: can structures with IFOV on and IFOV off be mixed in one experiment? Today `setup`
      is a field of the experiment. Investigate the setup programs first and report.

- [ ] T71: Design the DHM service integration and measurement recipes [phase: 5]
  - Priority: high
  - Depends on: –
  - Done when:
    - `docs/design/DHM_SERVICE_INTEGRATION.md` defines the role protocol `DhmMeasurement`, the
      adapters (legacy, service, fake), the recipe (section 5 below), the capture sequence with
      `reserve_capture()`, asynchronous jobs, acknowledgement and `sync` after a restart;
    - `docs/design/EXPERIMENT_STORAGE.md` gets a proposed amendment for `raw_ref`, `job_id`, frame
      metrics, `/dhm_groups` and the schema version;
    - Decision: behaviour when the DHM-PC is unreachable (`abort`, `continue`, `legacy`).
    - Decision: may the next layer print while the DHM-PC still computes?

- [ ] T72: Design the operating modes and feedback hooks [phase: 5]
  - Priority: high
  - Depends on: –
  - Done when:
    - `docs/design/OPERATING_MODES.md` defines the modes `off`, `monitor`, `shadow`, `correct`, the
      evaluation hook after a layer capture, the protocols `Evaluator` and `Controller`, where
      reports and plans are stored, and the hard limits of a correction;
    - it defines `LayerTarget` for sliced and for drawn structures;
    - Decision: for drawn structures (`DrawableObject`), is the target rasterised from the toolpath
      with the voxel model, or are they limited to `monitor` without a target?

- [ ] T73: Design the camera live feed and recording [phase: 5]
  - Priority: medium
  - Depends on: –
  - Done when:
    - `docs/design/CAMERA_LIVE.md` defines the single owner of the camera, how `measure()`, the
      overview and the focus tools get frames while the feed runs, exposure conflicts, stop on abort;
    - Decision: frame rate, codec, resolution, storage location and size limit of recordings.

- [ ] T74: Design the headless core for a GUI [phase: 5]
  - Priority: medium
  - Depends on: –
  - Done when:
    - `docs/design/GUI_READINESS.md` lists every blocking call in the run path (message boxes,
      `plt.show()`), defines progress events, pause, resume and abort between layers, and the
      threading rule towards the single-writer experiment file;
    - Decision: GUI framework (overview decision G3).

### Phase 6 — Information model

- [ ] T75: Material registry for photopolymers [phase: 6]
  - Priority: high
  - Depends on: T70
  - Done when:
    - a `Material` record with name, supplier, batch and extensible properties (at least refractive
      index uncured and cured per wavelength) is stored and queried as designed in T70;
    - new property keys can be added without a code change, with unit and source;
    - `SubstrateSpec.material` accepts a registry name; the free dictionary keeps working;
    - the material used is copied as a snapshot into `experiment.h5`, test on the dummy backend.

- [ ] T76: Substrate record v2 and experiment overview [phase: 6]
  - Priority: medium
  - Depends on: T75
  - Done when:
    - the substrate record carries material reference, substrate type and dimensions;
    - a function returns all experiments of a substrate with status (planned, running, done,
      aborted) and area, for display and for running them one after another;
    - `run_substrate()` behaviour is unchanged for existing substrate mains.

- [ ] T77: Serialisable experiments and structures [phase: 6]
  - Priority: high
  - Depends on: T70
  - Done when:
    - `ExperimentSpec` and `StructureSpec` have `to_json()` / `from_json()`; structures use
      `DrawableObject.to_json()` and `structure_from_json()`;
    - `factory` callables are replaced by named, registered factories; a spec that cannot be stored
      raises a clear error that names the field;
    - the spec is stored in the experiment folder and in `experiment.h5`; a round-trip test covers
      every ported script in `mains/Experiments/`.

- [ ] T78: Settings per structure with experiment defaults [phase: 6]
  - Priority: high
  - Depends on: T70, T77
  - Done when:
    - a structure can set program source, slicing and hatching parameters, power, velocity, voxel
      model, DHM recipe and operating mode itself; unset values come from the experiment;
    - `spec.resolved()` shows the effective values per structure and validates combinations;
    - setup per structure is implemented as far as T70 allows, otherwise a follow-up todo exists;
    - existing specs give identical programs (golden files unchanged).

- [ ] T79: Experiment configuration record [phase: 6]
  - Priority: medium
  - Depends on: T70
  - Done when:
    - objective, camera information and DHM information (service version, parameter set and hash,
      pixel size and its source) are stored once per session and shown in the summary.

- [ ] T80: Layout of an experiment as data [phase: 6]
  - Priority: medium
  - Depends on: T77
  - Done when:
    - a function returns cells, structure areas, corner markers and QR code of an experiment as
      plain data, and a figure without `plt.show()`;
    - `plot_experiment()` and the substrate plot use it.

- [ ] T81: Extension point for dynamic slicing and hatching [phase: 6]
  - Priority: low
  - Depends on: T78
  - Done when:
    - a structure names a strategy and its parameters, the name and parameters are stored in the
      metadata, and an unknown strategy is refused; no new strategy is implemented (F7, F17).

### Phase 7 — DHM service integration

- [ ] T82: Role protocol `DhmMeasurement` and legacy adapter [phase: 7]
  - Priority: high
  - Depends on: T71
  - Done when:
    - `backends/protocols.py` defines `DhmMeasurement` (`configure`, `calibrate`,
      `acquire_background`, `measure`, `result`, `close`);
    - `LegacyDhmAdapter` wraps today's `Dhm`; `Experiment` talks only to the protocol;
    - golden command logs and stored data of the legacy mode are unchanged.

- [ ] T83: Capture reservation, raw references and groups in the store [phase: 7]
  - Priority: high
  - Depends on: T71
  - Done when:
    - `ExperimentStore.reserve_capture()` returns the capture id before the acquisition;
    - a capture can be stored without holograms but with `raw_ref` (host, path, SHA-256), `job_id`,
      frame metrics and selection mask; group products are stored under `/dhm_groups/<group_id>`;
    - the schema version is raised and old files are still read; `json_copies` and the unpack tool
      in `analysis/` know the new entries.

- [ ] T84: Service adapter and fake service [phase: 7]
  - Priority: high
  - Depends on: T82, T83
  - Done when:
    - `ServiceDhmAdapter` uses `dhmlink.client`; configuration section `dhm_service` (host, port,
      timeouts) exists; `dhm` stays for the legacy mode;
    - the dummy backend offers `FakeDhmService` (uses `dhmlink.testing` when installed, otherwise an
      internal fake with the same interface);
    - a dry run of an experiment in service mode passes and stores products with
      `add_dhm_product()`.

- [ ] T85: Measurement recipes and planner [phase: 7]
  - Priority: high
  - Depends on: T78, T83
  - Done when:
    - a `DhmRecipe` with the fields of section 5 exists at experiment level with an override per
      structure and per phase (before, layer, after); today's counts 10 / 20 / 1 are its defaults;
    - a `MeasurementPlanner` produces the capture positions for a single capture, a speckle set
      (pattern, count, amplitude) and a stitching raster (structure size, DHM field of view, overlap);
    - recipe and planned positions are stored; unknown recipe keys are passed to the service
      untouched, so that new modes need no change here.
    - Decision: speckle pattern, count and amplitude; stitching overlap (F5, F6).

- [ ] T86: Experiment flow with the service [phase: 7]
  - Priority: high
  - Depends on: T84, T85
  - Done when:
    - `measure()` reserves the capture, sends context and recipe, continues after the event
      `frames_acquired` as the design allows, collects results in the writing process and
      acknowledges them after they are stored;
    - calibration (exposure, OPL scan, background) runs through the service and is stored;
    - an unreachable or busy service follows the rule from T71; a restart fetches unacknowledged
      results with `sync`; all of it is covered by dummy tests.

- [ ] T87: Lab check of the service mode [phase: 7]
  - Priority: high
  - Depends on: T86
  - Done when:
    - a lab-check script has run on the lab PC against the real service and every point of its
      checklist is recorded in `WORKLOG.md`, including a comparison of the products with an offline
      reconstruction of the same raw data.

### Phase 8 — Operating modes

- [ ] T88: Operating mode in specification and storage [phase: 8]
  - Priority: high
  - Depends on: T72, T78
  - Done when:
    - `FeedbackMode` (`off`, `monitor`, `shadow`, `correct`) is a field of the experiment with an
      override per structure, default `off`, stored and shown in the summary;
    - `correct` is refused with a clear error while no controller is registered.

- [ ] T89: Layer targets [phase: 8]
  - Priority: high
  - Depends on: T72
  - Done when:
    - for sliced structures a `LayerTarget` (expected height map and mask after layer n in µm, grid
      definition, position, material) is produced from the toolpath job and stored;
    - drawn structures follow the decision of T72;
    - a test checks the target of a simple block against its analytic shape.

- [ ] T90: Evaluation hook after a layer capture [phase: 8]
  - Priority: high
  - Depends on: T86, T88, T89
  - Done when:
    - after `measure(layer)` an `Evaluator` receives product, target and context and returns a
      `DeviationReport`, which is stored under `/analysis` (uses the access layer of T66 / T67);
    - the default evaluator is an adapter to `mira`; without `mira` the modes other than `off` are
      refused with a clear message; the dummy backend has a fake evaluator;
    - the hook can run without blocking the print, as designed in T72.

- [ ] T91: Behaviour of `monitor` and `shadow` [phase: 8]
  - Priority: medium
  - Depends on: T90
  - Done when:
    - `monitor` logs RMSE, MAE and validity per layer, warns above the threshold of the spec and
      can stop the structure by rule;
    - `shadow` additionally asks a `Controller` for a `CorrectionPlan` and stores it without
      applying it; the default controller returns an empty plan;
    - both are shown in the summary and covered by dummy tests.

- [ ] T92: Simple process model in the simulated world [phase: 8]
  - Priority: low
  - Depends on: T90
  - Done when:
    - the dummy world can produce layer products from the printed programs with a simple height per
      dose model and a configurable disturbance, so that the modes can be dry-run end to end.

### Phase 9 — Camera live feed

- [ ] T93: Camera service with one owner [phase: 9]
  - Priority: medium
  - Depends on: T73
  - Done when:
    - one thread owns the camera driver and offers the latest frame and subscriptions;
    - `measure()`, the overview and the focus tools get their images through it; no behaviour change
      when the feed is off (default); the dummy camera supports it.

- [ ] T94: Recording of the print process [phase: 9]
  - Priority: low
  - Depends on: T93
  - Done when:
    - a recording can be started per experiment or structure, is written next to the experiment
      file, and a table in `experiment.h5` maps frame times to structures and layers;
    - the recording stops cleanly on abort.

- [ ] T95: Live feed interface [phase: 9]
  - Priority: medium
  - Depends on: T93
  - Done when:
    - a consumer can subscribe to frames with timestamps while an experiment prints; a headless
      test on the dummy backend receives frames during a dry run.

### Phase 10 — Core ready for a GUI

- [ ] T96: Remove blocking user interaction from the run path [phase: 10]
  - Priority: medium
  - Depends on: T74
  - Done when:
    - `messagebox_confirm` is an injectable callback (default as today);
    - the `plt.show()` calls in `experiment.py` and `substrate_plan.py` return the figure instead;
      the entry scripts show it.

- [ ] T97: Run control and progress events [phase: 10]
  - Priority: medium
  - Depends on: T74, T96
  - Done when:
    - a runner emits events for structure, layer and capture start and end, and accepts pause,
      resume and abort between layers from another thread;
    - a dummy test pauses and aborts a dry run and restarts it with `restart_experiment()`.

## 5. Recipe fields (input for T71 and T85)

| Field | Meaning | Default |
|---|---|---|
| `mode` | `off`, `legacy_raw` (today), `products`, `products_and_raw` | `legacy_raw` |
| `n_frames` | number, or mapping by structure type and phase | 10 / 20 / 1 as today |
| `selection` | policy and limits (`all`, `threshold`, `robust`, `best_k`, `min_frames`, `max_extra`) | `all` |
| `keep_raw` | `none`, `rejected`, `selected`, `all` | Decision D7 of the dhmlink plan |
| `products` | `phase`, `intensity`, `field` | phase and intensity |
| `speckle` | pattern, count, amplitude | off |
| `stitching` | overlap | off |
| `post` | free dictionary for further processing steps on the DHM-PC | empty |
| `unavailable` | `abort`, `continue`, `legacy` | Decision in T71 |

## 6. Entries for `future_todo.md`

### Changes to existing entries

- F3 (camera video) → T73, T93, T94, T95.
- F4 (reconstruction on the DHM-PC) → T71, T82–T87, together with the plans for `offaxisholo` and `dhmlink`.
- F5 (speckle reduction) and F6 (DHM stitching) → T85 for planning and capture; registration,
  fusion and stitching move to `mira` (MI-20–MI-42); group handling to `dhmlink` (DL-B9).
- F7 (strategies) and F12 (shell printing): extension point T81.

### New entries

## F15: GUI for the whole system
- **Goal:** One application to define substrates, experiments and structures, see the layout, run
  and supervise prints, with live camera and DHM views.
- **Context:** Details follow separately from the maintainer. The core is prepared by T77, T80,
  T93–T97.
- **Prerequisites:** T74, T96, T97; decision G3 (same framework as the OffAxisHolo GUI).
- **Watch out for:** The GUI process must respect the single writer of `experiment.h5`; hardware
  stays the default backend; a GUI crash must not stop a running print uncontrolled.
- **Open questions:** Scope of the first version; one application with the DHM GUI or two.

## F16: Controller and operating mode `correct`
- **Goal:** A measured deviation changes the following layers: power or dose, z offset, local
  re-exposure, additional or skipped layer.
- **Context:** Masterplan phases P4 and P5. T88–T92 provide modes, targets, reports and the
  `Controller` protocol; `shadow` stores plans without applying them.
- **Prerequisites:** T91; a measured noise floor (N-study, speckle fusion); a process model from
  dose to in-situ phase to final height.
- **Watch out for:** Layer programs are built before the print today, so late generation per layer
  is needed; global corrections can use controller variables, local ones need a new layer program;
  over-exposure cannot be undone; every correction needs hard limits and a stop rule.
- **Open questions:** First demonstration structure and objective; comparison in the phase or in
  the height domain; does the print wait for the evaluation?

## F17: Dynamic slicing and hatching per structure
- **Goal:** Slicing and hatching adapt within a structure, for example to geometry or measured
  deviations.
- **Context:** Extension point T81; related to F7 and F12.
- **Prerequisites:** T78, T81, F7.
- **Watch out for:** Reproducibility: the algorithm, its version and parameters go into the metadata.
- **Open questions:** Which algorithms first, and which quality metric decides between them?
