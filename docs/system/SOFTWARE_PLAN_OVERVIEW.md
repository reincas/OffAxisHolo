# Software plan — real-time feedback 2PP system: overview

Status: v0.2, 2026-11-15. v0.1 (draft of the same day) plus the maintainer decisions G1 and the
todo format below. The per-repository plans `01_…`–`04_…` of v0.1 are absorbed into the `CLAUDE.md`
(role, rules) and `TODO.md` (todos, open decisions) of each repository.

This file exists as an identical copy in all four repositories under `docs/system/`.
**Master copy:** `NanoFactorySystem_automated/docs/system/SOFTWARE_PLAN_OVERVIEW.md`. Change the master
first, then copy it unchanged into the other three repositories (one `docs(system): …` commit each).

| Repository | Python package | Branch | Todo IDs | Where the todos live |
|---|---|---|---|---|
| OffAxisHolo (upstream `reincas/OffAxisHolo`) | `offaxisholo` | `Holo_HR` | `K…`, `L…`, `P…`, carried over `B…`, `C…`, `H…` | `TODO.md` |
| DHMConnect | `dhmconnect` (working name in v0.1: `dhmlink`) | `main` | `DL-…` | `TODO.md` |
| NanoFactorySystem_automated | `nanofactorysystem` | feature branches from `main_HR` | `T…`, `F…` | `TODO.md`, `future_todo.md` |
| MIRA | `mira` | `main` | `MI-…` | `TODO.md` |

Basis: Masterplan "Echtzeit-Feedback-fähiges 2PP-System" (2026-11-13), DHM service architecture draft
v0.1 (decisions E1–E10, stages M0–M8), as-is analysis of the DHM interface (findings B1–B17),
`future_todo.md` of nanofactorysystem (F1–F17), `docs/realtime/` of OffAxisHolo.

## 1. Packages and ownership

| Package | Runs on | Owns | Must not contain |
|---|---|---|---|
| `offaxisholo` | DHM-PC (inside the service), any PC (offline, GUI) | Everything about one hologram stack: frame metrics and selection, demodulation, averaging, compensation, propagation, phase and intensity, quality numbers, timings. Analysis and live GUI. | Network server, hardware drivers, stage or position logic, stitching, fusion, target comparison. |
| `dhmconnect` | Server on the DHM-PC, client on the Working-PC and in GUIs | Protocol v1, client, DHM service process (state machine, jobs, hardware gateway, calibration procedures, result store, raw archive), slot for post-processing plug-ins. | Reconstruction math (calls `offaxisholo`), registration math (calls `mira`). Never writes `experiment.h5`. |
| `nanofactorysystem` | Working-PC | Orchestration, information model (material, substrate, experiment, structure), operating modes, measurement recipes, layer targets, later the controller, `experiment.h5` (single writer), camera live feed, system GUI. | Reconstruction, registration, stitching. |
| `mira` | DHM-PC (group post-processing), Working-PC (comparison, offline analysis) | Registration by stage position, speckle fusion, stitching, comparison with the target structure, deviation report. Existing statistical analysis. | Hardware, network, hologram reconstruction. |

## 2. Data flow

```
nanofactorysystem --measure(context, recipe)--> dhmconnect service --frames--> offaxisholo
       ^                                              |  <--ReconstructionResult--+
       |                                              +--(group closed)--> mira fusion / stitching
       +<----------- ProductSet / GroupProduct -------+
       |
       +-- LayerTarget + ProductSet --> mira compare --> DeviationReport --> nanofactorysystem
       +-- controller (later) --> CorrectionPlan --> next layer
```

## 3. Dependency rule

```
nanofactorysystem    --> dhmconnect.client, mira (both optional at import time)
dhmconnect.server    --> offaxisholo, mira (mira optional)
offaxisholo.gui      --> dhmconnect.client (optional extra)
mira                 --> offaxisholo (optional)
offaxisholo core     --> none of the others
```

No cycles. The test suite of every package passes without the other three installed
(optional imports, fakes, vendored example messages).

## 4. Shared contracts

| Contract | Produced by | Consumed by | Content |
|---|---|---|---|
| `CaptureContext` | nanofactorysystem | dhmconnect, stored with every product | experiment UUID and label, substrate, session, structure, phase, layer id, capture id, image index, group id, commanded and actual position, material snapshot, z hint |
| `MeasurementRecipe` | nanofactorysystem | dhmconnect | number of frames, selection policy, `keep_raw`, requested products, exposure policy, group kind, post-processing steps with a free parameter dict |
| `ReconstructionResult` | offaxisholo | dhmconnect, GUI, mira | phase, intensity, optional complex field, pixel pitch after crop, z, wavelength, medium, per-frame records, quality, timings, config hash, versions |
| `ProductSet`, `GroupProduct` | dhmconnect | nanofactorysystem, mira, GUI | reconstruction result plus context, job ids and raw reference; for groups the fused field or mosaic and the member transforms |
| `LayerTarget` | nanofactorysystem | mira | expected geometry of a structure after layer n in µm, position, material |
| `DeviationReport` | mira | nanofactorysystem | deviation map, RMSE, MAE, sigma, validity, registration transform, versions |
| `CorrectionPlan` | nanofactorysystem controller | nanofactorysystem planning | later, see F16 |

Conventions for all four packages: English; units in names (`_um`, `_rad`, `_s`, `_us`, `_mw`);
float32 / complex64, C order, little-endian; image axis 0 = y, axis 1 = x; timestamps UTC ISO 8601;
every message carries `schema_version`; never `pickle` across a process or network boundary.

## 5. Global decisions

Claude Code: do not decide open items. Ask the maintainer, or implement only the part that does not
depend on the decision (see the partial-work rule in §7).

| ID | Decision | Status / recommendation | Blocks |
|---|---|---|---|
| G1 | Home and name of the communication code. | **Decided 2026-11-15:** own repository `DHMConnect`, Python package `dhmconnect`. Todo IDs stay `DL-…`. | – |
| G2 | Home of the shared contracts. | Recommendation: JSON Schema plus example messages in `dhmconnect/schemas/`; other packages vendor the examples as test fixtures. | K4, DL-A3, T71, MI-10 |
| G3 | One GUI framework for the OffAxisHolo GUI and the system GUI. Legacy `DHM_User_Interface` uses PyQt5 with matplotlib canvases. | Recommendation: PySide6 with pyqtgraph for images. | L0, T74, F15 |
| G4 | Result plane (hologram or focus plane), propagation distance (preset 0.11 µm against measured about −2.6 µm), medium index, sign convention phase to height. Same as H1–H3 of OffAxisHolo. | open | K9, K10, MI-51 |
| G5 | Where fusion and stitching run. | Recommendation: DHM-PC, as a worker process started by the service, CPU first. | DL-B9, MI-60 |
| G6 | Where the target comparison runs. | Recommendation: Working-PC, because the target lives there and products are small. | T90, MI-53 |
| G7 | Single source of truth for material data (refractive indices). | Recommendation: material registry in nanofactorysystem; a snapshot travels in `CaptureContext`; OffAxisHolo presets remain a fallback for offline use. | T75, K9, MI-51 |
| G8 | Hardware access without Koala. See D1 in `DHMConnect/TODO.md`. | Decide after the inventory DL-H1. | DL-H2 |
| G9 | Python baseline and installation on both PCs; repositories with `.git` inside Seafile sync folders (B5, B10). | Recommendation: Python 3.12, NumPy 2; one virtual environment per PC with editable installs of all four packages; repositories outside the sync folders. | S0 |

## 6. Cross-package milestones

| Step | Content | Result | Hardware |
|---|---|---|---|
| S0 | Repository hygiene in all four, project files (`CLAUDE.md`, `TODO.md`, `WORKLOG.md`, done 2026-11-15), contracts v0 (DL-A3), decisions G2–G9. | Clean baselines. | no |
| S1 | OffAxisHolo product API (K2–K8), dhmconnect protocol and service on sim/replay (DL-A, DL-B1–B7, DL-C), nanofactorysystem adapter with fake (T82–T84). | Dry run of an experiment through the service without hardware. | no |
| S2 | Hardware inventory and gateway (DL-H1–H3), GPU validation (K14), experiment in service mode (T86, T87). | Masterplan stage R1. | yes |
| S3 | Information model and operating modes (T75–T81, T88–T91), MIRA product model and comparison (MI-10, MI-50–MI-53). | Masterplan stage R2 (monitor, shadow). | partly |
| S4 | Frame selection study (K13), speckle fusion and stitching (MI-20–MI-42, DL-B9, T85). | Known noise floor, large structures. | partly |
| S5 | OffAxisHolo GUI (L0–L8), camera live feed (T93–T95), system GUI (F15). | Usability. | partly |
| S6 | Process model and controller (F16). | Masterplan stage R3. Separate plan. | yes |

S3 can start in parallel to S2. S5 can start once S1 exists.

## 7. Working rules for Claude Code in all four repositories

1. The repository's own `CLAUDE.md` wins over this file.
2. Planning first: todos named "Design" end with a document and wait for maintainer approval
   (line `Status: approved <date>` written by the maintainer).
3. A todo with an open decision is never guessed. Implement the independent part, keep the todo in
   "In Progress" and add a follow-up todo that names the missing part and the decision or measurement
   it needs.
4. All four `TODO.md` files use the same format (sections Open / In Progress / Blocked / Done, fields
   Priority, Size, Depends on, Done when; format rules in the comment at the top of each file).
   Sizes: S (hours), M (about a day), L (several days).
5. The existing paths keep working: offline OffAxisHolo is the golden reference, the legacy DHM path
   of nanofactorysystem stays selectable, MIRA's publication analysis keeps its results.
6. No assumed performance numbers. Measure, record the number with date and machine, then optimise.
7. Everything is testable without hardware (dummy backend, simulated gateway, replay). Tests that
   need the lab are marked `hardware`.
8. Nothing is deleted without the maintainer's approval; legacy files move to an `attic/` (or
   `legacy/`) folder after approval.
9. A change to a shared contract is made in `dhmconnect/schemas/` first; the other repositories update
   their vendored examples in their own commits and reference the DHMConnect commit.
