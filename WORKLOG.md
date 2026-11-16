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
