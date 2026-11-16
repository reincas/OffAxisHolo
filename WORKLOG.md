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

### 2026-11-16 09:56 CET — [B5] Repository hygiene: legacy scripts, old notes and planning inputs to attic
- **Status:** partial (open: `Notes_reconstruction_order.txt` → `docs/`, decision G9 on `.git` / `.venv`
  / `build/` in Seafile)
- **Context:** Maintainer decisions of 2026-11-16: legacy scripts to `attic/`; scan `ToDos.txt` for
  unknown items, add them to the md files and move the file to `attic/`; delete `console.log`; move the
  planning folders to `attic/` if no todo needs them. Branch `chore/b5-hygiene`, created from
  `docs/k1-scope` (not yet merged) because it builds on the K1 state of `TODO.md`.
- **Findings:**
  - `attic/` is in `.gitignore`. Moves were done with `git mv`, so the files stay versioned with their
    history; files created in `attic/` later stay local (like `attic/conf.py`, `attic/future_todo`).
  - `ToDos.txt`: all items were already in `TODO.md` (P1–P14, O5, K14, P10–P12). Missing details added:
    the criteria for Least Squares as a default (O5), the decision on `"single"` as GPU default after
    K14, and documentation of the batch workflow `ReconstructionPipeline` ("docker" in the old notes,
    formerly `DockerImageFile`) in P13.
  - Planning folders: `Claude outputs/00, 01, CLAUDE.md` are byte-identical to `todo_claude/00, 01,
    CLAUDE_nanofactory.md`; `todo_claude/anweisung.txt` equals `docs/realtime/anweisung_realtime.txt`
    without the K1 notes. The other repositories already hold their todos (DHMConnect DL-A0…DL-H5,
    NanoFactorySystem_automated T82–T87 and F15, MIRA MI-…). Only content not found elsewhere: a link to
    an early planning artifact (decisions D1–D8, risks) in `uebergabe_echtzeit_dhm.md`, superseded by the
    specification.
  - `console.log` is written by `ReconstructionPipeline` into its output directory (`get_logger(logfile=
    f"{save_path}/console.log")`); hence `*.log` is ignored now.
  - O10: the minimum versions in `pyproject.toml` equal the versions installed on the maintainer's
    laptop; numpy 2.4.5 has no Python 3.10 wheel (`pip download … --python-version 3.10` fails), so
    `requires-python = ">=3.10"` is not achievable as written. Recorded in O10.
- **Changes:**
  - `tests/test/*` (renamed → `attic/legacy_tests/*`): legacy scripts and reference data; the untracked
    `test_4_einheit.py` moved along on disk (stays local).
  - `pyproject.toml`: `"test"` removed from `norecursedirs`, comment fixed.
  - `.gitignore`: line for `tests/test/test_4_einheit.py` removed; `*.log` added.
  - `src/offaxisholo/ToDos.txt` (renamed → `attic/ToDos.txt`).
  - `src/offaxisholo/reconstruction/console.log` (deleted): empty log file.
  - `todo_claude/*` (renamed → `attic/planning_2026-11-15/todo_claude/*`); the gitignored local files
    `Notes_guteHandhabung.txt` and `OffAxisHolo_projectStructure_18052026.txt` moved along on disk.
  - `Claude outputs/*` (renamed → `attic/planning_2026-11-15/claude_outputs/*`).
  - `README.md`: project structure matches the tree (project files, `single_functions/`, `utils.py`,
    `docs/system/`, `attic/`, `scripts/` local only).
  - `CLAUDE.md`: `attic/` described; references to `tests/test/` and `ToDos.txt` updated.
  - `TODO.md`: B5 moved to In Progress with checked sub-items; O5, O10, K14, P13, P15 updated.
  - `WORKLOG.md`: this entry.
- **Tests:** `python -m pytest`: 90 passed, 8 skipped (GPU), same as before the move.
- **Commits:** `ed74765` chore(tests): move the legacy test scripts to attic/legacy_tests [B5];
  `04e5030` chore: move the old todo notes to attic and delete console.log [B5];
  `5c18e66` chore: move the planning inputs of 2026-11-15 to attic [B5];
  `083fcd4` docs: describe the real repository tree and the attic in README and CLAUDE.md [B5];
  todo bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** maintainer decisions on `Notes_reconstruction_order.txt` (proposal: version it as
  `docs/reconstruction_order.md`) and G9; whether `attic/` should stay gitignored.

### 2026-11-16 11:40 CET — [B5] Version the reconstruction order note, close B5
- **Status:** done
- **Context:** Maintainer decisions of 2026-11-16: version `Notes_reconstruction_order.txt` if useful
  (in English); `.git`, `.venv` and `build/` stay in the Seafile folder, the maintainer knows the risks
  (G9; to be recorded in the master copy of the system plan in NanoFactorySystem_automated);
  `attic/` stays gitignored.
- **Changes:**
  - `docs/reconstruction_order.md` (new): the note (already English) converted to Markdown with tables;
    open points of §6 linked to O1, O2, P16, P17.
  - `src/offaxisholo/Notes_reconstruction_order.txt` (untracked; moved on disk → `attic/notes/`).
  - `.gitignore`: ignore line of the old note removed.
  - `CLAUDE.md`: points to `docs/reconstruction_order.md`.
  - `TODO.md`: B5 moved to Done; new backlog todos P16 (compensation before propagation vs.
    `P_z{E_obj} / P_z{E_ref}`) and P17 (numerical aberration compensation), both found during B5;
    phase table P1–P17.
  - `WORKLOG.md`: this entry.
- **Tests:** none run for this step (documentation only; the full suite passed at 09:56 with the same
  code).
- **Commits:** `8755f70` docs: version the note on the order of the reconstruction steps [B5];
  todo bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** record G9 ("kept knowingly") in the master system plan.

### 2026-11-16 11:45 CET — [B3] Target CUDA 13 and Python ≥ 3.11
- **Status:** partial (blocked: install check on the DHM-PC, `gui` / `live` extras, script entry)
- **Context:** Maintainer: the DHM-PC supports CUDA 13; Python must be installed there from scratch (the PC
  was only reached through the C server so far). Branch `build/b3-dependencies` from `chore/b5-hygiene`.
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
