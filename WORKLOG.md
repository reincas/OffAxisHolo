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
  - `docs/handover/to_dhmconnect_2026-11-16.md` (new): decisions (G2, O3, metrics default, O6, O10,
    O4 reopened), scope mapping of the old groups D–G, draft `ReconstructionResult` schema 0.3 for
    DL-A3, `ReconstructionSession` usage and rules for DL-B4, background model, DHM-PC facts, the
    contract side of K18 (SA2–SA4), requests.
  - `docs/handover/to_nanofactorysystem_2026-11-16.md` (new): updates for the master system plan (G2,
    G9, CUDA 13, SA1–SA7 as global decisions, K18 milestone), recipe defaults and selection policies,
    result content for `ProductSet` / `experiment.h5`, the `LayerTarget` view on the a priori mask,
    open decisions, requests.
- **Tests:** none (documentation only).
- **Commits:** `21589eb` docs: add handover notes for DHMConnect and nanofactorysystem [K16]
- **Follow-ups:** the maintainer hands the notes to the two repositories; the master system plan is
  updated there. No new todo here.
