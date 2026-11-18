# Attic index

The `attic/` folder is local and not on GitHub (decision X-B, 2026-11-18). The files below were moved
into `attic/` with `git mv` on 2026-11-16 (B5) and stayed versioned there; on 2026-11-18 the maintainer
decided to apply X-B to them as well. They were removed from Git with `git rm --cached` and stay on the
maintainer's PC at the same path. To restore a file anywhere:

```
git show <last commit>:<path in Git> > <target path>
```

| Path in Git (last) | Original path before B5 | Last commit | Removed from Git | Reason |
|---|---|---|---|---|
| `attic/ToDos.txt` | `src/offaxisholo/ToDos.txt` | `04e5030` | 2026-11-18 | B5 old todo notes, items are in TODO.md (group P) |
| `attic/legacy_tests/ablage 2.py` | `tests/test/ablage 2.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/ablage.py` | `tests/test/ablage.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker/data_test_docker.npy` | `tests/test/test/docker/data_test_docker.npy` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker/data_test_docker_information.txt` | `tests/test/test/docker/data_test_docker_information.txt` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker_image/data_test_docker_-1e-06.npy` | `tests/test/test/docker_image/data_test_docker_-1e-06.npy` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker_image/data_test_docker_-1e-06_information.txt` | `tests/test/test/docker_image/data_test_docker_-1e-06_information.txt` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker_image/data_test_docker_-5.555555555555555e-07.npy` | `tests/test/test/docker_image/data_test_docker_-5.555555555555555e-07.npy` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker_image/data_test_docker_-5.555555555555555e-07_information.txt` | `tests/test/test/docker_image/data_test_docker_-5.555555555555555e-07_information.txt` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker_image/data_test_docker_-7.777777777777778e-07.npy` | `tests/test/test/docker_image/data_test_docker_-7.777777777777778e-07.npy` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test/docker_image/data_test_docker_-7.777777777777778e-07_information.txt` | `tests/test/test/docker_image/data_test_docker_-7.777777777777778e-07_information.txt` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test_dhm.py` | `tests/test/test_dhm.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test_docker.py` | `tests/test/test_docker.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test_filedialog.py` | `tests/test/test_filedialog.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test_usaf1951.py` | `tests/test/test_usaf1951.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/legacy_tests/test_zernike.py` | `tests/test/test_zernike.py` | `ed74765` | 2026-11-18 | B5 legacy test script or data, not collected |
| `attic/planning_2026-11-15/claude_outputs/00_SOFTWARE_PLAN_OVERVIEW.md` | `Claude outputs/00_SOFTWARE_PLAN_OVERVIEW.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/claude_outputs/01_OFFAXISHOLO_PLAN.md` | `Claude outputs/01_OFFAXISHOLO_PLAN.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/claude_outputs/02_DHMLINK_PLAN.md` | `Claude outputs/02_DHMLINK_PLAN.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/claude_outputs/03_NANOFACTORYSYSTEM_PLAN.md` | `Claude outputs/03_NANOFACTORYSYSTEM_PLAN.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/claude_outputs/04_MIRA_PLAN.md` | `Claude outputs/04_MIRA_PLAN.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/claude_outputs/CLAUDE.md` | `Claude outputs/CLAUDE.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/todo_claude/00_SOFTWARE_PLAN_OVERVIEW.md` | `todo_claude/00_SOFTWARE_PLAN_OVERVIEW.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/todo_claude/01_OFFAXISHOLO_PLAN.md` | `todo_claude/01_OFFAXISHOLO_PLAN.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/todo_claude/CLAUDE_nanofactory.md` | `todo_claude/CLAUDE_nanofactory.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/todo_claude/anweisung.txt` | `todo_claude/anweisung.txt` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/todo_claude/todo.txt` | `todo_claude/todo.txt` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
| `attic/planning_2026-11-15/todo_claude/uebergabe_echtzeit_dhm.md` | `todo_claude/uebergabe_echtzeit_dhm.md` | `5c18e66` | 2026-11-18 | B5 planning input of 2026-11-15, superseded by plan v0.4 |
