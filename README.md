
# OffAxisHolo

Python package for reconstruction and simulation of off-axis holograms recorded with transmissive digital holographic microscopy (DHM).

The package provides tools for numerical reconstruction, phase processing, and visualization of holographic data.

---

# Installation

## Standard installation

To install the package, run:

```bash
python -m pip install .
```

---

## Development installation (editable mode)

For active development:

```bash
python -m pip install -e .
```

---

## Install with development dependencies

```bash
python -m pip install -e ".[dev]"
```

---

# Project Structure

The codebase follows a **src-layout architecture**:

```text
OffAxisHolo/
├── pyproject.toml
├── README.md
├── scripts/
│   └── clean.py
├── src/
│   └── offaxisholo/
│       ├── config/            # DHM and material presets
│       ├── io/                # Data loading utilities (in refactoring)
│       ├── reconstruction/    # Core reconstruction algorithms
│       ├── simulation/        # Numerical simulations (in development)
│       ├── pipeline/          # Workflow orchestration (in development)
│       └── __init__.py
├── tests/                     # In refactoring
```

---

# Module Overview

* **config/**           → microscope parameters, material presets
* **io/**               → loading holograms and experimental data
* **reconstruction/**   → phase retrieval and propagation algorithms
* **simulation/**       → synthetic hologram generation
* **pipeline/**         → experimental workflow orchestration

---

# Usage Examples

Usage examples will be added after ongoing refactoring is completed.


# Development Tools

This project uses:

* `black`   → code formatting
* `ruff`    → linting
* `pytest`  → testing

---

# Cleaning build artifacts

To remove temporary build files:

```bash
python scripts/clean.py
```

---

# Notes

This project is actively under refactoring towards a modular scientific architecture suitable for reproducible research in digital holography.

````