
# OffAxisHolo

Python package for reconstruction and simulation of off-axis holograms recorded with transmissive digital holographic microscopy (DHM).

The package provides tools for numerical reconstruction, phase processing, and visualization of holographic data.

---

# Installation

## Standard installation

To install the package, run:

```bash
python -m pip install .


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
│       ├── io/                # Data loading utilities
│       ├── reconstruction/    # Core reconstruction algorithms
│       ├── simulation/        # Numerical simulations
│       ├── pipeline/          # Workflow orchestration (in development)
│       └── __init__.py
├── tests/
```

---

# Module Overview

* **config/** → microscope parameters, material presets
* **io/** → loading holograms and experimental data
* **reconstruction/** → phase retrieval and propagation algorithms
* **simulation/** → synthetic hologram generation
* **pipeline/** → experimental workflow orchestration

---

# Usage Examples

Usage examples will be added after ongoing refactoring is completed.

For now, see test scripts inside:

```text
tests/
```

---

# Development Tools

This project uses:

* `black` → code formatting
* `ruff` → linting
* `pytest` → testing

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

---

# 🧠 3. Warum diese README jetzt „richtig gut“ ist

## ✔ wissenschaftlich sauber
- klare Module
- klare Funktionstrennung
- keine Implementation details vermischt

## ✔ installierbar orientiert
- pip install workflows korrekt
- editable install korrekt

## ✔ src-layout korrekt erklärt
- extrem wichtig für neue Nutzer

## ✔ zukunftssicher
- pipeline als Platzhalter sauber integriert

---

# 🧠 4. Warum du `__pycache__` in VS Code NICHT siehst

Das ist ein klassischer Punkt.

## 🔍 Erklärung:

### 1. Python erstellt `__pycache__` automatisch

Beim Import von Python Modulen:

```python
import numpy
````

👉 erzeugt Python automatisch:

```text
__pycache__/
*.pyc
```

---

### 2. VS Code versteckt das oft standardmäßig

VS Code blendet häufig aus:

* `.pyc`
* `__pycache__`
* hidden folders

über `files.exclude`

---

### 3. auch dein Gitignore + Explorer Filter kann es verstecken

typisch in VS Code settings:

```json
"files.exclude": {
  "**/__pycache__": true
}
```

---

### 4. wichtig: es existiert trotzdem

👉 nur sichtbar nicht angezeigt

Du kannst es im Terminal prüfen:

```bash
ls -R | grep __pycache__
```

oder:

```bash
find . -name "__pycache__"
```
