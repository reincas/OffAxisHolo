
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

## Install with GPU support (NVIDIA CUDA)

```bash
python -m pip install -e ".[gpu]"
```

The `gpu` extra installs `cupy-cuda12x`. With a CUDA 13 driver, install `cupy-cuda13x` instead. Run `nvidia-smi` to see which CUDA version your driver supports.

---

# GPU Backend

All calculations (spatial filtering, aberration compensation, propagation, simulation) run on the selected array backend:

| Backend   | Device            | Requirement            |
|-----------|-------------------|------------------------|
| `"numpy"` | CPU (default)     | –                      |
| `"cupy"`  | NVIDIA GPU (CUDA) | `pip install ".[gpu]"` |

```python
import offaxisholo as oah
from offaxisholo.single_functions import reconstruct_hologram

# per call
processor = reconstruct_hologram(holo, background, backend="cupy", return_numpy=True)

# globally / temporarily
oah.set_backend("cupy")
with oah.use_backend("cupy", precision="single"):
    processor = reconstruct_hologram(holo, background)
phase = oah.to_numpy(processor.phase_map)   # or processor.to_cpu()

# pipeline
pipe = ReconstructionPipeline(data_path, save_path, backend="cupy")
```

Notes:

* Result arrays stay on the GPU until they are converted with `to_numpy()` / `processor.to_cpu()`. Plotting and saving convert them automatically.
* **Precision:** `"double"` (float64/complex128) is the default and reproduces the CPU results. `"single"` (float32/complex64) is much faster on consumer GPUs, whose float64 throughput is only 1/32 to 1/64 of their float32 throughput. It is not yet validated for quantitative evaluation.
* **Phase unwrapping:** the default `"Fast 2D"` (scikit-image) always runs on the CPU, so GPU data is copied to the host and back. `phase_unwrapping_method="Least Squares"` (DCT, Ghiglia & Romero) runs fully on the GPU. Its results differ from `"Fast 2D"` and are not yet validated (see `ToDos.txt`).
* Tests: `pytest tests/test_backend.py` (GPU tests are skipped without CUDA); runtime comparison: `python tests/benchmark_backend.py 1024 2048`.

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
│       ├── backend.py         # Array backend selection (NumPy / CuPy GPU)
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