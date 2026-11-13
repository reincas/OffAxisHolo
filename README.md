
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

# Realtime Kernel

`offaxisholo.realtime` is the computational core of the planned realtime DHM service (specification: `docs/realtime/`). It reconstructs a stream of camera frames on the GPU (or CPU) and averages several object and background holograms:

```python
from offaxisholo.realtime import (
    DHMConfig, DemodConfig, PropagationConfig, RealtimeConfig,
    OffAxisPipeline, calibrate_order,
)

cal = calibrate_order(background_frames[:5], dc_radius=304)   # once, offline
config = RealtimeConfig(
    dhm=DHMConfig.from_preset("Zeiss 63x"),
    demod=DemodConfig(size=1024, order=cal.order, crop=512, radius=cal.radius),
    propagation=PropagationConfig(z=(-2.6e-6,)),
    precision="single",
    backend="cupy",                 # or "numpy"
)
pipe = OffAxisPipeline(config)
pipe.set_background(background_frames)   # averaged, K = conj(Ub)/(|Ub|^2+eps)
for frame in object_frames:              # hot path: demodulation + piston + average
    pipe.push(frame)
result = pipe.result()                   # compensation + propagation, once
phase = result.to_numpy()["phase"]       # wrapped phase, amplitude, ...
```

* Averaging of complex, piston corrected fields (`block`, `sliding`, `ema`) before compensation and propagation; the IFFT runs on the cropped sideband only (pixel pitch `config.crop_pitch`).
* Parity mode (`crop=None`, `mask="circle"`, `eps_rel="machine"`, `piston=False`, `precision="double"`) reproduces `HologramProcessor` up to round-off; the offline library is unchanged and serves as golden reference (`tests/test_offline_golden.py`).
* Configuration as JSON with hash (`RealtimeConfig.to_json` / `from_json` / `config_hash`).
* Tests: `pytest tests/realtime`; benchmark: `python tests/benchmark_realtime.py 1024 2048`; tools: `tools/compare_unwrapping.py`, `tools/noise_vs_n.py`.

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
│       ├── realtime/          # Realtime GPU kernel (config, core, accumulators)
│       ├── backend.py         # Array backend selection (NumPy / CuPy GPU)
│       └── __init__.py
├── docs/realtime/             # Realtime specification, todo, reference skeleton
├── tools/                     # Measurement / comparison scripts
├── tests/                     # pytest (tests/test/ = legacy scripts)
```

---

# Module Overview

* **config/**           → microscope parameters, material presets
* **io/**               → loading holograms and experimental data
* **reconstruction/**   → phase retrieval and propagation algorithms
* **simulation/**       → synthetic hologram generation
* **pipeline/**         → experimental workflow orchestration
* **realtime/**         → realtime reconstruction kernel (GPU)

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