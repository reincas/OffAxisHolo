
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

The `gpu` extra installs `cupy-cuda13x` (CUDA 13, as on the DHM-PC). With a CUDA 12 driver, install `cupy-cuda12x` by hand instead. Run `nvidia-smi` to see which CUDA version your driver supports. Python 3.11 or newer is required.

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
from offaxisholo import ReconstructionPipeline
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

# Product Interface (DHM service, replay)

The DHMConnect service and offline replays use two entry points of `offaxisholo.realtime`. Both return a
`ReconstructionResult` that follows the shared contract `ReconstructionResult` 1.0 (schemas in
`dhmconnect/schemas/`, vendored under `tests/contracts/`): wrapped `phase_rad` and `intensity_rel`
(I_object / I_background) as float32, optional complex `field`, `pixel_um` `(y, x)`, `z_um` with its
`propagation_index`, per-frame records, quality numbers, timings, `config_hash`, `parameter_set_id` and
`versions` (package version and git hash).

```python
from offaxisholo.realtime import (
    DHMConfig, DemodConfig, RealtimeConfig, SelectionPolicy,
    ReconstructionSession, calibrate_order, reconstruct_frames, reference_mask,
)

cal = calibrate_order(background_frames, dc_radius=10)           # position of the first order
config = RealtimeConfig(
    dhm=DHMConfig.from_preset("Zeiss 63x"),
    demod=DemodConfig(size=1024, order=cal.order),
).validate()

# one measurement
result = reconstruct_frames(frames, background_frames, config,
                            selection=SelectionPolicy(),          # policy "all"
                            reference_mask=mask,                  # optional, see below
                            min_reference_fraction=0.1,
                            parameter_set_id="zeiss63x-v1")
header, buffers = result.to_message()                             # JSON header + arrays, no pickle

# frame by frame (service)
session = ReconstructionSession(config, parameter_set_id="zeiss63x-v1")
session.set_background(background_frames)                         # or use_background(model)
session.set_reference_mask(mask)                                  # per capture
for frame in frames:
    session.push(frame, {"frame_id": 0})
result = session.result(z=-2.6e-6)
```

* **Reference region (quality, K8):** `mask = reference_mask(shape, pixel_um, center_um, occupied_um=..., pose_tolerance_um=..., optical_margin_um=...)`
  marks structure-free pixels (K19). Without a mask the border ring of `compensation.ref_border` is used, else the whole
  field (`reference_fraction = 0`, `phase_noise_rad = null`). `quality` also holds frame counts, piston spread,
  selection summary and the median `sideband_snr`.
* **Frame selection:** `SelectionPolicy` (`all`, thresholds, robust outliers) decides from the per-frame metrics
  (`METRIC_NAMES`).
* **Background:** `session.background_model()` stores a reusable `BackgroundModel` (id, age, hash).
* **Version:** `offaxisholo.__version__`; `offaxisholo.git_revision()` gives the short git hash of the source tree.

---

# Project Structure

The codebase follows a **src-layout architecture**:

```text
OffAxisHolo/
├── pyproject.toml
├── README.md
├── CLAUDE.md                  # Architecture, rules and conventions for contributors
├── TODO.md                    # Maintained task list
├── WORKLOG.md                 # Append-only work log
├── src/
│   └── offaxisholo/
│       ├── config/            # DHM and material presets
│       ├── io/                # Data loading utilities (in refactoring)
│       ├── reconstruction/    # Core reconstruction algorithms
│       ├── simulation/        # Numerical simulations (in development)
│       ├── pipeline/          # Workflow orchestration (in development)
│       ├── single_functions/  # One-call reconstruction functions
│       ├── realtime/          # Realtime GPU kernel (config, core, accumulators)
│       ├── backend.py         # Array backend selection (NumPy / CuPy GPU)
│       ├── utils.py           # Logger and hologram helpers
│       └── __init__.py
├── docs/
│   ├── realtime/              # Realtime specification, old todo list, reference skeleton
│   └── system/                # Plan of the four cooperating repositories
├── tools/                     # Measurement / comparison scripts
├── tests/                     # pytest (realtime/, golden/ = golden data generator)
├── attic/                     # Local only (gitignored): archived legacy files, see docs/ATTIC_INDEX.md
└── scripts/                   # Local only (gitignored): clean.py
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

See "GPU Backend" (offline reconstruction), "Realtime Kernel" and "Product Interface" above, and the
scripts in `tools/`.


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
