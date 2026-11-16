# Handover from OffAxisHolo to DHMConnect (2026-11-16)

For: the maintainer and Claude Code sessions working in DHMConnect. Source: the OffAxisHolo work of
2026-11-16 (todos K1, B5, B3, K2–K7, K18), branch stack ending at `feat/k7-background` (commit
`b70838b`), not yet merged into `Holo_HR`. Everything below is the state of that branch.

What to do with this file: take the decisions into the DHMConnect `TODO.md` / `CLAUDE.md`, use
section 3 as the starting point for the `ReconstructionResult` contract (DL-A3) and section 4 for the
GPU worker (DL-B4). Nothing here changes a DHMConnect file; the requests are listed in section 7.

## 1. Decisions taken by the maintainer

| ID | Decision | Consequence for DHMConnect |
|---|---|---|
| G2 | **Decided 2026-11-16:** shared contracts live as JSON Schema plus example messages in `dhmconnect/schemas/`; the other repositories vendor the examples as test fixtures (OffAxisHolo: `tests/contracts/`). | DL-A3 owns the schemas. A contract change is made in DHMConnect first (system plan §7 rule 9). |
| O3 | Default frame selection policy is `all` until the N-study (K13) is evaluated. All policies exist. | Default of `MeasurementRecipe.selection_policy` should be `all`. |
| – | Frame metrics are **on** by default in the OffAxisHolo entry points (re-evaluate after the GPU benchmark K14). | Every result carries per-frame metrics; the service can switch them off with `metrics=False`. |
| O6 | A background never expires. A configurable maximum age (default: none) only adds a warning to the result. Limits follow after a drift measurement (Masterplan V4). | The service decides when to record a new background; the kernel only warns. |
| O10 (part) | The DHM-PC driver supports CUDA 13 → `cupy-cuda13x>=13.6`. Python is not yet installed on the DHM-PC; recommended Python 3.12. `requires-python >= 3.11`. | Same baseline for the service environment on the DHM-PC (G9). |
| O4 | **Reopened:** a fixed reference region per objective does not hold (every print has another structure, large structures are stitched). Replaced by SA1–SA7 in `docs/stitching_and_averaging.md`, to be decided in the cross-repository planning session K18. | The contract may need a per-capture reference mask (SA2, see section 6). |

## 2. Scope boundaries confirmed

- OffAxisHolo contains no network server, protocol, camera driver, state machine or raw archive. The
  former groups D–G of `docs/realtime/todo_realtime.txt` belong to DHMConnect:
  D1–D4 → DL-B1, DL-B2, DL-H2; E1 → DL-A2, DL-A4; E2 → DL-B3; E3 → DL-B4; E4 → DL-B5; E5 → DL-B10;
  E6 → DL-B7; E7 → DL-B11; F1, F2 → DL-C1, DL-C3; G1, G3, G4 → DL-H4; G2 (version in every result) →
  DL-B11 plus K15 here; G5 → DL-H5 / D5; Q3 (N, n, frame rate, latency) → DL-H3; Q7 → D4, D5.
- `docs/realtime/anweisung_realtime.txt` §4, §7, §8, §10 now point to DHMConnect; its service, camera
  and protocol parts are background reading only.
- The OffAxisHolo GUI (later, L5/L6) will only import `dhmconnect.client`; it needs DL-C1, DL-B8,
  DL-B10 and the observer role of D4.

## 3. Proposed `ReconstructionResult` contract (draft schema 0.3)

Implemented in `src/offaxisholo/realtime/result.py`. Proposed as the starting point for DL-A3; once
the schema exists, OffAxisHolo vendors the examples, validates against them and bumps to 1.0 (K16).

Message form: `header, buffers = result.to_message()`; `ReconstructionResult.from_message(header,
buffers)` reverses it. The header is JSON-able (NaN and infinities become `null`); buffers are named,
C-contiguous, little-endian NumPy arrays, described in `header["buffers"]` in the same order.

**Header fields**

| Field | Type | Meaning |
|---|---|---|
| `message_type` | str | `"ReconstructionResult"` |
| `schema_version` | str | `"0.3"` (0.2 added `timings_profiled`, 0.3 added `warnings`) |
| `created_utc` | str | UTC ISO 8601 |
| `pixel_pitch_um` | float | pixel pitch of the product after the sideband crop (`dx · N / n`) |
| `z_um` | float | propagation distance used |
| `wavelength_um` | float | laser wavelength |
| `n_medium` | float | refractive index of the propagation medium |
| `config_hash` | str | hash of the full kernel configuration |
| `frames` | list | one `FrameRecord` per object frame (below) |
| `quality` | dict | quality numbers (below) |
| `timings_s` | dict | stage timings in seconds plus caller values (below) |
| `timings_profiled` | bool | true only if the device was synchronised per stage |
| `versions` | dict | `offaxisholo`, `numpy`, array namespace (`cupy`); git revision follows with K15 |
| `config` | dict | full kernel configuration (`RealtimeConfig` as JSON) |
| `warnings` | list of str | e.g. background older than the configured maximum age |
| `buffers` | list | `{name, dtype, shape, byte_order}` per buffer |

**Buffers:** `phase_rad` (float32, wrapped phase), `intensity` (float32, linear), `field` (complex64,
only when requested). Axis 0 = y, axis 1 = x.

**FrameRecord:** `index` (0-based position in the measurement), `used` (entered the average),
`reject_reason` (null for used frames; `"forced by min_frames: …"` for frames kept only to reach
`min_frames`), `metrics` (dict, below), `acquisition` (dict passed through unchanged from the caller:
exposure, camera and host time, frame id, …).

**Per-frame metrics** (computed on the device from the demodulation FFT, no host sync per frame):
`saturated_count`, `black_count`, `intensity_p05_counts`, `intensity_median_counts`,
`intensity_p95_counts`, `sideband_fraction` (sideband energy / total energy incl. DC; drops with
fringe contrast), `sideband_peak_fraction`, `sideband_peak_dx_px`, `sideband_peak_dy_px` (peak offset
from the calibrated order), `piston_rad`, `phase_noise_rad` (NaN → `null` without a reference region,
which is today's default; see section 6).

**Quality keys (today):** `n_obj`, `n_effective`, `n_bg`, `piston_std_rad`, selection summary
(`selection_policy`, `n_frames_total`, `n_frames_used`, `selection_min_frames`,
`selection_min_frames_enforced`), background (`background_id`, `background_created_utc`,
`background_age_s`). Open (K8, blocked by K18): phase noise after plane removal, sideband SNR.

**Timing stages:** `upload`, `demodulation`, `piston`, `averaging`, `metrics`, `compensation`,
`propagation`, `download`, `total`. Without `profile=True` GPU numbers are enqueue times. The caller
can add its own values (e.g. `{"grab": 0.27}`) via `result(timings_s=...)`.

Points for DL-A3 to decide when writing the schema:
- whether `config` travels in every message or only `config_hash` plus a parameter-set reference;
- the unit name of `intensity` (today linear camera-derived values without a unit suffix);
- whether `ProductSet` wraps this message unchanged (proposal) and adds `CaptureContext`, job ids and
  the raw reference.

## 4. Entry points for the service (DL-B4)

The service imports only these names (import is lightweight: no tkinter, matplotlib, scikit-image,
OpenCV or scidatacontainer):

```python
from offaxisholo.realtime import (
    RealtimeConfig, SelectionPolicy, ReconstructionSession, reconstruct_frames,
    BackgroundModel,
)
from offaxisholo.io.background import save_background, load_background

session = ReconstructionSession(config, selection=SelectionPolicy("all"),
                                outputs=("phase", "intensity"))   # + "field" on request
session.warmup()                      # FFT plans and transfer functions before the first frame
session.set_background(bg_frames)     # or session.use_background(load_background(path))
for frame, info in frames:
    session.push(frame, info)         # hot path, no host synchronisation
result = session.result(z=None, timings_s={"grab": t_grab})   # one sync, one device→host copy
header, buffers = result.to_message()
session.reset()                       # next measurement, background kept
```

`reconstruct_frames(frames, background_or_model, config, selection=..., outputs=..., frame_info=...,
z=...)` does the same for one complete stack (forces block averaging with `n_obj = len(frames)`).

Rules the service must respect:
- **One thread per session.** The session owns the CUDA context of its pipeline; the backend globals
  (`set_backend`, `set_precision`) are not thread-safe. The GPU worker should be the only owner.
- `push` accepts host or device arrays (N, N). The `info` dict is passed through into
  `FrameRecord.acquisition` without interpretation.
- A selection policy other than `all` needs averaging mode `block` and keeps all corrected fields until
  `result()`: memory `n_obj · n²` complex values (n = crop size; e.g. 16 frames at n = 256 in double
  precision ≈ 16 MiB). Sliding / EMA modes support no selection yet; device-side threshold gating for
  live mode is OffAxisHolo K17 (useful for DL-B10).
- **Background model (K7):** `session.background_model()` returns a `BackgroundModel` (field, compensation
  factor, n_frames, quality, `model_id`, `created_utc`, `config_hash`). Saved as `.npz` without pickle;
  loading recomputes the id and rejects changed files. A model fits a configuration when its
  `config_hash` equals `RealtimeConfig.background_hash()`, which covers microscope, demodulation,
  `eps_rel`, piston and precision only: z, averaging, metrics and backend may change without a new
  background. A mismatch raises `ValueError("… record a new background")`.
- `background_max_age_s` (session argument) only adds a warning (O6). The result reports
  `background_id` and `background_age_s`; the service should store the id with every product.
- The pixel size comes from the parameter set, never from Koala (finding B17: configuration 180
  reports the pixel size of a 20x objective).

Mapping from `MeasurementRecipe` (proposal): number of frames → frames pushed; selection policy and its
parameters → `SelectionPolicy(policy, thresholds, robust_metrics, robust_k, k, rank_by, ascending,
min_frames)`; requested products → `outputs`; z hint of `CaptureContext` → `result(z=...)` (units:
metres in the kernel, `z_um` in the result). `keep_raw` stays a service matter (raw archive).

## 5. Facts for the DHM-PC and the service

- Camera: Basler acA1920-155um, ROI 1024 × 1024, 8 bit (1 MiB per frame), exposure about 16.8 ms.
- Zeiss 63x preset: pixel pitch 0.0869 µm, `dcRadius` 304, `propagationDistance` 1.1e-7 m, while a
  focus of about −2.6 µm was measured (open decision G4 / O1). z must come from the request or the
  parameter set, never be hard-wired.
- Precision: `"double"` reproduces the CPU; `"single"` is much faster on consumer GPUs but not yet
  validated on real holograms (K14 decides the GPU default).
- No GPU number exists yet. K14 (with the maintainer on the DHM-PC) records GPU model, driver, OS and
  the benchmark numbers with and without metrics in `docs/realtime/benchmarks.md`. Until then the
  service must not assume a frame rate (DL-H3 measures it).
- Install on the DHM-PC: Python 3.12, `pip install -e ".[gpu,dev]"` in OffAxisHolo,
  `python -m pytest -m gpu`, `python tests/benchmark_realtime.py 1024 2048` with and without
  `--metrics`.

## 6. Open topic that touches the contract: reference mask, averaging, stitching (K18)

Details: `docs/stitching_and_averaging.md`. Summary:
- The kernel aligns the frames of one measurement by their piston phase (constant phase offset per
  frame) before averaging. Today it uses the whole field, which is good enough for relative alignment
  of a static structure. The absolute offset of a product is levelled later by MIRA (Otsu detection,
  background set to 0).
- An honest noise number needs a structure-free reference region **per capture**. Sources: a priori
  from the print job (`LayerTarget` footprint, dilated) or a posteriori from MIRA's detection.
- Decisions for the planning session that concern DHMConnect:
  - **SA2:** format and coordinates of the mask in the contract (proposal: stage µm in `CaptureContext`
    or the recipe, rasterised in OffAxisHolo with the calibrated pixel size);
  - **SA3:** two-pass flow (re-align kept fields after MIRA's detection): never, on request, always
    (proposal: on request only; the GPU worker would have to keep the session alive until MIRA
    answers, cost to be measured in K14);
  - **SA4:** fallback flag in the result when no mask exists.
- Averaging across positions (speckle fusion) and stitching are MIRA's (MI-30, MI-40); the service only
  hands closed groups to the MIRA worker (DL-B9, G5).

## 7. Requests to DHMConnect

1. **DL-A3:** create `dhmconnect/schemas/` with `ReconstructionResult` and `FrameRecord`, starting from
   section 3; tell OffAxisHolo the commit so K16 can vendor the examples.
2. **DL-B4:** build the GPU worker on `ReconstructionSession` (section 4); store `background_id` with
   every product; call `warmup()` before the first measurement.
3. **DL-B7 (raw archive):** document the file layout; OffAxisHolo K12 will read raw archive files with
   `h5py` only and without HDF5 file locking for replay and the N-study (K13). Captures with
   `keep_raw = all` are the input for the O3 decision.
4. **DL-B10 (live mode):** sliding / EMA averaging without selection today; tell OffAxisHolo if
   live-mode gating (K17) is needed early.
5. **DL-C1, DL-B8, D4:** needed later by the OffAxisHolo GUI live mode (L5) and operation panel (L6).
6. **Planning session K18:** bring the DHMConnect view on SA2 and SA3.

## 8. References in OffAxisHolo

- `src/offaxisholo/realtime/api.py`, `result.py`, `background.py`, `metrics.py`, `selection.py`,
  `timing.py`; `src/offaxisholo/io/background.py`.
- `docs/stitching_and_averaging.md` (SA1–SA7), `docs/reconstruction_order.md`.
- `TODO.md` (K8, K12, K14, K16, K17, K18), `WORKLOG.md` (entries of 2026-11-16).
