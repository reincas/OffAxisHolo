# dhmlink — communication between Working-PC and DHM-PC: instructions and todos

Status: draft v0.1, 2026-11-15, waiting for maintainer approval. Read `00_SOFTWARE_PLAN_OVERVIEW.md` first.
`dhmlink` is a working name (decision G1). The package replaces protocol v6 and `DhmClient`.

## 1. Role

A new communication base over TCP/IP that carries commands, status and structured information
(dictionaries) as well as images, and the service process on the DHM-PC that uses it.

- **Protocol v1:** framed messages with a JSON header and binary buffers, request ids, asynchronous
  jobs, events, errors that do not close the connection, version and capability handshake.
- **Service on the DHM-PC:** owns the hardware access and the GPU. Takes orders with context, records
  frames, calls `offaxisholo`, optionally calls `mira` for group products, keeps results until they
  are acknowledged, archives raw holograms by rule.
- **Client:** small library (NumPy plus the transport library) for nanofactorysystem and for GUIs.

Out of scope: reconstruction math, registration math, stage control, writing `experiment.h5`.

## 2. What is known today

- `DhmServ.exe` (C#, LyncéeTec Koala SDK) listens on `192.168.22.2:27182`; the Working-PC reaches it
  through the network card `Lyncee_Eth` (192.168.22.1/24).
- Protocol v6: 55 command codes, no length framing, no request ids, no timeouts, one client at a
  time, error codes are lost in the client.
- Camera: Basler acA1920-155um, ROI 1024 × 1024, 8 bit, 1 MiB per frame; exposure 16.8 ms; one image
  fetch takes 17 ms in the median.
- Camera, OPL motor, laser sources and measurement configurations are reachable only through the
  Koala SDK today. `SetConfig` loads a configuration and switches the laser on; there is no command
  to read the laser state. Koala configuration 180 reports a wrong pixel size for 63x (B17).
- Unknown: GPU, CUDA version and operating system of the DHM-PC; the build of `DhmServ` running
  there; whether Koala releases the camera; controller and interface of the OPL motor and the laser.

## 3. Rules

1. The protocol is specified before it is implemented (DL-A2) and is independent of the hardware
   access. No Koala types, names or codes appear in protocol or client.
2. Hardware is reached only through the `HardwareGateway` interface. The simulated and the replay
   gateway come first; everything else is developed and tested against them.
3. New measurement modes are added as handlers with a capability string and a parameter dictionary.
   They never require a protocol change (DL-A5).
4. Never `pickle`. Every header is validated against its schema; unknown fields are ignored, missing
   mandatory fields are an error reply.
5. Bounded queues everywhere. Dropped frames are counted and reported, never silent.
6. The legacy path (v6, `DhmServ`, `nanofactorysystem.dhm`) stays untouched and usable until the
   maintainer switches over. Only one of the two paths may hold the hardware at a time.
7. No assumed frame rates or latencies. DL-H3 measures them.

## 4. Open decisions

| ID | Decision | Options and recommendation | Blocks |
|---|---|---|---|
| D1 | Hardware access without Koala (maintainer's wish: no Koala if possible). | H1 `DhmServ` as local gateway (no C# change, keeps Koala). H2 camera through pypylon, motor and laser through `DhmServ`. H3 load the Koala assemblies with pythonnet. H4 own drivers for camera, OPL motor and laser, no Koala at all. H4 needs facts that only DL-H1 can deliver. Recommendation: decide after DL-H1; the camera part of H4 (pypylon) is the likely first step. | DL-H2 |
| D2 | Transport. | ZeroMQ over TCP (ROUTER / DEALER for commands, PUB / SUB for events), or plain TCP with length-prefixed frames. Recommendation: ZeroMQ; keep the framing specification transport-neutral. | DL-A2 |
| D3 | Header encoding. | JSON (recommended, readable in logs) or MessagePack. | DL-A2 |
| D4 | Several clients at once. | One controller plus any number of observers (GUI watches while an experiment runs), control handed over explicitly. Recommended. | DL-B3, L5 |
| D5 | Port, bind address, authentication. | Port 27183, bind to the direct-link address only, no authentication inside the closed link. | DL-B5 |
| D6 | Calibration (exposure, OPL scan, background) runs in the service. | Yes (E3 of the architecture draft). | DL-B8 |
| D7 | Result delivery and raw data rule. | Store and forward with acknowledgement (E7). `keep_raw` default `selected` or `all`, retention time. | DL-B6, DL-B7 |
| D8 | Group post-processing: inside the service process or a worker process (G5). | Worker process. | DL-B9 |
| D9 | Compression of products and raw frames. | None until measured; raw holograms gain almost nothing with gzip. | DL-A2 |
| D10 | Start of the service on the DHM-PC: Windows service, Task Scheduler or manual. | Decide with DL-H1. | DL-H4 |

## 5. Package layout (proposal)

```
dhmlink/
  schemas/       contracts as JSON Schema plus example messages (G2)
  protocol/      messages, framing, versions, error codes
  client/        DhmServiceClient
  server/        state machine, job manager, threads, configuration
  hardware/      gateway interface, sim, replay, real implementations
  calibration/   exposure, OPL scan, background
  storage/       result store (until acknowledged), raw archive
  postprocess/   plug-in registry (fusion, stitching from mira)
  testing/       in-process service for the tests of other packages
```

## 6. Todos

### Phase A — specification (no hardware)

- [?] **DL-A1 Repository skeleton (S).** `pyproject.toml` with extras `client`, `server`, `sim`;
  `CLAUDE.md`, `TODO.md`, `WORKLOG.md` with the nanofactorysystem conventions; pytest setup.
  Needs G1.

- [?] **DL-A2 Design: protocol v1 (M).** `docs/PROTOCOL_v1.md`: framing (part 1 header, further
  parts buffers; each buffer described by name, dtype, shape, byte order), mandatory header fields
  (`protocol`, `type`, `req_id`, optional `context_id`), error model (code plus readable reason),
  handshake `hello` (protocol, service version, OffAxisHolo commit, parameter set hash, capability
  list), command table, events, timeouts and heartbeat, size limits, versioning rules.
  Commands: `hello`, `status`, `describe`, `open_context`, `close_context`, `configure`, `calibrate`,
  `acquire_background`, `measure`, `open_group`, `close_group`, `job_status`, `result`, `ack`, `sync`,
  `abort`, `live_start`, `live_stop`, `gateway_info`, `call`, `shutdown`.
  Events: `state_changed`, `frames_acquired`, `job_done`, `job_failed`, `warning`, `live_frame`.
  Needs D2, D3, D9. Ends with the maintainer's approval.

- [ ] **DL-A3 Contracts (M).** JSON Schema and one valid example each for `CaptureContext`,
  `MeasurementRecipe`, `FrameRecord`, `ProductSet`, `GroupSpec`, `GroupProduct`, `Status`, `Error`;
  a validation helper; a script that exports the examples for vendoring into the other repositories.
  Needs G2.

- [ ] **DL-A4 Messages (M).** Message classes, encoding and decoding with ndarray buffers without
  copies. Done when: round-trip tests pass for every message type, including empty and large buffers,
  and a malformed header produces an error reply instead of an exception in the receive loop.

- [ ] **DL-A5 Generic command channel (M).** `call(name, params) -> (dict, buffers)` with a handler
  registry in the service; `describe` lists commands, parameter schemas and capabilities.
  Done when: a test adds a new mode as a handler without touching protocol or client code.

### Phase B — service on simulation and replay

- [ ] **DL-B1 Hardware gateway (M).** Interface: `configure`, exposure get / set in µs, gain, OPL get
  / move, laser on / off / state, `grab(n)` and streaming with frame id, camera time and host time,
  device information. `SimGateway` ports the physics of `DummyDhmClient` (carrier fringes, noise,
  contrast falling off around an OPL optimum). `ReplayGateway` plays frames from `experiment.h5` or
  from the raw archive.

- [ ] **DL-B2 Acquisition (M).** Acquisition thread, bounded ring buffer, dropped-frame counter, the
  settings in force stored with every frame.

- [ ] **DL-B3 State machine and jobs (M–L).** Service states `STARTING`, `HW_WAIT`, `IDLE`,
  `CONFIGURED`, `CALIBRATING`, `BACKGROUND`, `READY`, `MEASURING`, `LIVE`, `ERROR`. Job states
  `QUEUED`, `ACQUIRING`, `PROCESSING`, `DONE`, `DELIVERED`, `ACKED`, `FAILED`, `ABORTED`. Bounded job
  queue with `busy` reply, abort, commands allowed per state. Needs D4 for the roles.

- [ ] **DL-B4 Reconstruction worker (M).** Exactly one thread owns the `offaxisholo` session and the
  CUDA context. Maps `ReconstructionResult` to `ProductSet`. Depends on K4, K6 of the OffAxisHolo plan.

- [ ] **DL-B5 Network front end (M).** Command and event sockets, handshake, schema validation,
  error replies, heartbeat. Needs D2, D5.

- [ ] **DL-B6 Result store (M).** Products stay on disk until `ack`; `sync(context)` lists what is
  not acknowledged; survives a restart of the service. Needs D7.

- [ ] **DL-B7 Raw archive (M).** `<root>/<experiment_uuid>/captures/<capture_id>_<job_id>.h5` with
  frames (original bit depth, one chunk per frame), frame records, context and request; `manifest.json`;
  SHA-256; files immutable after writing; rule `keep_raw` = `none`, `rejected`, `selected`, `all`;
  retention; asynchronous writer; disk-space warning in `status`. Needs D7 for defaults.

- [ ] **DL-B8 Calibration procedures (M–L).** Exposure optimisation (port of `optImage` and
  `optImageMedian`), OPL scan (port of `motorscan`), background acquisition with id and quality.
  Done when: tests on `SimGateway` find a known optimum. Needs D6.

- [ ] **DL-B9 Groups and post-processing (M).** `open_group` / `close_group` for speckle sets and
  stitching rasters, nested groups, a group job that hands the member products to a plug-in and
  returns a `GroupProduct`. Without `mira` installed the capability is absent and the group returns
  its member list only. Depends on MI-60. Needs D8.

- [ ] **DL-B10 Live mode (M).** `live_start` / `live_stop`; preview frames over the event channel,
  reduced in size and rate; optional live phase and intensity; never during `MEASURING`.

- [ ] **DL-B11 Diagnostics (S).** `status`, `gateway_info`, logging with timings per stage and queue
  depth, versions in every product.

### Phase C — client

- [ ] **DL-C1 Client (M).** `DhmServiceClient`: `connect`, `open_context`, `configure`, `calibrate`,
  `acquire_background`, `measure -> Job`, `result`, `ack`, `sync`, `abort`, event subscription, live
  subscription, `call`. Exceptions `DhmUnavailable`, `DhmBusy`, `DhmProtocolError`,
  `DhmRemoteError(code, reason)`. Timeouts on every call. Thread-safe.

- [ ] **DL-C2 Test service (S–M).** `dhmlink.testing`: starts a service with `SimGateway` in the
  test process on a free port. Used by nanofactorysystem (T84) and the OffAxisHolo GUI (L7).

- [ ] **DL-C3 Examples and documentation (S).** One script that measures against the simulated
  service; README.

### Phase H — hardware (DHM-PC, with the maintainer)

- [ ] **DL-H1 Hardware inventory (S–M, on site, read-only).** Find out: does pypylon reach the
  camera while Koala is closed, and while it is open; which controller and interface drive the OPL
  motor; how the laser sources are switched and which interlocks exist; what loading a Koala
  measurement configuration actually sets; which `DhmServ` build runs; GPU, CUDA, operating system,
  Python. Result: `docs/HARDWARE_ACCESS.md` with a recommendation among the options H1–H4 of D1. Decides D1.

- [?] **DL-H2 Real gateway (L).** Implement what D1 selects. Hardware tests carry the marker
  `hardware`. Blocked by DL-H1.

- [ ] **DL-H3 Measurements (S).** Frame rate, latency, transfer time locally and over the link, GPU
  times. Record in `docs/benchmarks.md`.

- [ ] **DL-H4 Operation (M).** Autostart and watchdog, log rotation, runbook, firewall and bind
  address, long run over hours. Needs D10.

- [ ] **DL-H5 Safety (S–M).** Readable laser state, laser off on shutdown by rule, exclusive
  hardware lock that refuses to start while the legacy path holds the hardware.

## 7. Suggested order

DL-A1 → DL-A2 (approval) → DL-A3, DL-A4 → DL-B1, DL-B2 → DL-B3, DL-B5 → DL-B4, DL-B6, DL-C1, DL-C2 →
dry run with nanofactorysystem (T84) → DL-B7, DL-B8, DL-A5 → DL-B10, DL-B11 → DL-B9 (needs `mira`).
DL-H1 can run at any time and should run early; DL-H2–DL-H5 follow it.
