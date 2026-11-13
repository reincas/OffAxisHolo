"""
Benchmark of the realtime kernel (C12 / gate G1.7 of docs/realtime).

    python tests/benchmark_realtime.py [N ...] [--frames 200] [--backend cupy]

For every configuration (crop n, precision, padding) it measures
  * the stage times of one frame (upload, demodulation, piston, accumulation)
    - GPU: CUDA events, CPU: perf_counter; median over the frames,
  * the steady-state throughput of push() in frames/s (synchronised at the end),
  * the time of one result (compensation + propagation, one z) and of the
    download of the phase.
Camera frames are uint16 (raw camera format, converted on the device).
Run it on the measurement PC; the numbers decide on padding, rfft2, kernel
fusion and CUDA graphs (C13).
"""

import argparse
import statistics
import time

import numpy as np

from offaxisholo import backend
from offaxisholo.realtime import (
    CompensationConfig,
    DemodConfig,
    DHMConfig,
    OffAxisPipeline,
    PropagationConfig,
    RealtimeConfig,
)


def synthetic_frames(n, count, seed=0):
    """uint16 off-axis holograms, carrier at (N/4, N/4), random piston."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n), dtype=np.float32)
    reference = np.exp(2j * np.pi * (x + y) / 4)
    obj = np.exp(1j * 3 * np.exp(-((x - n / 2) ** 2 + (y - n / 2) ** 2) / (n / 8) ** 2))
    frames = []
    for _ in range(count):
        holo = np.abs(obj * np.exp(1j * rng.normal(0, 0.3)) + reference) ** 2
        holo = holo + 0.05 * rng.standard_normal(holo.shape)
        frames.append(np.clip(holo * 15000, 0, 65535).astype(np.uint16))
    return frames


class StageTimer:
    def __init__(self, gpu):
        self.gpu = gpu
        self.times = {}
        if gpu:
            import cupy

            self._cupy = cupy

    def mark(self):
        if self.gpu:
            event = self._cupy.cuda.Event()
            event.record()
            return event
        return time.perf_counter()

    def add(self, name, start, stop):
        self.times.setdefault(name, []).append((start, stop))

    def medians_ms(self):
        if self.gpu:
            self._cupy.cuda.Device().synchronize()
            elapsed = self._cupy.cuda.get_elapsed_time
            return {
                k: statistics.median(elapsed(a, b) for a, b in v)
                for k, v in self.times.items()
            }
        return {
            k: statistics.median((b - a) * 1e3 for a, b in v)
            for k, v in self.times.items()
        }


def synchronize(gpu):
    if gpu:
        import cupy

        cupy.cuda.Device().synchronize()


def benchmark(n, crop, precision, padding, backend_name, frames):
    gpu = backend_name == "cupy"
    config = RealtimeConfig(
        dhm=DHMConfig(wavelength=0.675e-6, pixel_pitch=(0.1e-6, 0.1e-6), dc_radius=10),
        demod=DemodConfig(
            size=n, order=(-n // 4, -n // 4), crop=crop, radius=n / 8 - 1
        ),
        compensation=CompensationConfig(piston=True),
        propagation=PropagationConfig(z=(-2e-6,), padding=padding),
        precision=precision,
        backend=backend_name,
    )
    pipe = OffAxisPipeline(config)
    xp = pipe.xp
    pipe.warmup()
    pipe.set_background(frames[:16])
    for frame in frames[:5]:  # warm-up
        pipe.push(frame)

    # stage times (same calls as OffAxisPipeline.push)
    timer = StageTimer(gpu)
    for frame in frames:
        t0 = timer.mark()
        device_frame = xp.asarray(frame)
        t1 = timer.mark()
        U = pipe.demodulate(device_frame)
        t2 = timer.mark()
        _, correction = pipe._piston(U, pipe.Ub, pipe._ref_weight)
        U = U * correction
        t3 = timer.mark()
        pipe._acc.add(U)
        t4 = timer.mark()
        timer.add("upload", t0, t1)
        timer.add("demodulation", t1, t2)
        timer.add("piston", t2, t3)
        timer.add("accumulate", t3, t4)
    stages = timer.medians_ms()

    # steady-state throughput of push (host frames incl. upload)
    pipe.reset()
    synchronize(gpu)
    t0 = time.perf_counter()
    for frame in frames:
        pipe.push(frame)
    synchronize(gpu)
    fps = len(frames) / (time.perf_counter() - t0)

    # one result + download
    timer = StageTimer(gpu)
    for _ in range(10):
        a = timer.mark()
        res = pipe.result()
        b = timer.mark()
        backend.to_numpy(res.phase)
        c = timer.mark()
        timer.add("result (comp.+prop.)", a, b)
        timer.add("download phase", b, c)
    stages.update(timer.medians_ms())
    return stages, fps


def main():
    parser = argparse.ArgumentParser(description="Realtime kernel benchmark")
    parser.add_argument("sizes", nargs="*", type=int, default=[1024, 2048])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--backend", default=None, help="numpy or cupy (default: both)")
    args = parser.parse_args()

    backends = [args.backend] if args.backend else ["numpy"]
    if args.backend is None:
        try:
            backend.namespace("cupy")
            backends.append("cupy")
        except RuntimeError as err:
            print(f"GPU backend not available ({err}); CPU only.\n")

    for n in args.sizes:
        frames = synthetic_frames(n, args.frames)
        for backend_name in backends:
            print(f"\nN = {n}, backend {backend_name}, {args.frames} frames")
            rows = []
            for crop in (None, n // 2, n // 4):
                for precision in ("single", "double"):
                    for padding in (False, True):
                        stages, fps = benchmark(
                            n, crop, precision, padding, backend_name, frames
                        )
                        label = f"n={crop or n:<5d} {precision:6s} pad={padding!s:5s}"
                        rows.append((label, stages, fps))
            names = list(rows[0][1])
            print(
                f"{'configuration':28s}"
                + "".join(f"{s[:14]:>15s}" for s in names)
                + f"{'push fps':>12s}"
            )
            for label, stages, fps in rows:
                print(
                    f"{label:28s}"
                    + "".join(f"{stages[s]:13.3f}ms" for s in names)
                    + f"{fps:12.1f}"
                )


if __name__ == "__main__":
    main()
