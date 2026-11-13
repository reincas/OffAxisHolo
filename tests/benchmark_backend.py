"""
Runtime comparison of the reconstruction steps for the available backends and
precisions (CPU NumPy vs. GPU CuPy, double vs. single).

    python tests/benchmark_backend.py [N ...]

Each step is timed separately (median of several runs, after a warm-up run that
contains the CUDA initialisation / kernel compilation). GPU timings include the
synchronisation of the device.
"""

import statistics
import sys
import time

import numpy as np

import offaxisholo as oah
from offaxisholo import backend
from offaxisholo.reconstruction import Hologram, HologramProcessor, ReferenceHologram
from offaxisholo.reconstruction.numericalPropagation import angularSpectrum

DHM = {
    "name": "benchmark",
    "dcRadius": 60,
    "pixelPitch": [0.1e-6, 0.1e-6],
    "wavelength": 0.675e-6,
    "propagationDistance": 1.1e-7,
}
REPEATS = 5


def synthetic_hologram(n, obj_phase, seed):
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n), dtype=float)
    carrier = np.exp(2j * np.pi * (x + y) / 4)
    obj = np.exp(1j * obj_phase)
    return np.abs(obj + carrier) ** 2 + 0.05 * rng.random((n, n))


def synchronize():
    if backend.get_backend() == "cupy":
        import cupy

        cupy.cuda.Device().synchronize()


def timed(func):
    times = []
    for i in range(REPEATS + 1):
        synchronize()
        t0 = time.perf_counter()
        result = func()
        synchronize()
        if i > 0:  # first run = warm-up
            times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def benchmark(n, backend_name, precision):
    y, x = np.indices((n, n), dtype=float)
    bump = 9.0 * np.exp(-(((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * (n / 12) ** 2)))
    holo_np = synthetic_hologram(n, bump, 1)
    bg_np = synthetic_hologram(n, np.zeros((n, n)), 2)

    rows = {}
    with oah.use_backend(backend_name, precision=precision):
        rows["transfer to backend"], holo = timed(lambda: backend.asarray(holo_np))
        rows["locate order"], _ = timed(
            lambda: Hologram(data=holo, dhm_parameter=DHM).first_diffraction_order_pos
        )
        hologram = Hologram(data=holo, dhm_parameter=DHM)
        reference = ReferenceHologram(
            data=bg_np,
            first_diffraction_order_pos=hologram.first_diffraction_order_pos,
            dhm_parameter=DHM,
        )
        rows["spatial filtering"], field = timed(lambda: hologram.holo2field())
        background = reference.holo2field()
        proc = HologramProcessor(
            hologram=hologram, reference=reference, dhm_parameter=DHM
        )
        rows["compensation"], comp = timed(lambda: proc.compensate(field, background))
        rows["propagation (ASM)"], prop = timed(
            lambda: angularSpectrum(comp, 1.1e-7, 0.675e-6, 0.1e-6, 0.1e-6)
        )
        rows["propagation (ASM, padding)"], _ = timed(
            lambda: angularSpectrum(
                comp, 1.1e-7, 0.675e-6, 0.1e-6, 0.1e-6, padding=True
            )
        )
        wrapped = proc.phase(prop)
        rows["unwrap Fast 2D (CPU)"], _ = timed(
            lambda: proc.phase_unwrapping(wrapped, method="Fast 2D")
        )
        rows["unwrap Least Squares"], _ = timed(
            lambda: proc.phase_unwrapping(wrapped, method="Least Squares")
        )
        rows["full run()"], _ = timed(lambda: proc.run())
        rows["transfer to CPU"], _ = timed(lambda: oah.to_numpy(proc.phase_map))
    return rows


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [1024, 2048]
    configs = [("numpy", "double"), ("numpy", "single")]
    try:
        oah.set_backend("cupy")
        oah.set_backend("numpy")
        configs += [("cupy", "double"), ("cupy", "single")]
    except RuntimeError as err:
        print(f"GPU backend not available ({err}); CPU only.\n")

    # silence the info logging of the processor
    import logging

    logging.disable(logging.INFO)

    for n in sizes:
        results = {cfg: benchmark(n, *cfg) for cfg in configs}
        header = "".join(f"{b + '/' + p:>16s}" for b, p in configs)
        print(f"\nN = {n}  (median of {REPEATS} runs, seconds)")
        print(f"{'step':28s}{header}")
        for step in results[configs[0]]:
            row = "".join(f"{results[cfg][step]:16.4f}" for cfg in configs)
            print(f"{step:28s}{row}")


if __name__ == "__main__":
    main()
