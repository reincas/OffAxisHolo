"""
Generate the golden reference of the offline reconstruction library.

The file golden_offline.npz freezes the results of the offline API
(Hologram / ReferenceHologram / HologramProcessor via reconstruct_hologram,
angularSpectrum) on a small synthetic hologram. tests/test_offline_golden.py
checks that refactorings (e.g. for the realtime kernel) do not change them.

Only regenerate on purpose (intended change of the offline results):

    python tests/golden/make_golden_offline.py
"""

import logging
from pathlib import Path

import numpy as np

N = 128
DHM = {
    "name": "golden",
    "dcRadius": 10,
    "pixelPitch": [0.1e-6, 0.1e-6],
    "wavelength": 0.675e-6,
    "propagationDistance": 1.1e-7,
}
MATERIAL = {"refractiveIndex": 1.5}
GOLDEN_FILE = Path(__file__).with_name("golden_offline.npz")


def synthetic_inputs():
    """Object and background hologram (float64, deterministic PCG64 stream)."""
    rng = np.random.default_rng(2026)
    y, x = np.indices((N, N), dtype=float)
    carrier = np.exp(2j * np.pi * (x * 32 / N + y * 32 / N))
    aberration = 3.0 * (((x - N / 2) / N) ** 2 + ((y - N / 2) / N) ** 2)
    bump = 6.0 * np.exp(-(((x - N / 2) ** 2 + (y - N / 2) ** 2) / (2 * 12.0**2)))

    def holo(phase):
        obj = np.exp(1j * (phase + aberration))
        return np.abs(obj + carrier) ** 2 + 0.05 * rng.random((N, N))

    return holo(bump), holo(np.zeros((N, N)))


def compute():
    from offaxisholo.reconstruction.numericalPropagation import angularSpectrum
    from offaxisholo.single_functions import (
        reconstruct_hologram,
        reconstruct_single_hologram,
    )

    holo, bg = synthetic_inputs()
    kw = {"dhm_dictionary": DHM, "material_dictionary": MATERIAL}
    proc = reconstruct_hologram(holo, bg, propagation_padding=True, **kw)
    single = reconstruct_single_hologram(holo, **kw)
    comp = proc.field_compensated
    return {
        "order": np.array(proc.hologram.first_diffraction_order_pos),
        "radius": np.array(proc.hologram.radius_mask),
        "field_compensated": comp,
        "field_propagated": proc.field_propagated,
        "phase_map": proc.phase_map,
        "height_profile": proc.height_profile,
        "intensity_db": proc.intensity_reconstructed,
        "single_phase_map": single.phase_map,
        "asm": angularSpectrum(comp, 2e-6, 0.675e-6, 0.1e-6, 0.1e-6),
        "asm_padding": angularSpectrum(
            comp, -1.5e-6, 0.675e-6, 0.1e-6, 0.1e-6, padding=True
        ),
    }


if __name__ == "__main__":
    logging.disable(logging.INFO)
    np.savez_compressed(GOLDEN_FILE, **compute())
    print(f"written: {GOLDEN_FILE}")
