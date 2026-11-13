"""
Elementary conversions of complex fields as pure functions (no class state).

They are used by the offline classes (HologramCore, HologramProcessor) and by
the realtime kernel, so that both produce identical outputs. Theory and
references are documented in HologramCore (base_class.py) and
HologramProcessor.phase_to_height (processor_class.py).
"""

import math

from .. import backend

INTENSITY_DB_EPS = 1e-12  # avoids log10(0) for the dB intensity


def phase(field):
    """Wrapped phase arg(E) = atan2(Im E, Re E) in (-pi, pi]."""
    return backend.angle(field)


def intensity(field, mode="db"):
    """
    Intensity of a complex field: 'linear' -> |E|^2,
    'db' -> 10 * log10(|E|^2 + 1e-12).
    """
    xp = backend.get_namespace(field)
    result = xp.abs(field) ** 2
    if mode.lower() == "db":
        result = 10 * xp.log10(result + INTENSITY_DB_EPS)
    elif mode.lower() != "linear":
        raise ValueError("Intensity mode must be 'linear' or 'db'.")
    return result


def phase_to_height(phase_unwrapped, wavelength, n_structure, n_medium):
    """
    Height of a homogeneous structure in transmission (normal incidence):

        h = lambda0 * delta_phi / (2 * pi * (n_structure - n_medium)).

    The height is relative (arbitrary phase offset of the unwrapped phase).
    """
    delta_n = n_structure - n_medium
    if delta_n == 0:
        raise ValueError(
            "Refractive index of structure and medium are identical; "
            "height cannot be calculated."
        )
    return wavelength * phase_unwrapped / (2 * math.pi * delta_n)
