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


# ---------------------------------------------------------------------------
# Phase <-> height with explicit indices (K20, decision G4). These two functions
# are the owner of the formulas for all repositories: MIRA imports them or
# mirrors them with a parity test (MI-51). There are no default indices
# (G7: no silent defaults).
# ---------------------------------------------------------------------------
def _check_optics(wavelength_um, n_structure, n_surround, phase_sign):
    if phase_sign not in (1, -1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign!r}.")
    if not wavelength_um > 0:
        raise ValueError(f"wavelength_um must be positive, got {wavelength_um!r}.")
    if not (n_structure > 0 and n_surround > 0):
        raise ValueError(
            f"Refractive indices must be positive (n_structure={n_structure!r}, "
            f"n_surround={n_surround!r})."
        )
    delta_n = n_structure - n_surround
    if delta_n == 0:
        raise ValueError(
            "n_structure equals n_surround; phase and height are not related."
        )
    return delta_n


def phase_to_height_um(
    phase_rad, *, wavelength_um, n_structure, n_surround, phase_sign
):
    """Height of a homogeneous structure from its unwrapped phase (transmission).

    Light that passes a structure of height ``h`` and index ``n_structure``
    surrounded by a medium of index ``n_surround`` gains the optical path
    ``(n_structure - n_surround) * h``. With the convention of G4 (more optical
    path -> positive phase; ``phase_sign`` corrects a setup that measures the
    opposite sign)

        phase_rad = phase_sign * 2 * pi * (n_structure - n_surround) * h / wavelength
        h_um      = phase_sign * wavelength_um * phase_rad
                    / (2 * pi * (n_structure - n_surround))

    so ``n_structure > n_surround`` and a positive phase give a positive height.
    The height is relative: the offset of the unwrapped phase (e.g. of the
    substrate) must be removed first. Normal incidence, homogeneous structure
    (Nguyen et al., Appl. Opt. 55, 5666 (2016)).

    Parameters
    ----------
    phase_rad : array or float
        Unwrapped phase in rad (NumPy or CuPy array, or a scalar).
    wavelength_um : float
        Vacuum wavelength in µm (DHM: 0.674994).
    n_structure : float
        Index of the structure (in-situ: ``cured_in_resin`` of the material).
    n_surround : float
        Index around the structure (in-situ: liquid resin; developed: air).
    phase_sign : {1, -1}
        Sign convention of the setup (parameter set).

    Returns
    -------
    array or float
        Height in µm, same type and shape as ``phase_rad``.

    Raises
    ------
    ValueError
        For ``phase_sign`` not in {1, -1}, non-positive wavelength or indices,
        or ``n_structure == n_surround``.
    """
    delta_n = _check_optics(wavelength_um, n_structure, n_surround, phase_sign)
    return phase_sign * wavelength_um * phase_rad / (2 * math.pi * delta_n)


def height_to_phase_rad(
    height_um, *, wavelength_um, n_structure, n_surround, phase_sign
):
    """Phase of a homogeneous structure of known height; inverse of
    :func:`phase_to_height_um` (same parameters and convention).

    Parameters
    ----------
    height_um : array or float
        Height in µm.
    wavelength_um, n_structure, n_surround, phase_sign
        See :func:`phase_to_height_um`.

    Returns
    -------
    array or float
        Unwrapped phase in rad.
    """
    delta_n = _check_optics(wavelength_um, n_structure, n_surround, phase_sign)
    return phase_sign * 2 * math.pi * delta_n * height_um / wavelength_um
