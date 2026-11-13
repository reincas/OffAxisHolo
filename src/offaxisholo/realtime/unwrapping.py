"""
Phase unwrapping for realtime results (NOT part of the default hot path).

The realtime kernel delivers the wrapped phase. Unwrapping is optional:

    unwrap(phase, "fast2d")         Herraez et al. (scikit-image), CPU only. GPU
                                    data is copied to the host and back; in a
                                    service run it in a separate CPU thread.
    unwrap(phase, "least_squares")  DCT least squares (Ghiglia-Romero), runs on
                                    the GPU. NOT YET VALIDATED for DHM data
                                    (see docs/realtime, G1.8) - results differ
                                    from 'fast2d' (smoothing, step heights).
    TemporalUnwrapper               temporal unwrapping of a live stream,
                                    phi_t = phi_{t-1} + angle(U_t conj(U_{t-1})).
                                    Pixelwise and GPU trivial, but drifts with
                                    noise -> for tracking only.
"""

from .. import backend
from ..reconstruction import fieldops
from ..reconstruction.phaseUnwrapping import (
    phase_unwrapping_fast2d,
    phase_unwrapping_ls_dct,
)

METHODS = ("none", "fast2d", "least_squares")


def unwrap(phase_wrapped, method="fast2d", **kwargs):
    """Unwrap a 2D wrapped phase with ``method`` (see module docstring)."""
    method = method.lower()
    if method == "none":
        return phase_wrapped
    if method == "fast2d":
        return phase_unwrapping_fast2d(phase_wrapped)
    if method == "least_squares":
        return phase_unwrapping_ls_dct(phase_wrapped, **kwargs)
    raise ValueError(f"Unknown unwrapping method '{method}'. Possible: {METHODS}.")


class TemporalUnwrapper:
    """
    Temporal unwrapping of a sequence of complex fields of the same scene:
    the phase increment between consecutive fields (small for a slowly changing
    scene) is added up per pixel. Errors (noise, increments > pi) accumulate.
    """

    def __init__(self):
        self._previous = None
        self._phase = None

    def reset(self):
        self._previous = None
        self._phase = None

    def update(self, field):
        """Add the next complex field and return the accumulated phase."""
        xp = backend.get_namespace(field)
        if self._previous is None:
            self._phase = fieldops.phase(field)
        else:
            self._phase = self._phase + fieldops.phase(field * xp.conj(self._previous))
        self._previous = field
        return self._phase

    @property
    def phase(self):
        return self._phase
