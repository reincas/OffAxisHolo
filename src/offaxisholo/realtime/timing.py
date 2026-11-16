"""
Stage timers of the realtime kernel (K5).

PROCESSING helper without device access of its own: the timer reads the host
clock (``time.perf_counter``). By default it never synchronises the device, so
it is safe in the hot path. On the GPU the stages then measure the time to
*enqueue* the kernels, not their run time. With ``profile=True`` the timer
synchronises the device before every clock reading and measures the real run
time of each stage, at the price of one synchronisation per stage.

Stages (cumulative seconds over all frames of one measurement):

    upload        host frame -> device array               per frame
    demodulation  window, FFT, sideband crop, mask, IFFT    per frame
    piston        piston phase correction                   per frame
    averaging     accumulation of the complex field         per frame
    metrics       per-frame metrics (only when enabled, K2) per frame
    compensation  mean(U) * K                               per result
    propagation   angular spectrum propagation (all z)      per result
    download      device -> host copy of the outputs        per result

``total`` is the wall-clock time from the first timed step after ``reset``
(the first ``push``) to the last one (usually the download). It includes the
time the caller spends between frames (camera exposure, grabbing), so the sum
of the stages never exceeds it.
"""

from __future__ import annotations

import time

STAGES = (
    "upload",
    "demodulation",
    "piston",
    "averaging",
    "metrics",
    "compensation",
    "propagation",
    "download",
)


def _synchronize_function(xp):
    """Return a callable that waits for the device of ``xp`` (no-op for NumPy)."""
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return None
    return cuda.get_current_stream().synchronize


class StageTimer:
    """Cumulative stage timer.

    Parameters
    ----------
    xp : module
        Array namespace of the pipeline (``numpy`` or ``cupy``).
    profile : bool, optional
        Synchronise the device before every clock reading (exact GPU stage
        times, slower). Without effect on the CPU.
    clock : callable, optional
        Clock in seconds; ``time.perf_counter`` by default (replaceable in
        tests).

    Examples
    --------
    >>> timer = StageTimer(numpy)
    >>> t = timer.start()
    >>> ...                        # upload
    >>> t = timer.lap("upload", t)
    >>> timer.timings()["upload"]
    """

    def __init__(self, xp, profile: bool = False, clock=time.perf_counter):
        self.profile = bool(profile)
        self._clock = clock
        self._synchronize = _synchronize_function(xp) if self.profile else None
        self.reset()

    def reset(self):
        """Clear all stages and the total (start of a new measurement)."""
        self._seconds = dict.fromkeys(STAGES, 0.0)
        self._begin = None
        self._last = None

    def _now(self) -> float:
        if self._synchronize is not None:
            self._synchronize()
        return self._clock()

    def start(self) -> float:
        """Read the clock at the beginning of a timed section.

        Returns
        -------
        float
            The clock value to pass to :meth:`lap`.
        """
        now = self._now()
        if self._begin is None:
            self._begin = now
        return now

    def lap(self, stage: str, since: float) -> float:
        """Add the time since ``since`` to ``stage`` and return the clock value.

        Parameters
        ----------
        stage : str
            One of :data:`STAGES`.
        since : float
            Clock value of :meth:`start` or of the previous :meth:`lap`.

        Returns
        -------
        float
            The current clock value (start of the next stage).
        """
        now = self._now()
        self._seconds[stage] += now - since
        self._last = now
        return now

    def timings(self) -> dict[str, float]:
        """Cumulative stage times and the total in seconds.

        Returns
        -------
        dict
            One entry per stage and ``total`` (0.0 before the first timed
            step). Whether the values are device synchronised is
            :attr:`profile`.
        """
        out = dict(self._seconds)
        if self._begin is None or self._last is None:
            out["total"] = 0.0
        else:
            out["total"] = self._last - self._begin
        return out
