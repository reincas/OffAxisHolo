"""
Averaging of complex fields on the device (no knowledge of the physics).

All accumulators keep fixed-size device buffers, allocated on the first
``add``; the frame counters are host integers (no synchronisation). The mean of
complex (piston corrected) fields is averaged - never phases.

    block   : sum of all frames since reset, mean = sum / count
    sliding : ring buffer of the last ``window`` frames, running sum
              S += new - oldest, re-summed every ``renorm_every`` frames
              against numerical drift
    ema     : S <- S + alpha * (U - S), effective number of frames
              N_eff = (2 - alpha) / alpha

Frame gating (K17): the sliding and EMA accumulators accept a 0/1 weight per
frame as a 0-d device array of the field's real dtype (``add(field,
weight)``). The weight is computed on the device from fixed thresholds of the
frame metrics, so no host decision is needed in the hot path:

    sliding : the ring holds w * U and w; mean = sum(w U) / sum(w) over the
              window; n_effective = sum(w) in the window
    ema     : S <- S + w * a * (U - S) with a = 1 until the first accepted
              frame and a = alpha afterwards; n_effective =
              min(accepted frames, (2 - alpha) / alpha)

A rejected frame (w = 0) leaves the EMA unchanged and occupies its slot of the
sliding window without contributing. Without a weight the accumulators run
exactly the ungated code. Reading ``n_effective`` or ``mean`` of a gated
accumulator synchronises once (result time only).
"""


class BlockAccumulator:
    def __init__(self, xp):
        self.xp = xp
        self._sum = None
        self.count = 0

    def reset(self):
        self.count = 0
        if self._sum is not None:
            self._sum[...] = 0

    def add(self, field):
        if self._sum is None:
            self._sum = self.xp.zeros_like(field)
        self._sum += field
        self.count += 1

    @property
    def n_effective(self) -> float:
        return float(self.count)

    def mean(self):
        if self.count == 0:
            raise RuntimeError("No frames accumulated.")
        return self._sum / self.count


class SlidingAccumulator:
    def __init__(self, xp, window, renorm_every=1000):
        if window < 1:
            raise ValueError("window must be >= 1.")
        self.xp = xp
        self.window = int(window)
        self.renorm_every = int(renorm_every)
        self._ring = None
        self._sum = None
        self._index = 0
        self.count = 0  # frames added since reset
        self._since_renorm = 0
        self._weights = None  # ring of the gate weights (K17), only when gated
        self._weight_sum = None

    @property
    def gated(self) -> bool:
        """True once a frame was added with a gate weight (K17)."""
        return self._weights is not None

    def reset(self):
        self.count = 0
        self._index = 0
        self._since_renorm = 0
        if self._ring is not None:
            self._ring[...] = 0
            self._sum[...] = 0
        if self._weights is not None:
            self._weights[...] = 0
            self._weight_sum[...] = 0

    def add(self, field, weight=None):
        """Add one field (hot path, no host synchronisation).

        Parameters
        ----------
        field : array
            Complex field.
        weight : 0-d array, optional
            Gate weight 0 or 1 (K17) of the field's real dtype.
        """
        if self._ring is None:
            self._ring = self.xp.zeros((self.window,) + field.shape, dtype=field.dtype)
            self._sum = self.xp.zeros_like(field)
        if weight is not None:
            if self._weights is None:
                self._weights = self.xp.zeros((self.window,), dtype=weight.dtype)
                self._weight_sum = self.xp.zeros((), dtype=weight.dtype)
            field = field * weight
            self._weight_sum += weight - self._weights[self._index]
            self._weights[self._index] = weight
        self._sum += field - self._ring[self._index]
        self._ring[self._index] = field
        self._index = (self._index + 1) % self.window
        self.count += 1
        self._since_renorm += 1
        if self._since_renorm >= self.renorm_every:
            self._sum[...] = self.xp.sum(self._ring, axis=0)
            if self._weights is not None:
                self._weight_sum[...] = self.xp.sum(self._weights)
            self._since_renorm = 0

    @property
    def n_effective(self) -> float:
        """Frames in the current average (gated: accepted ones; synchronises)."""
        if self._weights is not None:
            return float(self._weight_sum)
        return float(min(self.count, self.window))

    def mean(self):
        if self.count == 0:
            raise RuntimeError("No frames accumulated.")
        if self._weights is not None:
            if self.n_effective < 0.5:
                raise RuntimeError("No frame in the window passed the gate.")
            return self._sum / self._weight_sum
        return self._sum / min(self.count, self.window)


class EMAAccumulator:
    def __init__(self, xp, alpha):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.xp = xp
        self.alpha = float(alpha)
        self._state = None
        self.count = 0
        self._started = None  # gated (K17): 1 after the first accepted frame
        self._accepted = None  # gated: number of accepted frames

    @property
    def gated(self) -> bool:
        """True once a frame was added with a gate weight (K17)."""
        return self._started is not None

    def reset(self):
        self.count = 0
        if self._started is not None:
            self._started[...] = 0
            self._accepted[...] = 0

    def add(self, field, weight=None):
        """Add one field (hot path, no host synchronisation).

        Parameters
        ----------
        field : array
            Complex field.
        weight : 0-d array, optional
            Gate weight 0 or 1 (K17) of the field's real dtype.
        """
        if self._state is None:
            self._state = self.xp.zeros_like(field)
        if weight is not None:
            if self._started is None:
                self._started = self.xp.zeros((), dtype=weight.dtype)
                self._accepted = self.xp.zeros((), dtype=weight.dtype)
            # step 1 until the first accepted frame (copies it), alpha after it
            step = weight * (self._started * (self.alpha - 1.0) + 1.0)
            self._state += step * (field - self._state)
            self._started[...] = self.xp.maximum(self._started, weight)
            self._accepted += weight
        elif self.count == 0:
            self._state[...] = field
        else:
            self._state += self.alpha * (field - self._state)
        self.count += 1

    @property
    def n_effective(self) -> float:
        """Effective number of averaged frames in the steady state.

        Gated: ``min(accepted frames, (2 - alpha) / alpha)`` (synchronises).
        """
        if self.count == 0:
            return 0.0
        count = float(self._accepted) if self._started is not None else self.count
        return min(float(count), (2.0 - self.alpha) / self.alpha)

    def mean(self):
        if self.count == 0:
            raise RuntimeError("No frames accumulated.")
        if self._started is not None and float(self._accepted) < 0.5:
            raise RuntimeError("No frame passed the gate.")
        return self._state


def make_accumulator(xp, averaging):
    """Accumulator for an AveragingConfig."""
    if averaging.mode == "block":
        return BlockAccumulator(xp)
    if averaging.mode == "sliding":
        return SlidingAccumulator(xp, averaging.window, averaging.renorm_every)
    if averaging.mode == "ema":
        return EMAAccumulator(xp, averaging.ema_alpha)
    raise ValueError(f"Unknown averaging mode '{averaging.mode}'.")


class FieldStore:
    """Fixed-size device stack of the fields of one block measurement.

    Keeps the piston-corrected fields until the frame selection (K3), so the
    block average can be rebuilt from the used frames only. The buffer of
    ``capacity`` fields is allocated on the first ``add`` and reused after
    ``reset``.

    Parameters
    ----------
    xp : module
        Array namespace.
    capacity : int
        Maximum number of fields (``AveragingConfig.n_obj``).
    """

    def __init__(self, xp, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1.")
        self.xp = xp
        self.capacity = int(capacity)
        self._stack = None
        self.count = 0

    def reset(self):
        """Forget the stored fields (the buffer is kept)."""
        self.count = 0

    def add(self, field):
        """Store the next field (hot path: one device copy, no synchronisation)."""
        if self.count >= self.capacity:
            raise RuntimeError(
                "More frames than the field store holds (averaging.n_obj); "
                "increase n_obj or reset between measurements."
            )
        if self._stack is None:
            self._stack = self.xp.empty(
                (self.capacity,) + field.shape, dtype=field.dtype
            )
        self._stack[self.count] = field
        self.count += 1

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(f"Field {index} not stored (count {self.count}).")
        return self._stack[index]
