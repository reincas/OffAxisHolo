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

    def reset(self):
        self.count = 0
        self._index = 0
        self._since_renorm = 0
        if self._ring is not None:
            self._ring[...] = 0
            self._sum[...] = 0

    def add(self, field):
        if self._ring is None:
            self._ring = self.xp.zeros((self.window,) + field.shape, dtype=field.dtype)
            self._sum = self.xp.zeros_like(field)
        self._sum += field - self._ring[self._index]
        self._ring[self._index] = field
        self._index = (self._index + 1) % self.window
        self.count += 1
        self._since_renorm += 1
        if self._since_renorm >= self.renorm_every:
            self._sum[...] = self.xp.sum(self._ring, axis=0)
            self._since_renorm = 0

    @property
    def n_effective(self) -> float:
        return float(min(self.count, self.window))

    def mean(self):
        if self.count == 0:
            raise RuntimeError("No frames accumulated.")
        return self._sum / min(self.count, self.window)


class EMAAccumulator:
    def __init__(self, xp, alpha):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.xp = xp
        self.alpha = float(alpha)
        self._state = None
        self.count = 0

    def reset(self):
        self.count = 0

    def add(self, field):
        if self._state is None:
            self._state = self.xp.zeros_like(field)
        if self.count == 0:
            self._state[...] = field
        else:
            self._state += self.alpha * (field - self._state)
        self.count += 1

    @property
    def n_effective(self) -> float:
        """Effective number of averaged frames in the steady state."""
        if self.count == 0:
            return 0.0
        return min(float(self.count), (2.0 - self.alpha) / self.alpha)

    def mean(self):
        if self.count == 0:
            raise RuntimeError("No frames accumulated.")
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
