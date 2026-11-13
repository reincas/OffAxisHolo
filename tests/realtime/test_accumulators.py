"""Block / sliding / EMA accumulators against naive references (G1.5)."""

import numpy as np
import pytest

from offaxisholo import backend
from offaxisholo.realtime import BlockAccumulator, EMAAccumulator, SlidingAccumulator

XP = backend.namespace("numpy")


def _fields(count, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))).astype(
            np.complex64
        )
        for _ in range(count)
    ]


def test_block_mean_and_reset():
    acc = BlockAccumulator(XP)
    fields = _fields(5)
    for f in fields:
        acc.add(f)
    np.testing.assert_allclose(acc.mean(), np.mean(fields, axis=0), rtol=1e-6)
    assert acc.n_effective == 5
    acc.reset()
    with pytest.raises(RuntimeError):
        acc.mean()
    acc.add(fields[0])
    np.testing.assert_array_equal(acc.mean(), fields[0])


@pytest.mark.parametrize("renorm_every", [1, 3, 1000])
def test_sliding_equals_mean_of_last_window(renorm_every):
    acc = SlidingAccumulator(XP, window=4, renorm_every=renorm_every)
    fields = _fields(11, seed=1)
    for i, f in enumerate(fields):
        acc.add(f)
        last = fields[max(0, i - 3) : i + 1]
        np.testing.assert_allclose(acc.mean(), np.mean(last, axis=0), atol=1e-5)
        assert acc.n_effective == len(last)


def test_sliding_renorm_limits_drift():
    rng = np.random.default_rng(2)
    acc = SlidingAccumulator(XP, window=8, renorm_every=50)
    big = None
    for _ in range(2000):
        big = (1e3 * rng.standard_normal((4, 4))).astype(np.complex64)
        acc.add(big)
    # the running sum is re-summed regularly -> equals the ring buffer sum
    np.testing.assert_allclose(
        acc._sum, np.sum(acc._ring, axis=0), rtol=1e-3, atol=1e-2
    )


def test_ema_recursion_and_effective_number():
    alpha = 0.2
    acc = EMAAccumulator(XP, alpha)
    fields = _fields(30, seed=3)
    state = None
    for f in fields:
        acc.add(f)
        state = f.copy() if state is None else state + alpha * (f - state)
        np.testing.assert_allclose(acc.mean(), state, rtol=1e-5, atol=1e-6)
    assert acc.n_effective == pytest.approx((2 - alpha) / alpha)


def test_ema_noise_reduction_matches_effective_number():
    rng = np.random.default_rng(4)
    alpha = 0.1
    acc = EMAAccumulator(XP, alpha)
    for _ in range(400):
        acc.add(rng.standard_normal((64, 64)).astype(np.complex64))
    variance = np.var(acc.mean().real)
    assert variance == pytest.approx(1 / acc.n_effective, rel=0.15)
