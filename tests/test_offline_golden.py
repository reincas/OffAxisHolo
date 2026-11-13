"""
The offline reconstruction library must keep its results (golden reference).

The reference was generated with tests/golden/make_golden_offline.py before the
realtime refactorings. Tolerances are at round-off level, so that the test is
portable between machines (FFT/BLAS builds) but detects any real change.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).with_name("golden")))
from make_golden_offline import GOLDEN_FILE, compute  # noqa: E402


@pytest.fixture(scope="module")
def results():
    logging.disable(logging.INFO)
    try:
        yield compute(), np.load(GOLDEN_FILE)
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize(
    "key",
    [
        "field_compensated",
        "field_propagated",
        "phase_map",
        "height_profile",
        "intensity_db",
        "single_phase_map",
        "asm",
        "asm_padding",
    ],
)
def test_offline_results_unchanged(results, key):
    new, golden = results
    expected = golden[key]
    assert new[key].dtype == expected.dtype
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(new[key], expected, rtol=1e-12, atol=1e-12 * scale)


def test_order_and_radius_unchanged(results):
    new, golden = results
    assert tuple(int(v) for v in new["order"]) == tuple(golden["order"])
    assert float(new["radius"]) == float(golden["radius"])
