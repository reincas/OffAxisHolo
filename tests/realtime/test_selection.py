"""
Frame selection (K3): policies on metrics tables, min_frames, the field store
of the block mode and the selection in the result.
"""

import ast
import inspect
import json
import math
import textwrap

import numpy as np
import pytest
from synthetic_dhm import SyntheticDHM, make_config

from offaxisholo.realtime import (
    OffAxisPipeline,
    ReconstructionResult,
    SelectionPolicy,
    select_frames,
)
from offaxisholo.realtime.accumulators import FieldStore
from offaxisholo.realtime.selection import FORCED_PREFIX


def _table(**columns):
    """Metrics table as returned by OffAxisPipeline.frame_metrics()."""
    count = len(next(iter(columns.values())))
    return [
        (i, {name: float(values[i]) for name, values in columns.items()})
        for i in range(count)
    ]


NOISE = [0.010, 0.011, 0.009, 0.050, 0.012, 0.010]
FRACTION = [0.16, 0.15, 0.17, 0.16, 0.05, 0.165]
SATURATED = [0, 0, 0, 0, 0, 900]


@pytest.fixture
def table():
    return _table(
        phase_noise_rad=NOISE, sideband_fraction=FRACTION, saturated_count=SATURATED
    )


# ---------------------------------------------------------------- policies
def test_policy_all_uses_every_frame(table):
    sel = select_frames(table)
    assert sel.mask.tolist() == [True] * 6
    assert sel.reasons == [None] * 6
    assert sel.policy == "all"
    assert sel.min_frames_enforced is False
    assert sel.quality() == {
        "selection_policy": "all",
        "n_frames_total": 6,
        "n_frames_used": 6,
        "selection_min_frames": 1,
        "selection_min_frames_enforced": False,
    }


def test_threshold_policy(table):
    policy = SelectionPolicy(
        policy="threshold",
        thresholds={"saturated_count": (None, 100), "sideband_fraction": (0.1, None)},
    )
    sel = select_frames(table, policy)
    assert sel.mask.tolist() == [True, True, True, True, False, False]
    assert sel.reasons[4] == "sideband_fraction 0.05 < 0.1"
    assert sel.reasons[5] == "saturated_count 900 > 100"


def test_threshold_on_missing_metric_rejects(table):
    rows = _table(phase_noise_rad=[0.01, math.nan, 0.01])
    policy = SelectionPolicy(
        policy="threshold", thresholds={"phase_noise_rad": (None, 0.1)}
    )
    sel = select_frames(rows, policy)
    assert sel.mask.tolist() == [True, False, True]
    assert sel.reasons[1] == "phase_noise_rad: not available"


def test_robust_policy_rejects_outliers(table):
    sel = select_frames(table, SelectionPolicy(policy="robust"))
    # noise outlier (frame 3) and fringe-contrast outlier (frame 4)
    assert sel.mask.tolist() == [True, True, True, False, False, True]
    assert sel.reasons[3].startswith("phase_noise_rad 0.05 outlier")
    assert sel.reasons[4].startswith("sideband_fraction 0.05 outlier")


def test_robust_policy_skips_constant_and_missing_values():
    rows = _table(
        saturated_count=[0, 0, 0, 900],  # MAD = 0: not tested
        phase_noise_rad=[math.nan] * 4,  # no reference region
    )
    policy = SelectionPolicy(
        policy="robust", robust_metrics=("saturated_count", "phase_noise_rad")
    )
    assert select_frames(rows, policy).mask.all()


def test_best_k_policy(table):
    sel = select_frames(table, SelectionPolicy(policy="best_k", k=3))
    # lowest phase noise: frames 2 (0.009), 0 and 5 (0.010, tie by index)
    assert sel.mask.tolist() == [True, False, True, False, False, True]
    assert sel.reasons[1] == "not among the best 3 by phase_noise_rad"


def test_best_k_descending_and_nan_last():
    rows = _table(sideband_fraction=[0.1, math.nan, 0.3, 0.2])
    policy = SelectionPolicy(
        policy="best_k", k=2, rank_by="sideband_fraction", ascending=False
    )
    assert select_frames(rows, policy).mask.tolist() == [False, False, True, True]


def test_min_frames_is_enforced_and_reported(table):
    policy = SelectionPolicy(
        policy="threshold", thresholds={"phase_noise_rad": (None, 0.0)}, min_frames=2
    )
    sel = select_frames(table, policy)
    assert sel.n_used == 2
    # best by phase noise: frame 2 (0.009), then frame 0 (0.010, tie with 5)
    assert sel.mask.tolist() == [True, False, True, False, False, False]
    assert sel.min_frames_enforced is True
    assert sel.reasons[2].startswith(FORCED_PREFIX + "phase_noise_rad 0.009 > 0")
    assert sel.quality()["selection_min_frames_enforced"] is True


def test_min_frames_larger_than_the_measurement(table):
    sel = select_frames(table[:2], SelectionPolicy(min_frames=5))
    assert sel.n_used == 2
    assert sel.min_frames_enforced is False
    assert sel.quality()["selection_min_frames"] == 5


@pytest.mark.parametrize(
    "policy, message",
    [
        (SelectionPolicy(policy="median"), "Unknown policy"),
        (SelectionPolicy(min_frames=0), "min_frames"),
        (SelectionPolicy(policy="best_k"), "best_k"),
        (SelectionPolicy(policy="threshold"), "at least one threshold"),
        (SelectionPolicy(robust_k=0), "robust_k"),
    ],
)
def test_invalid_policies(table, policy, message):
    with pytest.raises(ValueError, match=message):
        select_frames(table, policy)


def test_unknown_metric_and_empty_table(table):
    with pytest.raises(ValueError, match="Unknown metric"):
        select_frames(table, SelectionPolicy(policy="best_k", k=1, rank_by="focus"))
    with pytest.raises(ValueError, match="No frames"):
        select_frames([])


def test_policy_dict_round_trip():
    policy = SelectionPolicy(
        policy="threshold", thresholds={"saturated_count": (None, 10)}, min_frames=4
    )
    text = json.dumps(policy.to_dict())
    assert SelectionPolicy.from_dict(json.loads(text)) == policy
    with pytest.raises(ValueError, match="Unknown keys"):
        SelectionPolicy.from_dict({"policy": "all", "limit": 3})


# ------------------------------------------------------------ pipeline
def _pipeline(keep_fields=True, n_obj=6, mode="block"):
    config = make_config(
        demod={"crop": 64, "radius": 20},
        compensation={"ref_border": 8},
        averaging={"n_obj": n_obj, "mode": mode},
        metrics={"enabled": True, "saturation_level": 3.5},
    )
    pipe = OffAxisPipeline(config, keep_fields=keep_fields)
    pipe.set_background(SyntheticDHM().frames(4, obj=False))
    return pipe


def _frames():
    good = list(SyntheticDHM(seed=5).frames(5, piston_std=0.3, noise=0.02))
    bad = SyntheticDHM(carrier=(34, 31)).frame()  # carrier drifted
    return good[:3] + [bad] + good[3:]


def test_apply_selection_averages_only_the_used_frames():
    frames = _frames()
    mask = [True, False, True, False, True, True]
    pipe = _pipeline()
    for frame in frames:
        pipe.push(frame)
    pipe.apply_selection(mask)
    assert pipe.n_frames == 4

    reference = _pipeline(keep_fields=False)
    for frame, used in zip(frames, mask, strict=True):
        if used:
            reference.push(frame)
    np.testing.assert_allclose(
        pipe.result().field, reference.result().field, rtol=1e-12, atol=1e-14
    )

    pipe.apply_selection([True] * 6)  # stored fields are unchanged
    assert pipe.n_frames == 6


def test_robust_selection_removes_a_drifted_frame():
    frames = _frames()
    pipe = _pipeline()
    for frame in frames:
        pipe.push(frame)
    sel = select_frames(pipe.frame_metrics(), SelectionPolicy(policy="robust"))
    assert sel.mask.tolist() == [True, True, True, False, True, True]
    pipe.apply_selection(sel.mask)
    res = ReconstructionResult.from_realtime(pipe.result(), pipe, selection=sel)

    assert [r.used for r in res.frames] == sel.mask.tolist()
    assert res.frames[3].reject_reason == sel.reasons[3]
    assert res.n_frames_used == 5
    assert res.quality["n_obj"] == 5
    assert res.quality["selection_policy"] == "robust"
    assert res.quality["n_frames_total"] == 6

    header, buffers = res.to_message()
    same = ReconstructionResult.from_message(json.loads(json.dumps(header)), buffers)
    assert [r.used for r in same.frames] == sel.mask.tolist()
    assert same.quality == res.quality


def test_selection_needs_block_mode_and_stored_fields():
    with pytest.raises(ValueError, match="block"):
        _pipeline(mode="sliding")
    pipe = _pipeline(keep_fields=False)
    pipe.push(SyntheticDHM().frame())
    with pytest.raises(RuntimeError, match="keep_fields"):
        pipe.apply_selection([True])


def test_apply_selection_checks_the_mask():
    pipe = _pipeline()
    for frame in _frames()[:2]:
        pipe.push(frame)
    with pytest.raises(ValueError, match="2 frames"):
        pipe.apply_selection([True])
    with pytest.raises(ValueError, match="at least one"):
        pipe.apply_selection([False, False])


def test_field_store_capacity_and_reset():
    pipe = _pipeline(n_obj=2)
    frames = _frames()
    pipe.push(frames[0])
    pipe.push(frames[1])
    with pytest.raises(RuntimeError, match="n_obj"):
        pipe.push(frames[2])
    pipe.reset()
    pipe.push(frames[2])
    pipe.apply_selection([True])
    assert pipe.n_frames == 1


def test_field_store_add_has_no_host_synchronisation():
    tree = ast.parse(textwrap.dedent(inspect.getsource(FieldStore.add)))
    names = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not names & {"float", "int", "bool", "item", "get", "to_numpy", "nonzero"}
