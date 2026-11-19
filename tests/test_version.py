"""Version and source revision in the result metadata (K15)."""

from __future__ import annotations

import offaxisholo
from offaxisholo import realtime


def test_version_and_git_revision():
    assert isinstance(offaxisholo.__version__, str) and offaxisholo.__version__
    revision = offaxisholo.git_revision()
    assert revision is None or (len(revision.split("-")[0]) >= 7)
    assert offaxisholo.git_revision() is revision or revision is None  # cached


def test_default_versions_carry_the_revision():
    versions = realtime.default_versions()
    assert versions["offaxisholo"] == offaxisholo.__version__
    assert versions["offaxisholo_git"] == offaxisholo.git_revision()
    assert all(v is None or isinstance(v, str) for v in versions.values())


def test_public_entry_points_are_exported():
    for name in (
        "reconstruct_frames",
        "ReconstructionSession",
        "ReconstructionResult",
        "FrameRecord",
        "RealtimeConfig",
        "calibrate_order",
        "OrderCalibration",
        "BackgroundModel",
        "SelectionPolicy",
        "reference_mask",
        "reference_quality",
        "default_versions",
    ):
        assert name in realtime.__all__
        assert getattr(realtime, name) is not None
