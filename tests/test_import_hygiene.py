"""
Lightweight imports: the package and the modules needed by the realtime kernel
must not load GUI, plotting, image IO or container libraries. Each check runs in
a fresh interpreter, because sys.modules of the test process is polluted.
"""

import subprocess
import sys
import textwrap

import pytest

HEAVY = ("tkinter", "matplotlib", "skimage", "cv2", "scidatacontainer")

LIGHT_MODULES = [
    "offaxisholo",
    "offaxisholo.backend",
    "offaxisholo.reconstruction",
    "offaxisholo.reconstruction.numericalPropagation",
    "offaxisholo.reconstruction.phaseUnwrapping",
]


def _loaded_heavy_modules(statement):
    code = textwrap.dedent(
        f"""
        import sys
        {statement}
        heavy = {HEAVY!r}
        print(",".join(m for m in heavy if m in sys.modules))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return [m for m in out.stdout.strip().split(",") if m]


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_import(module):
    assert _loaded_heavy_modules(f"import {module}") == []


def test_public_api_still_available():
    """The lazy imports must keep the old public API working."""
    statement = (
        "from offaxisholo import DataLoader, Hologram, HologramProcessor, "
        "ReconstructionPipeline; "
        "from offaxisholo.reconstruction import ReferenceHologram; "
        "import offaxisholo; assert set(offaxisholo.__all__) <= set(dir(offaxisholo))"
    )
    # heavy modules are expected here, the import itself must succeed
    _loaded_heavy_modules(statement)
