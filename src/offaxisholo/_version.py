"""Package version and source revision for result metadata (K15, old G2).

Lightweight: standard library only. ``git_revision`` runs ``git`` at most once
per process (cached) and returns None when the package is not inside a git
working copy (e.g. a tagged release installed from GitHub on a lab PC) or git
is missing.
"""

from __future__ import annotations

import functools
import subprocess
from importlib import metadata
from pathlib import Path

__all__ = ["__version__", "git_revision"]

try:
    __version__ = metadata.version("offaxisholo")
except metadata.PackageNotFoundError:  # source tree without installation
    __version__ = "0.0.0+unknown"


@functools.lru_cache(maxsize=1)
def git_revision() -> str | None:
    """Short git hash of the source tree, with ``-dirty`` for local changes.

    Returns
    -------
    str or None
        e.g. ``"0c4a3b8"`` or ``"0c4a3b8-dirty"``; None outside a git working
        copy, without git, or when git does not answer within 5 s.
    """
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7", "--exclude=*"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    revision = out.stdout.strip()
    return revision or None
