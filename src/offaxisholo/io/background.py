"""
Saving and loading of background models (K7), IO layer.

Format: one NumPy ``.npz`` file with the arrays ``field`` and ``compensation``
and a JSON text ``meta`` (format version, number of frames, background hash,
creation time, id, quality, configuration). It is written and read without
pickle (``allow_pickle=False``), so a file from another process cannot run
code. The id is recomputed on loading; a changed or damaged file is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..realtime.background import BackgroundModel
from ..realtime.result import to_jsonable

FORMAT = "offaxisholo.background"
FORMAT_VERSION = 1


def save_background(model: BackgroundModel, path) -> Path:
    """Write a background model to ``path`` (``.npz``).

    Parameters
    ----------
    model : BackgroundModel
        The model to save.
    path : str or pathlib.Path
        Target file; the suffix ``.npz`` is added when missing.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    meta = to_jsonable(
        {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "n_frames": model.n_frames,
            "config_hash": model.config_hash,
            "created_utc": model.created_utc,
            "model_id": model.model_id,
            "quality": model.quality,
            "config": model.config,
        }
    )
    with path.open("wb") as handle:
        np.savez(
            handle,
            field=model.field,
            compensation=model.compensation,
            meta=np.array(json.dumps(meta, allow_nan=False)),
        )
    return path


def load_background(path) -> BackgroundModel:
    """Read a background model written by :func:`save_background`.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.npz`` file.

    Returns
    -------
    BackgroundModel

    Raises
    ------
    ValueError
        Not a background file, unsupported format version, or an id that does
        not match the content.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        try:
            meta = json.loads(str(data["meta"]))
            field = data["field"]
            compensation = data["compensation"]
        except KeyError as err:
            raise ValueError(f"{path} is not a background file ({err}).") from None
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not a background file.")
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported background format version {meta.get('format_version')}."
        )
    return BackgroundModel(
        field=field,
        compensation=compensation,
        n_frames=meta["n_frames"],
        config_hash=meta["config_hash"],
        created_utc=meta["created_utc"],
        quality=meta.get("quality") or {},
        config=meta.get("config"),
        model_id=meta["model_id"],
    )
