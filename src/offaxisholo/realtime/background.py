"""
Background model of the realtime kernel (K7).

DATA layer: :class:`BackgroundModel` holds an averaged background with
everything needed to reuse it later or in another process: the background
field ``Ub``, the compensation factor ``K = conj(Ub) / (|Ub|^2 + eps)``, its
quality, an id, the creation time and the hash of the configuration parts it
depends on (``RealtimeConfig.background_hash``). Saving and loading live in
``offaxisholo.io.background`` (IO layer).

Lifetime (decision O6, 2026-11-16): a background never expires by itself. A
maximum age can be configured by the caller (``max_age_s``, default none); an
older background only produces a warning in the result. Limits follow after a
drift measurement (Masterplan V4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .result import to_jsonable


def utc_now() -> datetime:
    """Current time, timezone-aware UTC."""
    return datetime.now(timezone.utc)


def _parse_utc(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError(f"Timestamp {text!r} has no timezone.")
    return value


def model_id(field_array, config_hash: str, created_utc: str) -> str:
    """Short content id of a background (first 16 hex digits of a sha256).

    Parameters
    ----------
    field_array : numpy.ndarray
        Background field on the host.
    config_hash : str
        ``RealtimeConfig.background_hash`` of the configuration.
    created_utc : str
        Creation time, ISO 8601.
    """
    digest = hashlib.sha256()
    digest.update(config_hash.encode("utf-8"))
    digest.update(created_utc.encode("utf-8"))
    digest.update(np.ascontiguousarray(field_array).tobytes())
    return digest.hexdigest()[:16]


@dataclass
class BackgroundModel:
    """Averaged background, ready to compensate object fields.

    Parameters
    ----------
    field : numpy.ndarray
        Background field ``Ub`` on the crop grid, complex.
    compensation : numpy.ndarray
        Compensation factor ``K``, same shape.
    n_frames : int
        Number of averaged background frames.
    config_hash : str
        ``RealtimeConfig.background_hash`` of the configuration it was made with.
    created_utc : str, optional
        Creation time, UTC ISO 8601; set automatically when empty.
    quality : dict, optional
        ``OffAxisPipeline.background_quality`` at creation time.
    config : dict or None, optional
        Full configuration at creation time (for information).
    model_id : str, optional
        Content id; computed when empty.
    """

    field: np.ndarray
    compensation: np.ndarray
    n_frames: int
    config_hash: str
    created_utc: str = ""
    quality: dict[str, Any] = dc_field(default_factory=dict)
    config: dict | None = None
    model_id: str = ""

    def __post_init__(self):
        self.field = np.ascontiguousarray(self.field)
        self.compensation = np.ascontiguousarray(self.compensation)
        if self.field.ndim != 2 or self.field.shape != self.compensation.shape:
            raise ValueError(
                f"field {self.field.shape} and compensation "
                f"{self.compensation.shape} must be equal 2-D shapes."
            )
        if not np.iscomplexobj(self.field) or not np.iscomplexobj(self.compensation):
            raise ValueError("field and compensation must be complex.")
        self.n_frames = int(self.n_frames)
        # metadata in JSON form, so a model and its saved copy compare equal
        self.quality = to_jsonable(dict(self.quality))
        if self.config is not None:
            self.config = to_jsonable(dict(self.config))
        if not self.created_utc:
            self.created_utc = utc_now().isoformat(timespec="milliseconds")
        _parse_utc(self.created_utc)
        expected = model_id(self.field, self.config_hash, self.created_utc)
        if not self.model_id:
            self.model_id = expected
        elif self.model_id != expected:
            raise ValueError(
                f"Background id {self.model_id} does not match its content "
                f"({expected}); the data is corrupted or was changed."
            )

    def age_s(self, now: datetime | None = None) -> float:
        """Age in seconds at ``now`` (default: the current time)."""
        now = now or utc_now()
        return (now - _parse_utc(self.created_utc)).total_seconds()

    def is_valid_for(self, config) -> bool:
        """True if the model was made with an equivalent configuration."""
        return self.config_hash == config.background_hash()

    def check_config(self, config) -> None:
        """Raise ValueError if the model does not fit ``config``.

        Parameters
        ----------
        config : RealtimeConfig
            The active configuration.
        """
        if not self.is_valid_for(config):
            raise ValueError(
                f"Background {self.model_id} was made with another demodulation, "
                f"microscope, compensation or precision setting "
                f"(background hash {self.config_hash[:12]} != "
                f"{config.background_hash()[:12]}); record a new background."
            )

    def warnings(self, max_age_s: float | None = None, now=None) -> list[str]:
        """Warnings for the result (age above ``max_age_s``; O6).

        Parameters
        ----------
        max_age_s : float or None
            Maximum age in seconds; None disables the check (default).
        now : datetime, optional
            Reference time (default: now).
        """
        if max_age_s is None:
            return []
        age = self.age_s(now)
        if age > max_age_s:
            return [
                f"Background {self.model_id} is {age:.0f} s old "
                f"(maximum {max_age_s:.0f} s)."
            ]
        return []
