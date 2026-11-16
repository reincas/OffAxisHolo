"""
Product result of the realtime kernel: ``ReconstructionResult`` and ``FrameRecord``.

DATA layer: plain host data (NumPy arrays, JSON-able dicts), no device arrays,
no file IO, no network. A result is built once per measurement from a
``RealtimeResult`` of ``OffAxisPipeline`` (``ReconstructionResult.from_realtime``);
this is the only place where the device arrays are copied to the host.

Message form (decision G2: the shared contract will be defined in DHMConnect,
``dhmconnect/schemas/``; until then this is draft ``schema_version`` 0.x):

    header, buffers = result.to_message()
    # header: JSON-able dict, buffers: list of (name, C-contiguous little-endian
    # ndarray); header["buffers"] describes every buffer (name, dtype, shape,
    # byte_order) in the same order.
    same = ReconstructionResult.from_message(header, buffers)

Conventions of the shared contracts (docs/system/SOFTWARE_PLAN_OVERVIEW.md §4):
units in the names (``_um``, ``_rad``, ``_s``), float32 / complex64, image axis
0 = y and axis 1 = x, timestamps in UTC ISO 8601, ``schema_version`` in every
message, never pickle. The kernel configuration keeps metres; the conversion to
micrometres happens here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from typing import Any

import numpy as np

SCHEMA_VERSION = "0.1"
MESSAGE_TYPE = "ReconstructionResult"

# name of the buffer -> dtype in the message
_BUFFER_DTYPES = {
    "phase_rad": np.dtype("<f4"),
    "intensity": np.dtype("<f4"),
    "field": np.dtype("<c8"),
}
_DTYPE_NAMES = {np.dtype("<f4"): "float32", np.dtype("<c8"): "complex64"}


def _jsonable(value):
    """Convert NumPy scalars, tuples and non-finite floats into plain JSON values.

    Parameters
    ----------
    value : object
        Nested structure of dicts, lists, tuples, scalars and None.

    Returns
    -------
    object
        The same structure with ``dict``, ``list``, ``str``, ``int``, ``float``,
        ``bool`` and ``None`` only. NaN and infinities become ``None``.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-able.")


def _major(version: str) -> str:
    return str(version).split(".", 1)[0]


@dataclass
class FrameRecord:
    """Record of one object frame of a measurement.

    Parameters
    ----------
    index : int
        Position of the frame in the measurement (0-based).
    used : bool, optional
        Whether the frame entered the average (frame selection, K3).
    reject_reason : str or None, optional
        Why the frame was rejected; None for used frames.
    metrics : dict, optional
        Per-frame metrics (K2), e.g. ``saturated_count`` or ``piston_rad``;
        values are JSON-able scalars or None.
    acquisition : dict, optional
        Values supplied by the caller, passed through unchanged (exposure,
        camera and host time, frame id; K5).
    """

    index: int
    used: bool = True
    reject_reason: str | None = None
    metrics: dict[str, Any] = dc_field(default_factory=dict)
    acquisition: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the record as a JSON-able dict."""
        return _jsonable(
            {
                "index": int(self.index),
                "used": bool(self.used),
                "reject_reason": self.reject_reason,
                "metrics": self.metrics,
                "acquisition": self.acquisition,
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> FrameRecord:
        """Build a record from the dict of :meth:`to_dict`."""
        return cls(
            index=int(data["index"]),
            used=bool(data.get("used", True)),
            reject_reason=data.get("reject_reason"),
            metrics=dict(data.get("metrics") or {}),
            acquisition=dict(data.get("acquisition") or {}),
        )


@dataclass
class ReconstructionResult:
    """Reconstructed phase and intensity of one measurement with its metadata.

    Parameters
    ----------
    phase_rad : numpy.ndarray
        Wrapped phase in rad, shape (n, n), stored as float32.
    intensity : numpy.ndarray
        Linear intensity ``|E|^2`` of the compensated field (relative to the
        background), shape (n, n), stored as float32.
    pixel_pitch_um : tuple of float
        Pixel pitch (dx along axis 1, dy along axis 0) of the result grid in µm,
        i.e. after the sideband crop (``dx' = dx * N / n``).
    z_um : float
        Propagation distance in µm (0 = hologram plane).
    wavelength_um : float
        Vacuum wavelength in µm.
    n_medium : float
        Refractive index of the propagation medium.
    config_hash : str
        sha256 of the kernel configuration (``RealtimeConfig.config_hash``).
    field : numpy.ndarray or None, optional
        Compensated (and propagated) complex field, complex64; only when
        requested, because it doubles the message size.
    frames : list of FrameRecord, optional
        One record per object frame (filled by the entry points, K6).
    quality : dict, optional
        Quality numbers, e.g. ``n_obj``, ``n_bg``, ``piston_std_rad`` (K8 adds
        more).
    timings_s : dict, optional
        Stage timings in seconds (K5).
    versions : dict, optional
        Package versions used to compute the result.
    config : dict or None, optional
        The full kernel configuration (``RealtimeConfig.to_dict``).
    created_utc : str, optional
        Creation time, UTC ISO 8601; set automatically when empty.
    schema_version : str, optional
        Version of the message schema.
    """

    phase_rad: np.ndarray
    intensity: np.ndarray
    pixel_pitch_um: tuple[float, float]
    z_um: float
    wavelength_um: float
    n_medium: float
    config_hash: str
    field: np.ndarray | None = None
    frames: list[FrameRecord] = dc_field(default_factory=list)
    quality: dict[str, Any] = dc_field(default_factory=dict)
    timings_s: dict[str, float] = dc_field(default_factory=dict)
    versions: dict[str, Any] = dc_field(default_factory=dict)
    config: dict | None = None
    created_utc: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        self.phase_rad = np.ascontiguousarray(self.phase_rad, dtype=np.float32)
        self.intensity = np.ascontiguousarray(self.intensity, dtype=np.float32)
        if self.phase_rad.ndim != 2:
            raise ValueError(
                f"phase_rad must be 2-D, got shape {self.phase_rad.shape}."
            )
        if self.intensity.shape != self.phase_rad.shape:
            raise ValueError(
                f"intensity shape {self.intensity.shape} != phase shape "
                f"{self.phase_rad.shape}."
            )
        if self.field is not None:
            self.field = np.ascontiguousarray(self.field, dtype=np.complex64)
            if self.field.shape != self.phase_rad.shape:
                raise ValueError(
                    f"field shape {self.field.shape} != phase shape "
                    f"{self.phase_rad.shape}."
                )
        dx, dy = self.pixel_pitch_um
        self.pixel_pitch_um = (float(dx), float(dy))
        self.z_um = float(self.z_um)
        self.wavelength_um = float(self.wavelength_um)
        self.n_medium = float(self.n_medium)
        # metadata is kept in its JSON form (lists, plain scalars, NaN -> None),
        # so a result and its message round trip compare equal
        self.quality = _jsonable(dict(self.quality))
        self.timings_s = _jsonable(dict(self.timings_s))
        self.versions = _jsonable(dict(self.versions))
        if self.config is not None:
            self.config = _jsonable(dict(self.config))
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )

    # ------------------------------------------------------------ properties
    @property
    def shape(self) -> tuple[int, int]:
        """Shape (rows = y, columns = x) of the result grid."""
        return tuple(self.phase_rad.shape)

    @property
    def n_frames_used(self) -> int:
        """Number of frame records marked as used."""
        return sum(1 for record in self.frames if record.used)

    # ---------------------------------------------------------- construction
    @classmethod
    def from_realtime(
        cls,
        result,
        pipeline,
        *,
        include_field: bool = False,
        frames: list[FrameRecord] | None = None,
        timings_s: dict[str, float] | None = None,
        versions: dict[str, Any] | None = None,
    ) -> ReconstructionResult:
        """Build the product result from a kernel result.

        Copies phase, intensity and (optionally) the field to the host and
        synchronises the device once (piston statistics). Not for the hot path.

        Parameters
        ----------
        result : offaxisholo.realtime.RealtimeResult
            Result of ``OffAxisPipeline.result`` or ``results``.
        pipeline : offaxisholo.realtime.OffAxisPipeline
            The pipeline that produced ``result`` (configuration, counters).
        include_field : bool, optional
            Also store the complex field.
        frames : list of FrameRecord, optional
            Per-frame records; empty when not given.
        timings_s : dict, optional
            Stage timings in seconds.
        versions : dict, optional
            Versions to store; default :func:`default_versions` of the
            pipeline's array namespace.

        Returns
        -------
        ReconstructionResult
        """
        outputs = (
            ("phase", "intensity", "field") if include_field else ("phase", "intensity")
        )
        host = result.to_numpy(outputs)
        config = pipeline.config
        dx, dy = pipeline.crop_pitch
        quality = {
            "n_obj": int(pipeline.n_frames),
            "n_effective": float(result.meta.get("n_effective", pipeline.n_frames)),
            "n_bg": int(pipeline.n_background),
            "piston_std_rad": pipeline.piston_std(),
        }
        return cls(
            phase_rad=host["phase"],
            intensity=host["intensity"],
            field=host.get("field"),
            pixel_pitch_um=(dx * 1e6, dy * 1e6),
            z_um=float(result.z) * 1e6,
            wavelength_um=config.dhm.wavelength * 1e6,
            n_medium=config.propagation.n_medium,
            config_hash=pipeline.config_hash,
            frames=list(frames or []),
            quality=quality,
            timings_s=dict(timings_s or {}),
            versions=(
                dict(versions)
                if versions is not None
                else default_versions(pipeline.xp)
            ),
            config=config.to_dict(),
        )

    # -------------------------------------------------------------- messages
    def to_message(self) -> tuple[dict, list[tuple[str, np.ndarray]]]:
        """Serialise into a JSON-able header and named binary buffers.

        Returns
        -------
        header : dict
            JSON-able metadata; ``header["buffers"]`` lists ``name``, ``dtype``,
            ``shape`` and ``byte_order`` of every buffer in order.
        buffers : list of (str, numpy.ndarray)
            C-contiguous little-endian arrays: ``phase_rad``, ``intensity`` and,
            if present, ``field``.
        """
        arrays = [("phase_rad", self.phase_rad), ("intensity", self.intensity)]
        if self.field is not None:
            arrays.append(("field", self.field))
        buffers = []
        descriptors = []
        for name, array in arrays:
            dtype = _BUFFER_DTYPES[name]
            data = np.ascontiguousarray(array, dtype=dtype)
            buffers.append((name, data))
            descriptors.append(
                {
                    "name": name,
                    "dtype": _DTYPE_NAMES[dtype],
                    "shape": list(data.shape),
                    "byte_order": "little",
                }
            )
        header = _jsonable(
            {
                "message_type": MESSAGE_TYPE,
                "schema_version": self.schema_version,
                "created_utc": self.created_utc,
                "pixel_pitch_um": self.pixel_pitch_um,
                "z_um": self.z_um,
                "wavelength_um": self.wavelength_um,
                "n_medium": self.n_medium,
                "config_hash": self.config_hash,
                "frames": [record.to_dict() for record in self.frames],
                "quality": self.quality,
                "timings_s": self.timings_s,
                "versions": self.versions,
                "config": self.config,
                "buffers": descriptors,
            }
        )
        return header, buffers

    @classmethod
    def from_message(cls, header: dict, buffers) -> ReconstructionResult:
        """Rebuild a result from :meth:`to_message` output.

        Parameters
        ----------
        header : dict
            The JSON header (as decoded from text).
        buffers : sequence
            One entry per ``header["buffers"]`` descriptor, in order: a NumPy
            array, a bytes-like object (raw little-endian data as received from
            a socket) or a ``(name, payload)`` pair of either.

        Returns
        -------
        ReconstructionResult

        Raises
        ------
        ValueError
            Wrong message type, incompatible major schema version, or buffers
            that do not match their descriptors.
        """
        if header.get("message_type") != MESSAGE_TYPE:
            raise ValueError(
                f"Not a {MESSAGE_TYPE} message: {header.get('message_type')!r}."
            )
        version = str(header.get("schema_version", ""))
        if _major(version) != _major(SCHEMA_VERSION):
            raise ValueError(
                f"Incompatible schema_version {version!r} (supported: "
                f"{_major(SCHEMA_VERSION)}.x)."
            )
        descriptors = header.get("buffers", [])
        buffers = list(buffers)
        if len(buffers) != len(descriptors):
            raise ValueError(
                f"{len(descriptors)} buffers described, {len(buffers)} received."
            )
        arrays = {}
        for descriptor, payload in zip(descriptors, buffers, strict=True):
            name = descriptor["name"]
            if isinstance(payload, tuple):
                given, payload = payload
                if given != name:
                    raise ValueError(
                        f"Buffer {given!r} received where {name!r} expected."
                    )
            if name not in _BUFFER_DTYPES:
                raise ValueError(f"Unknown buffer {name!r}.")
            if descriptor.get("byte_order", "little") != "little":
                raise ValueError(f"Buffer {name!r}: only little-endian is supported.")
            dtype = np.dtype(descriptor["dtype"]).newbyteorder("<")
            if dtype != _BUFFER_DTYPES[name]:
                raise ValueError(f"Buffer {name!r} has dtype {descriptor['dtype']}.")
            shape = tuple(descriptor["shape"])
            if isinstance(payload, np.ndarray):
                array = payload
            else:
                array = np.frombuffer(payload, dtype=dtype)
                if array.size != math.prod(shape):
                    raise ValueError(
                        f"Buffer {name!r}: {array.size} values for shape {shape}."
                    )
                array = array.reshape(shape).copy()
            if array.shape != shape:
                raise ValueError(f"Buffer {name!r}: shape {array.shape} != {shape}.")
            arrays[name] = array
        missing = {"phase_rad", "intensity"} - set(arrays)
        if missing:
            raise ValueError(f"Missing buffers: {sorted(missing)}.")
        return cls(
            phase_rad=arrays["phase_rad"],
            intensity=arrays["intensity"],
            field=arrays.get("field"),
            pixel_pitch_um=tuple(header["pixel_pitch_um"]),
            z_um=header["z_um"],
            wavelength_um=header["wavelength_um"],
            n_medium=header["n_medium"],
            config_hash=header["config_hash"],
            frames=[FrameRecord.from_dict(d) for d in header.get("frames", [])],
            quality=dict(header.get("quality") or {}),
            timings_s=dict(header.get("timings_s") or {}),
            versions=dict(header.get("versions") or {}),
            config=header.get("config"),
            created_utc=header.get("created_utc", ""),
            schema_version=version,
        )


def default_versions(xp=None) -> dict[str, str | None]:
    """Versions of offaxisholo, NumPy and the array namespace in use.

    Parameters
    ----------
    xp : module, optional
        Array namespace of the pipeline (``numpy`` or ``cupy``).

    Returns
    -------
    dict
        ``{"offaxisholo": ..., "numpy": ..., "<namespace>": ...}``; the git
        revision is added by K15.
    """
    from importlib import metadata

    try:
        own = metadata.version("offaxisholo")
    except metadata.PackageNotFoundError:
        own = None
    versions = {"offaxisholo": own, "numpy": np.__version__}
    if xp is not None and xp is not np:
        name = getattr(xp, "__name__", "xp").split(".")[0]
        versions[name] = getattr(xp, "__version__", None)
    return versions
