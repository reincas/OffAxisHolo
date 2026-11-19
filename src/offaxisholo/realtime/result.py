"""
Product result of the realtime kernel: ``ReconstructionResult`` and ``FrameRecord``.

DATA layer: plain host data (NumPy arrays, JSON-able dicts), no device arrays,
no file IO, no network. A result is built once per measurement from a
``RealtimeResult`` of ``OffAxisPipeline`` (``ReconstructionResult.from_realtime``);
this is the only place where the device arrays are copied to the host.

Message form: contract ``ReconstructionResult`` 1.0 of DHMConnect
(``dhmconnect/schemas/``, DL-A3, commit ``fc85f5c``; decision G2). The vendored
examples live in ``tests/contracts/`` (K16):

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

from .quality import median_metric, reference_quality
from .selection import threshold_reason

GATE_PREFIX = "gated: "  # reject reason of frames rejected by the K17 gate

# 0.2: timings_profiled (K5); 0.3: warnings (K7); 1.0: shared contract (K16):
# intensity -> intensity_rel, pixel_pitch_um (x, y) -> pixel_um (y, x),
# n_medium -> propagation_index, parameter_set_id, reference quality keys.
SCHEMA_VERSION = "1.0"
MESSAGE_TYPE = "ReconstructionResult"

# name of the buffer -> dtype in the message
_BUFFER_DTYPES = {
    "phase_rad": np.dtype("<f4"),
    "intensity_rel": np.dtype("<f4"),
    "field": np.dtype("<c8"),
}
# Quality keys of decision D13 that every result carries. Without a reference
# mask the kernel aligns on the whole field, so the reference fraction is 0 and
# no honest phase noise exists (null); K8 / K19 fill them from a mask.
REFERENCE_QUALITY_DEFAULTS = {
    "reference_source": "whole_field",
    "reference_fraction": 0.0,
    "phase_noise_rad": None,
}
_DTYPE_NAMES = {np.dtype("<f4"): "float32", np.dtype("<c8"): "complex64"}


def to_jsonable(value):
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
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
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
        Why the frame was rejected; None for used frames, except frames that
        were used only to reach ``min_frames`` (``"forced by min_frames: …"``).
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
        return to_jsonable(
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
    intensity_rel : numpy.ndarray
        Relative intensity ``|E_comp|^2 = I_object / I_background`` of the
        compensated field, dimensionless (1.0 = empty substrate), shape (n, n),
        stored as float32. The compensation divides the object field by the
        background field, so amplitudes divide and phases subtract (DL-A3 b).
    pixel_um : tuple of float
        Pixel size ``(dy, dx)`` (axis 0, axis 1) of the result grid in µm, i.e.
        after the sideband crop (``dx' = dx * N / n``); order ``(y, x)`` as at
        every contract boundary (X-G).
    z_um : float
        Propagation distance in µm (0 = hologram plane).
    wavelength_um : float
        Vacuum wavelength in µm.
    propagation_index : float
        Refractive index of the propagation medium that belongs to ``z_um``
        (G4); the index of the request (``RealtimeResult.meta``), default the
        kernel configuration ``propagation.n_medium``.
    config_hash : str
        sha256 of the kernel configuration (``RealtimeConfig.config_hash``).
    parameter_set_id : str, optional
        Id of the parameter set the configuration came from (DL-A3 a). When
        empty, ``"config-<first 12 hex digits of config_hash>"`` is used, which
        still identifies the configuration uniquely.
    field : numpy.ndarray or None, optional
        Compensated (and propagated) complex field, complex64; only when
        requested, because it doubles the message size.
    frames : list of FrameRecord, optional
        One record per object frame (filled by the entry points, K6).
    quality : dict, optional
        Quality numbers, e.g. ``n_obj``, ``n_bg``, ``piston_std_rad`` and the
        reference keys of :data:`REFERENCE_QUALITY_DEFAULTS` (K8 adds more).
    timings_s : dict, optional
        Stage timings in seconds (K5): cumulative per stage of
        ``realtime.timing.STAGES``, ``total`` and caller-supplied values.
    timings_profiled : bool, optional
        True if the timings were measured with device synchronisation
        (``profile=True``); otherwise GPU stage times are enqueue times.
    versions : dict, optional
        Package versions used to compute the result.
    config : dict or None, optional
        The full kernel configuration (``RealtimeConfig.to_dict``).
    created_utc : str, optional
        Creation time, UTC ISO 8601; set automatically when empty.
    warnings : list of str, optional
        Conditions the consumer should know about, e.g. an old background
        (decision O6).
    schema_version : str, optional
        Version of the message schema.
    """

    phase_rad: np.ndarray
    intensity_rel: np.ndarray
    pixel_um: tuple[float, float]
    z_um: float
    wavelength_um: float
    propagation_index: float
    config_hash: str
    parameter_set_id: str = ""
    field: np.ndarray | None = None
    frames: list[FrameRecord] = dc_field(default_factory=list)
    quality: dict[str, Any] = dc_field(default_factory=dict)
    timings_s: dict[str, float] = dc_field(default_factory=dict)
    timings_profiled: bool = False
    versions: dict[str, Any] = dc_field(default_factory=dict)
    config: dict | None = None
    created_utc: str = ""
    warnings: list[str] = dc_field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        self.phase_rad = np.ascontiguousarray(self.phase_rad, dtype=np.float32)
        self.intensity_rel = np.ascontiguousarray(self.intensity_rel, dtype=np.float32)
        if self.phase_rad.ndim != 2:
            raise ValueError(
                f"phase_rad must be 2-D, got shape {self.phase_rad.shape}."
            )
        if self.intensity_rel.shape != self.phase_rad.shape:
            raise ValueError(
                f"intensity_rel shape {self.intensity_rel.shape} != phase shape "
                f"{self.phase_rad.shape}."
            )
        if self.field is not None:
            self.field = np.ascontiguousarray(self.field, dtype=np.complex64)
            if self.field.shape != self.phase_rad.shape:
                raise ValueError(
                    f"field shape {self.field.shape} != phase shape "
                    f"{self.phase_rad.shape}."
                )
        dy, dx = self.pixel_um
        self.pixel_um = (float(dy), float(dx))
        self.z_um = float(self.z_um)
        self.wavelength_um = float(self.wavelength_um)
        self.propagation_index = float(self.propagation_index)
        if not self.parameter_set_id:
            self.parameter_set_id = f"config-{self.config_hash[:12]}"
        # metadata is kept in its JSON form (lists, plain scalars, NaN -> None),
        # so a result and its message round trip compare equal
        self.quality = to_jsonable({**REFERENCE_QUALITY_DEFAULTS, **self.quality})
        self.timings_s = to_jsonable(dict(self.timings_s))
        self.versions = to_jsonable(dict(self.versions))
        if self.config is not None:
            self.config = to_jsonable(dict(self.config))
        if not self.created_utc:
            # UTC ISO 8601 with "Z" as required by the contracts
            self.created_utc = (
                datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
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
        selection=None,
        warnings: list[str] | None = None,
        parameter_set_id: str = "",
        min_reference_fraction: float | None = None,
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
            Per-frame records. When not given, one record per frame is built
            from the pipeline's frame metrics (``config.metrics.enabled``,
            K2), all marked as used; empty when metrics are disabled.
        timings_s : dict, optional
            Additional timings in seconds supplied by the caller (for example
            ``{"grab": 0.27}``), passed through unchanged next to the stage
            timings of the pipeline (``realtime.timing``). A caller key with the
            name of a stage replaces the measured value.
        versions : dict, optional
            Versions to store; default :func:`default_versions` of the
            pipeline's array namespace.
        selection : offaxisholo.realtime.selection.Selection, optional
            Frame selection applied to the pipeline (``apply_selection``). Its
            mask and reasons go into the frame records built from the metrics,
            its summary into ``quality``.
        warnings : list of str, optional
            Warnings to pass on (e.g. ``BackgroundModel.warnings``).
        parameter_set_id : str, optional
            Id of the parameter set (see the class parameters).
        min_reference_fraction : float, optional
            Warn when the reference fraction of the result is below it (SA4).

        Notes
        -----
        The ``download`` stage covers the derivation of phase and intensity
        from the field and the copy to the host; it is added to the pipeline's
        timer, so several results of one ``results()`` call accumulate.

        Returns
        -------
        ReconstructionResult
        """
        # Kernel product names; the kernel's "intensity" is |E_comp|^2, i.e. the
        # relative intensity of the contract (intensity_rel).
        outputs = (
            ("phase", "intensity", "field") if include_field else ("phase", "intensity")
        )
        timer = pipeline.timer
        started = timer.start()
        host = result.to_numpy(outputs)
        timer.lap("download", started)
        timings = timer.timings()
        timings.update(timings_s or {})
        if frames is None:
            frames = [
                FrameRecord(index=index, metrics=metrics)
                for index, metrics in pipeline.frame_metrics()
            ]
            if selection is not None:
                verdict = dict(
                    zip(
                        selection.indices,
                        zip(selection.mask, selection.reasons, strict=True),
                        strict=True,
                    )
                )
                for record in frames:
                    if record.index in verdict:
                        used, reason = verdict[record.index]
                        record.used = bool(used)
                        record.reject_reason = reason
        gate = getattr(pipeline, "gate", None) or {}
        n_gated = 0
        if gate:
            # K17: the device weight decides, the host only names the reason
            passed = dict(pipeline.frame_gate())
            for record in frames:
                if not passed.get(record.index, True):
                    n_gated += 1
                    record.used = False
                    reason = threshold_reason(record.metrics, gate)
                    record.reject_reason = GATE_PREFIX + (reason or "rejected")
        config = pipeline.config
        dx, dy = pipeline.crop_pitch
        quality = {
            "n_obj": int(pipeline.n_frames),
            "n_effective": float(result.meta.get("n_effective", pipeline.n_frames)),
            "n_bg": int(pipeline.n_background),
            "piston_std_rad": pipeline.piston_std(),
        }
        if selection is not None:
            quality.update(selection.quality())
        else:
            quality.update(
                selection_policy="all",
                n_frames_total=int(pipeline.n_frames),
                n_frames_used=int(pipeline.n_frames),
            )
        if gate:
            quality["n_frames_used"] = max(0, quality["n_frames_used"] - n_gated)
            quality["n_frames_gated"] = n_gated  # the bounds are in the config
        # K8: reference ladder (SA4), noise after plane removal, sideband SNR
        reference, reference_warnings = reference_quality(
            host["phase"],
            getattr(pipeline, "reference_region", None),
            getattr(pipeline, "reference_source", "whole_field"),
            min_reference_fraction=min_reference_fraction,
        )
        quality.update(reference)
        quality["sideband_snr"] = median_metric(frames or [], "sideband_snr")
        warnings = [*(warnings or []), *reference_warnings]
        background = pipeline.background_model()
        quality["background_id"] = background.model_id
        quality["background_created_utc"] = background.created_utc
        quality["background_age_s"] = background.age_s()
        return cls(
            phase_rad=host["phase"],
            intensity_rel=host["intensity"],
            field=host.get("field"),
            pixel_um=(dy * 1e6, dx * 1e6),
            z_um=float(result.z) * 1e6,
            wavelength_um=config.dhm.wavelength * 1e6,
            propagation_index=float(
                result.meta.get("propagation_index", config.propagation.n_medium)
            ),
            config_hash=pipeline.config_hash,
            parameter_set_id=parameter_set_id,
            frames=list(frames or []),
            quality=quality,
            timings_s=timings,
            timings_profiled=timer.profile,
            versions=(
                dict(versions)
                if versions is not None
                else default_versions(pipeline.xp)
            ),
            config=config.to_dict(),
            warnings=[str(w) for w in (warnings or [])],
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
            C-contiguous little-endian arrays: ``phase_rad``, ``intensity_rel``
            and, if present, ``field``.
        """
        arrays = [("phase_rad", self.phase_rad), ("intensity_rel", self.intensity_rel)]
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
        header = to_jsonable(
            {
                "message_type": MESSAGE_TYPE,
                "schema_version": self.schema_version,
                "created_utc": self.created_utc,
                "pixel_um": self.pixel_um,
                "z_um": self.z_um,
                "propagation_index": self.propagation_index,
                "wavelength_um": self.wavelength_um,
                "parameter_set_id": self.parameter_set_id,
                "config_hash": self.config_hash,
                "frames": [record.to_dict() for record in self.frames],
                "quality": self.quality,
                "timings_s": self.timings_s,
                "timings_profiled": bool(self.timings_profiled),
                "versions": self.versions,
                "config": self.config,
                "warnings": list(self.warnings),
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
        missing = {"phase_rad", "intensity_rel"} - set(arrays)
        if missing:
            raise ValueError(f"Missing buffers: {sorted(missing)}.")
        return cls(
            phase_rad=arrays["phase_rad"],
            intensity_rel=arrays["intensity_rel"],
            field=arrays.get("field"),
            pixel_um=tuple(header["pixel_um"]),
            z_um=header["z_um"],
            wavelength_um=header["wavelength_um"],
            propagation_index=header["propagation_index"],
            config_hash=header["config_hash"],
            parameter_set_id=header["parameter_set_id"],
            frames=[FrameRecord.from_dict(d) for d in header.get("frames", [])],
            quality=dict(header.get("quality") or {}),
            timings_s=dict(header.get("timings_s") or {}),
            timings_profiled=bool(header.get("timings_profiled", False)),
            versions=dict(header.get("versions") or {}),
            config=header.get("config"),
            created_utc=header.get("created_utc", ""),
            warnings=list(header.get("warnings") or []),
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
        ``{"offaxisholo": ..., "offaxisholo_git": ..., "numpy": ...,
        "<namespace>": ...}``; ``offaxisholo_git`` is the short git hash of
        the source tree (``-dirty`` with local changes) or None for an
        installed release (K15).
    """
    from .._version import __version__, git_revision

    own = None if __version__.endswith("+unknown") else __version__
    versions = {
        "offaxisholo": own,
        "offaxisholo_git": git_revision(),
        "numpy": np.__version__,
    }
    if xp is not None and xp is not np:
        name = getattr(xp, "__name__", "xp").split(".")[0]
        versions[name] = getattr(xp, "__version__", None)
    return versions
