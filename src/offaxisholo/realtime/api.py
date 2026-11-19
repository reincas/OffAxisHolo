"""
Entry points of the product interface (K6): ``reconstruct_frames`` and
``ReconstructionSession``.

PIPELINE layer: thin orchestration over ``OffAxisPipeline`` (kernel),
``select_frames`` (selection) and ``ReconstructionResult`` (product). No file
IO, no logging, no network. The DHM service (DHMConnect DL-B4) and the GUI
import only these two names plus the configuration classes
(``RealtimeConfig`` and its parts, ``SelectionPolicy``):

    from offaxisholo.realtime import (
        RealtimeConfig, SelectionPolicy, ReconstructionSession, reconstruct_frames,
    )

    # one call per measurement
    result = reconstruct_frames(frames, background, config)

    # stateful, frame by frame (service / live)
    session = ReconstructionSession(config, selection=SelectionPolicy("robust"))
    session.warmup()
    session.set_background(background_frames)
    for frame, info in camera:
        session.push(frame, info)
    result = session.result()
    session.reset()

Defaults (maintainer decisions of 2026-11-16): frame metrics are on (they can be
switched off with ``metrics=False``); the selection policy is ``all`` (O3).
A policy other than ``all`` needs the block averaging mode, because the fields
are kept until the selection (``realtime.selection``, design note). The
sliding and EMA modes gate frames by fixed thresholds on the device instead
(``AveragingConfig.gate``, K17); gated frames appear in the frame records with
``used = False`` and a ``"gated: …"`` reason.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from .background import BackgroundModel
from .config import RealtimeConfig
from .core import OffAxisPipeline
from .result import FrameRecord, ReconstructionResult
from .selection import SelectionPolicy, select_frames

OUTPUTS = ("phase", "intensity_rel", "field")
# Accepted for callers written before the contract 1.0 (K16).
_OUTPUT_ALIASES = {"intensity": "intensity_rel"}


def _check_outputs(outputs) -> bool:
    """Validate ``outputs``; returns whether the complex field is requested."""
    outputs = tuple(_OUTPUT_ALIASES.get(name, name) for name in outputs)
    unknown = set(outputs) - set(OUTPUTS)
    if unknown:
        raise ValueError(f"Unknown outputs {sorted(unknown)}. Possible: {OUTPUTS}.")
    return "field" in outputs


class ReconstructionSession:
    """Stateful reconstruction of measurements frame by frame.

    Parameters
    ----------
    config : RealtimeConfig
        Kernel configuration with the calibrated order.
    selection : SelectionPolicy, optional
        Frame selection applied in :meth:`result`; default policy ``all``.
    outputs : sequence of str, optional
        Products of every result: ``"phase"`` and ``"intensity_rel"`` are
        always included, ``"field"`` adds the complex field (``"intensity"`` is
        accepted as an old name of ``"intensity_rel"``).
    metrics : bool, optional
        Compute the per-frame metrics (default True). Overrides
        ``config.metrics.enabled``; the stored configuration shows the value
        used.
    xp : module, optional
        Array namespace; default from ``config.backend``.
    profile : bool, optional
        Device-synchronised stage timings (diagnostics only).
    background_max_age_s : float or None, optional
        Results get a warning when the background is older (decision O6);
        None (default) disables the check. A background never expires.
    parameter_set_id : str, optional
        Id of the parameter set, stored in every result (DL-A3 a); default
        ``"config-<hash>"`` of the configuration.
    min_reference_fraction : float, optional
        Results warn when their reference fraction is below it (SA4, K8;
        parameter set). None disables the warning.

    Notes
    -----
    Exactly one thread may use a session (it owns the CUDA context of its
    pipeline). ``push`` is the hot path; ``result`` synchronises once.
    """

    def __init__(
        self,
        config: RealtimeConfig,
        *,
        selection: SelectionPolicy | None = None,
        outputs=("phase", "intensity_rel"),
        metrics: bool = True,
        xp=None,
        profile: bool = False,
        background_max_age_s: float | None = None,
        parameter_set_id: str = "",
        min_reference_fraction: float | None = None,
    ):
        self.parameter_set_id = parameter_set_id
        self.min_reference_fraction = min_reference_fraction
        self.selection = (selection or SelectionPolicy()).validate()
        self.background_max_age_s = background_max_age_s
        self._include_field = _check_outputs(outputs)
        config = replace(config, metrics=replace(config.metrics, enabled=bool(metrics)))
        selecting = self.selection.policy != "all"
        if selecting and not metrics:
            raise ValueError(
                f"Selection policy '{self.selection.policy}' needs the frame metrics."
            )
        if selecting and config.averaging.mode != "block":
            raise ValueError(
                f"Selection policy '{self.selection.policy}' needs averaging mode "
                f"'block'; sliding / EMA gate frames with averaging.gate (K17)."
            )
        self.pipeline = OffAxisPipeline(
            config, xp=xp, profile=profile, keep_fields=selecting
        )
        capacity = (
            self.pipeline._metrics.capacity
            if self.pipeline._metrics is not None
            else max(config.averaging.n_obj, config.averaging.window)
        )
        self._infos = deque(maxlen=capacity)
        self._pushed = 0

    @property
    def config(self) -> RealtimeConfig:
        """The configuration in use (with the effective metrics setting)."""
        return self.pipeline.config

    @property
    def n_frames(self) -> int:
        """Frames pushed since the last reset."""
        return self._pushed

    def warmup(self):
        """Create FFT plans and transfer functions before the first frame."""
        self.pipeline.warmup()

    def set_background(self, frames):
        """Average a new background from an iterable of frames.

        Resets the current measurement.

        Returns
        -------
        dict
            Quality of the background (``OffAxisPipeline.background_quality``;
            synchronises once).
        """
        self.pipeline.set_background(frames)
        self._infos.clear()
        self._pushed = 0
        return self.pipeline.background_quality()

    def use_background(self, model: BackgroundModel):
        """Use a stored background model (K7) instead of recording one.

        Resets the current measurement.

        Parameters
        ----------
        model : BackgroundModel
            Must fit the configuration (``RealtimeConfig.background_hash``).

        Returns
        -------
        dict
            The quality stored with the model.
        """
        self.pipeline.use_background(model)
        self._infos.clear()
        self._pushed = 0
        return self.pipeline.background_quality()

    def background_model(self) -> BackgroundModel:
        """The current background as a model to save or share."""
        return self.pipeline.background_model()

    def push(self, frame, info: dict | None = None):
        """Add one object frame (hot path).

        Parameters
        ----------
        frame : array
            Raw camera frame (N, N), host or device array.
        info : dict, optional
            Acquisition values of this frame (exposure, camera and host time,
            frame id, …), passed through unchanged into its ``FrameRecord``.
        """
        self.pipeline.push(frame)
        if info is not None:
            self._infos.append((self._pushed, info))
        self._pushed += 1

    def set_reference_mask(self, mask):
        """A-priori reference mask of the next measurement (SA1, K8).

        Parameters
        ----------
        mask : array_like of bool or None
            True = structure-free pixel of the reconstruction grid, e.g. from
            :func:`offaxisholo.realtime.reference_mask.reference_mask`; None
            returns to the border ring (``compensation.ref_border``) or the
            whole field. Set it before the first frame of a measurement; it
            stays until replaced.
        """
        self.pipeline.set_reference_mask(mask)

    def reset(self):
        """Start a new measurement (the background is kept)."""
        self.pipeline.reset()
        self._infos.clear()
        self._pushed = 0

    def result(self, z: float | None = None, timings_s: dict | None = None):
        """Select the frames and build the product result.

        Parameters
        ----------
        z : float, optional
            Propagation distance in metres; default the single configured z
            (or the hologram plane when none is configured).
        timings_s : dict, optional
            Additional caller timings in seconds (e.g. the grab time).

        Returns
        -------
        ReconstructionResult
        """
        pipe = self.pipeline
        selection = None
        if self.selection.policy != "all":
            selection = select_frames(pipe.frame_metrics(), self.selection)
            pipe.apply_selection(selection.mask)
        elif pipe._metrics is not None and pipe.n_frames:
            selection = select_frames(pipe.frame_metrics(), self.selection)
        warnings = pipe.background_model().warnings(self.background_max_age_s)
        result = ReconstructionResult.from_realtime(
            pipe.result(z),
            pipe,
            include_field=self._include_field,
            timings_s=timings_s,
            selection=selection,
            warnings=warnings,
            parameter_set_id=self.parameter_set_id,
            min_reference_fraction=self.min_reference_fraction,
        )
        infos = dict(self._infos)
        if result.frames:
            for record in result.frames:
                if record.index in infos:
                    record.acquisition = dict(infos[record.index])
        elif infos:
            result.frames = [
                FrameRecord(index=index, acquisition=dict(info))
                for index, info in sorted(infos.items())
            ]
        return result


def reconstruct_frames(
    frames,
    background,
    config: RealtimeConfig,
    *,
    selection: SelectionPolicy | None = None,
    outputs=("phase", "intensity_rel"),
    frame_info=None,
    metrics: bool = True,
    z: float | None = None,
    xp=None,
    background_max_age_s: float | None = None,
    parameter_set_id: str = "",
    reference_mask=None,
    min_reference_fraction: float | None = None,
) -> ReconstructionResult:
    """Reconstruct one measurement from its object and background frames.

    Parameters
    ----------
    frames : sequence of array
        Raw object frames (N, N); an iterable is materialised once.
    background : iterable of array or BackgroundModel
        Raw background frames (N, N), or a stored background model (K7).
    config : RealtimeConfig
        Kernel configuration with the calibrated order. ``averaging.n_obj``
        is set to the number of object frames and the averaging mode to
        ``block`` (one measurement).
    selection : SelectionPolicy, optional
        Frame selection; default policy ``all``.
    outputs : sequence of str, optional
        ``"phase"`` and ``"intensity_rel"`` are always included; ``"field"``
        adds the complex field.
    frame_info : sequence of dict, optional
        Acquisition values per object frame, passed through unchanged.
    metrics : bool, optional
        Compute the per-frame metrics (default True).
    z : float, optional
        Propagation distance in metres; default from the configuration.
    xp : module, optional
        Array namespace; default from ``config.backend``.
    background_max_age_s : float or None, optional
        Warn when a stored background is older (O6); None disables it.
    parameter_set_id : str, optional
        Id of the parameter set, stored in the result (DL-A3 a).
    reference_mask : array_like of bool, optional
        A-priori reference mask on the reconstruction grid (SA1, K8); default
        the border ring of ``compensation.ref_border`` or the whole field.
    min_reference_fraction : float, optional
        Warn when the reference fraction is below it (SA4).

    Returns
    -------
    ReconstructionResult
    """
    frames = list(frames)
    if not frames:
        raise ValueError("At least one object frame is required.")
    if frame_info is not None:
        frame_info = list(frame_info)
        if len(frame_info) != len(frames):
            raise ValueError(
                f"{len(frame_info)} frame_info entries for {len(frames)} frames."
            )
    averaging = replace(config.averaging, mode="block", n_obj=len(frames))
    session = ReconstructionSession(
        replace(config, averaging=averaging),
        selection=selection,
        outputs=outputs,
        metrics=metrics,
        xp=xp,
        background_max_age_s=background_max_age_s,
        parameter_set_id=parameter_set_id,
        min_reference_fraction=min_reference_fraction,
    )
    if reference_mask is not None:
        session.set_reference_mask(reference_mask)
    if isinstance(background, BackgroundModel):
        session.use_background(background)
    else:
        session.set_background(background)
    for index, frame in enumerate(frames):
        session.push(frame, None if frame_info is None else frame_info[index])
    return session.result(z)
