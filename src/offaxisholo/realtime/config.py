"""
Typed configuration of the realtime kernel (single source of all parameters).

All lengths in metres (as in the offline library). The position of the first
diffraction order ``order = (fx, fy)`` is given in pixels relative to the
centre of the centred spectrum, exactly as returned by Hologram.locate_order;
it is determined once by calibration (realtime.calibration) and never searched
in the hot path.

Open physical choices (plane / z, propagation medium, sign convention) are
explicit configuration values and not hard-wired (see docs/realtime).
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

PRECISIONS = ("single", "double")
BACKENDS = ("numpy", "cupy")
MASKS = ("circle", "tukey")
AVERAGING_MODES = ("block", "sliding", "ema")


def mask_radius(order, size, dc_radius) -> float:
    """
    Maximum radius of the sideband mask, identical to
    Hologram._calc_radius_mask: the circle must neither touch the DC term
    (radius dc_radius) nor the spectrum border.
    """
    fx, fy = order
    rmax = math.sqrt(fx**2 + fy**2) - dc_radius
    rmax = min(rmax, abs(fx), size // 2 - abs(fx), abs(fy), size // 2 - abs(fy))
    if rmax <= 0:
        raise ValueError(
            f"Invalid mask radius {rmax}: first diffraction order at ({fx}, {fy}) "
            f"is too close to the DC term (dcRadius={dc_radius}) or to the "
            f"spectrum border."
        )
    return rmax


@dataclass
class DHMConfig:
    """Microscope: vacuum wavelength, pixel pitch in the object plane."""

    wavelength: float
    pixel_pitch: tuple[float, float]  # (dx along columns, dy along rows)
    dc_radius: float  # radius of the zero order in spectrum pixels
    name: str = ""

    @classmethod
    def from_preset(cls, preset) -> DHMConfig:
        """
        From a DHM preset name (config/DHM_preset) or a DHM parameter dict.
        Uses 'wavelength' and 'pixelPitch' (metres); the duplicated keys
        'wavelengthUm', 'xPixelSizeUm', 'yPixelSizeUm' are only cross-checked.
        """
        if isinstance(preset, str):
            from ..config.presets import load_dhm_preset

            preset = load_dhm_preset(preset)
        wavelength = float(preset["wavelength"])
        pitch = preset["pixelPitch"]
        if isinstance(pitch, (int, float)):
            pitch = (pitch, pitch)
        pitch = (float(pitch[0]), float(pitch[1]))
        checks = (
            ("wavelengthUm", wavelength),
            ("xPixelSizeUm", pitch[0]),
            ("yPixelSizeUm", pitch[1]),
        )
        for key, value in checks:
            if key in preset and not math.isclose(
                float(preset[key]) * 1e-6, value, rel_tol=1e-6
            ):
                warnings.warn(
                    f"Preset value {key}={preset[key]} um differs from the value "
                    f"in metres ({value}); the value in metres is used.",
                    stacklevel=2,
                )
        return cls(
            wavelength=wavelength,
            pixel_pitch=pitch,
            dc_radius=float(preset["dcRadius"]),
            name=str(preset.get("name", "")),
        )


@dataclass
class DemodConfig:
    """Sideband demodulation (spatial filtering)."""

    size: int  # N of the square camera image (N x N)
    order: tuple[int, int] | None = None  # (fx, fy) from calibration
    crop: int | None = None  # n of the cropped sideband; None = N (parity mode)
    radius: float | None = None  # mask radius; None = mask_radius(order, ...)
    mask: str = "circle"  # 'circle' (offline parity) or 'tukey' (whole crop
    # weighted with a Tukey window, radius not used; reference skeleton)
    sideband_alpha: float = 0.5  # Tukey parameter of the sideband window
    edge_alpha: float = 0.0  # Tukey window on the hologram, 0 = no window

    @property
    def n(self) -> int:
        return self.size if self.crop is None else self.crop


@dataclass
class CompensationConfig:
    """Background compensation K = conj(Ub) / (|Ub|^2 + eps_rel * max|Ub|^2)."""

    eps_rel: float | str = 1e-3  # 'machine' = finfo(dtype).eps (offline parity)
    piston: bool = True  # correct the global phase of every frame
    ref_border: int | None = None  # border width (crop pixels) of the reference
    # region for the piston phase; None = whole field


@dataclass
class AveragingConfig:
    """Averaging of the complex (piston corrected) object fields."""

    mode: str = "block"
    n_obj: int = 16  # frames per measurement (block)
    n_bg: int = 64  # frames of the background
    window: int = 16  # number of frames of the sliding mode
    ema_alpha: float = 0.05
    renorm_every: int = 1000  # sliding: re-sum the ring buffer against drift


@dataclass
class PropagationConfig:
    """Angular spectrum propagation of the compensated field."""

    z: tuple[float, ...] = ()  # distances in metres; () = hologram plane
    padding: bool = False  # border-mean + zero padding (see angularSpectrum)
    band_limit: bool = True
    n_medium: float = 1.0  # refractive index of the propagation medium


@dataclass
class RealtimeConfig:
    dhm: DHMConfig
    demod: DemodConfig
    compensation: CompensationConfig = field(default_factory=CompensationConfig)
    averaging: AveragingConfig = field(default_factory=AveragingConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    precision: str = "single"
    backend: str = "numpy"

    # ------------------------------------------------------------ derived
    @property
    def radius(self) -> float:
        d = self.demod
        if d.radius is not None:
            return float(d.radius)
        return mask_radius(d.order, d.size, self.dhm.dc_radius)

    @property
    def crop_pitch(self) -> tuple[float, float]:
        """Pixel pitch dx' = dx * N / n of the demodulated field."""
        factor = self.demod.size / self.demod.n
        return (self.dhm.pixel_pitch[0] * factor, self.dhm.pixel_pitch[1] * factor)

    # --------------------------------------------------------- validation
    def validate(self) -> RealtimeConfig:
        """Check consistency; raises ValueError with a clear message."""
        d, c, a, p = self.demod, self.compensation, self.averaging, self.propagation
        if d.size <= 0 or d.size % 2:
            raise ValueError(f"Image size N={d.size} must be positive and even.")
        if d.n <= 0 or d.n % 2 or d.n > d.size:
            raise ValueError(f"Crop n={d.n} must be even and 0 < n <= N={d.size}.")
        if d.order is None:
            raise ValueError("Position of the first order missing (calibrate first).")
        fx, fy = d.order
        if not (abs(fx) < d.size // 2 and abs(fy) < d.size // 2):
            raise ValueError(f"Order ({fx}, {fy}) outside of the spectrum.")
        if d.mask not in MASKS:
            raise ValueError(f"Unknown mask '{d.mask}'. Possible: {MASKS}.")
        if d.mask == "circle":
            radius = self.radius  # raises for invalid automatic radius
            # the circle must lie completely inside the crop (offsets -n/2..n/2-1)
            if not 0 < radius < d.n / 2:
                raise ValueError(
                    f"Mask radius {radius} must be in (0, n/2) with n={d.n}; "
                    f"increase the crop size or reduce the radius."
                )
        for name in ("sideband_alpha", "edge_alpha"):
            if not 0.0 <= getattr(d, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if c.eps_rel != "machine" and not float(c.eps_rel) > 0:
            raise ValueError("eps_rel must be positive or 'machine'.")
        if c.ref_border is not None and not 0 < c.ref_border < d.n // 2:
            raise ValueError("ref_border must be in (0, n/2).")
        if a.mode not in AVERAGING_MODES:
            raise ValueError(f"Unknown mode '{a.mode}'. Possible: {AVERAGING_MODES}.")
        if min(a.n_obj, a.n_bg, a.window, a.renorm_every) < 1:
            raise ValueError("n_obj, n_bg, window and renorm_every must be >= 1.")
        if not 0.0 < a.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if not all(math.isfinite(z) for z in p.z):
            raise ValueError("Propagation distances must be finite.")
        if self.precision not in PRECISIONS:
            raise ValueError(f"Unknown precision. Possible: {PRECISIONS}.")
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend. Possible: {BACKENDS}.")
        return self

    # ------------------------------------------------- (de)serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RealtimeConfig:
        def build(klass, values):
            kwargs = {}
            for f in fields(klass):
                if f.name not in values:
                    continue
                value = values[f.name]
                sub = _SUBCONFIGS.get((klass, f.name))
                if sub is not None:
                    value = build(sub, value)
                elif isinstance(value, list):
                    value = tuple(value)
                kwargs[f.name] = value
            unknown = set(values) - {f.name for f in fields(klass)}
            if unknown:
                raise ValueError(f"Unknown keys for {klass.__name__}: {unknown}")
            return klass(**kwargs)

        return build(cls, data)

    def to_json(self, path=None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path_or_text) -> RealtimeConfig:
        text = str(path_or_text)
        if not text.lstrip().startswith("{"):
            text = Path(path_or_text).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def config_hash(self) -> str:
        """sha256 of the canonical JSON (identical configurations -> same hash)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_order(self, order, radius=None) -> RealtimeConfig:
        """Copy with the calibrated order (and optionally mask radius)."""
        demod = replace(self.demod, order=(int(order[0]), int(order[1])))
        if radius is not None:
            demod = replace(demod, radius=float(radius))
        return replace(self, demod=demod)


_SUBCONFIGS = {
    (RealtimeConfig, "dhm"): DHMConfig,
    (RealtimeConfig, "demod"): DemodConfig,
    (RealtimeConfig, "compensation"): CompensationConfig,
    (RealtimeConfig, "averaging"): AveragingConfig,
    (RealtimeConfig, "propagation"): PropagationConfig,
}
