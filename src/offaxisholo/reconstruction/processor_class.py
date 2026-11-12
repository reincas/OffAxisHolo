"""
Coordinates the entire reconstruction process, ensuring that all necessary steps
(spatial filtering, aberration compensation, propagation, phase unwrapping and
height calculation) are executed in the physically correct order.
"""

import warnings
from typing import Literal

import numpy as np

from ..utils import get_logger
from .base_class import HologramCore
from .hologram_class import Hologram, ReferenceHologram
from .numericalPropagation import angularSpectrum
from .phaseUnwrapping import phase_unwrapping_fast2d  # , phase_unwrapping_kamui_normal
from .plotter_class import DHMPlotter

# Refractive index of the surrounding medium for each structure mode.
#   "developed": structure after development, surrounded by air.
#   "print":     structure during printing, surrounded by the non-polymerised
#                liquid resin. The index of the liquid resin must be given in the
#                material dictionary ("refractiveIndexLiquid").
STRUCTURE_MODES = ("developed", "print")
N_AIR = 1.0


class HologramProcessor(HologramCore):
    """
    Reconstruction of an off-axis hologram (optionally with a background hologram
    for aberration compensation).

    Processing order (run):
        1. Spatial filtering of the first diffraction order -> complex field in the
           hologram plane (Hologram.reconstruct).
        2. Aberration compensation with the reference conjugated hologram in the
           hologram plane (compensate).
        3. Numerical propagation of the compensated field to the focal plane
           (propagate, angular spectrum method).
        4. Phase unwrapping of the final wrapped phase.
        5. Conversion of the unwrapped phase to a height profile (phase_to_height).

    The compensation is performed before the propagation, as the reference
    hologram describes the aberrations in the hologram plane. Compensating after
    propagating only the object field would mix fields of different planes.
    Reference:
        Tristan Colomb, Jonas Kuehn, Frederic Charriere, Christian Depeursinge,
        Pierre Marquet, and Nicolas Aspert, "Total aberrations compensation in
        digital holographic microscopy with a reference conjugated hologram,"
        Opt. Express 14, 4300-4306 (2006). https://doi.org/10.1364/OE.14.004300
    """

    hologram: Hologram
    background: ReferenceHologram = None
    required_dhm_keys = {"propagationDistance", "name", "pixelPitch", "wavelength"}
    refractive_index_default = 1.5  # default value for refractive index
    available_filters = []  # todo implement filter in the py file and add it here

    # compensated field (hologram plane)
    field_compensated = None  # E_obj * conj(E_ref) / |E_ref|^2
    phase_compensated = None  # wrapped phase of the compensated field
    intensity_compensated_linear = None  # I_obj / I_ref
    intensity_compensated_db = None  # 10 * log10(I_obj / I_ref)

    # propagated field (focal plane)
    field_propagated = None  # (compensated) field after propagation

    # filtered - to be done!
    field_filtered = None  # field after filtering (not yet implemented)

    # finished fields
    field_reconstructed = None  # final reconstructed complex field
    phase_wrapped = None  # wrapped phase of the final field
    intensity_reconstructed = None  # intensity (dB) of the final field
    phase_map = None  # unwrapped phase of the final field
    height_profile = None  # height profile calculated from the unwrapped phase

    def __init__(
        self,
        hologram: Hologram,
        reference: ReferenceHologram = None,
        dhm_parameter: dict = None,
        material_parameter: dict = None,
        logger=None,
        saving_path=None,
    ):
        super().__init__(logger if logger is not None else get_logger())
        self.hologram = hologram
        # copy, so that the caller's dictionary is not modified by update()
        self.params = dict(dhm_parameter) if dhm_parameter is not None else {}
        if material_parameter is not None:
            self.params["MaterialDictionary"] = material_parameter
        self.material = material_parameter if material_parameter is not None else {}
        self.reconstruction_dict = {}

        self.refractiveIndex = self.material.get(
            "refractiveIndex", self.refractive_index_default
        )

        if reference is not None:
            self.background = reference
            self.background_available = True
        else:
            self.background_available = False

        self._validate_input()
        self.plotter = DHMPlotter(img_path=saving_path)

    def plot_reconstruction(
        self,
        show_plot=False,
        save_single=False,
        cmap="gray",
        title=None,
        mode: Literal["short", "full"] = "short",
        compensation=True,
    ):
        """
        Plot the reconstruction. 'short' shows hologram, background, final phase
        and intensity; 'full' shows every intermediate step.
        """
        if not self.plotter.img_save_path:
            show_plot = True

        if mode.lower() == "short":
            self.plotter.plot_reconstruction_short(
                self,
                show_plot=show_plot,
                save_single=save_single,
                cmap=cmap,
                save_title=title,
                compensation=compensation,
            )
        elif mode.lower() == "full":
            self.plotter.plot_full_reconstruction_process(
                self,
                show_plot=show_plot,
                save_single=save_single,
                cmap=cmap,
                save_title=title,
            )
        else:
            raise ValueError(
                "Plotting mode can only be 'short' reconstruction or "
                "'full' reconstruction."
            )

    @property
    def pixel_pitch(self):
        return self.params["pixelPitch"]

    def _validate_input(self):
        self.logger.debug("Validating input for reconstruction process")
        assert isinstance(self.hologram, Hologram), "Hologram must be of type Hologram."
        assert all(
            key in self.params for key in self.required_dhm_keys
        ), f"Missing required parameter(s): \
                {self.required_dhm_keys - self.params.keys()}"

    def medium_refractive_index(self, mode: str) -> float:
        """
        Return the refractive index of the medium surrounding the structure.

        'developed': structure in air (n = 1.0) unless the material dictionary
                     defines "refractiveIndexMedium".
        'print':     structure embedded in the liquid, non-polymerised resin. The
                     index has to be given as "refractiveIndexLiquid" in the
                     material dictionary.
        """
        mode = mode.lower()
        if mode not in STRUCTURE_MODES:
            raise ValueError(
                f"Unknown structure mode '{mode}'. Possible modes: {STRUCTURE_MODES}."
            )
        if mode == "developed":
            return self.material.get("refractiveIndexMedium", N_AIR)
        if "refractiveIndexLiquid" not in self.material:
            raise ValueError(
                "Mode 'print' requires the refractive index of the liquid resin "
                "('refractiveIndexLiquid') in the material dictionary."
            )
        return self.material["refractiveIndexLiquid"]

    def run(
        self,
        hologram: Hologram = None,
        background_hologram: ReferenceHologram = None,
        *,
        prop_dist=None,
        propagate=True,
        compensate=True,
        compensation_mode: Literal["Background", "ZernikePolynomial"] = "Background",
        filtering=False,
        filter_applied: list = None,  # todo think of a better name for the list
        refractive_index=None,
        n_medium=None,
        mode: Literal["developed", "print"] = "developed",
        phase_unwrapping_method="Fast 2D",
        propagation_method="angularSpectrum",
        propagation_padding=False,
    ) -> None:
        """
        Full reconstruction (see class documentation for the processing order).

        Parameters
        ----------
        prop_dist : float
            Propagation distance in metres. Default: DHM parameter
            "propagationDistance". Set propagate=False to skip propagation.
        compensate : bool
            Aberration compensation with the background hologram.
        refractive_index : float
            Refractive index of the structure. Default: material dictionary.
        n_medium : float
            Refractive index of the surrounding medium. Overrides the value
            derived from 'mode'.
        mode : 'developed' or 'print'
            'developed': structure in air; 'print': structure in liquid resin
            (see medium_refractive_index).
        phase_unwrapping_method : str
            Only 'Fast 2D' is available.
        propagation_padding : bool
            Padding against wrap-around during propagation (see
            numericalPropagation.angularSpectrum).
        """
        self.logger.info("Starting hologram processing ...")

        if hologram is None:
            hologram = self.hologram
        if background_hologram is None:
            if self.background_available:
                background_hologram = self.background
            else:
                compensate = False

        if refractive_index is None:
            refractive_index = self.refractiveIndex
        if n_medium is None:
            n_medium = self.medium_refractive_index(mode)

        if prop_dist is None and propagate:
            prop_dist = self.params["propagationDistance"]

        self.logger.debug(
            f"Hologram processing with {mode} mode. "
            f"Refractive index structure: {refractive_index}, "
            f"medium: {n_medium}. "
            f"Propagation distance: {prop_dist}. "
            f"Compensation: {compensate}"
        )

        # 1. Spatial filtering -> complex field in the hologram plane
        holo_field = hologram.reconstruct(force=True)

        # 2. Aberration compensation in the hologram plane
        if compensate:
            if compensation_mode == "Background":
                self.logger.info("Starting compensation with the background field...")
                background_field = background_hologram.reconstruct()
                field = self.compensate(original=holo_field, reference=background_field)
            elif compensation_mode == "ZernikePolynomial":
                self.logger.warning("ZernikePolynomial not implemented yet.")
                raise NotImplementedError("ZernikePolynomial not yet implemented.")
            else:
                raise NotImplementedError(
                    f"Compensation method {compensation_mode} not implemented."
                )
        else:
            field = holo_field

        # 3. Propagation to the focal plane
        if propagate:
            self.logger.info(
                f"Starting propagation of reconstructed field "
                f"with {propagation_method} method."
            )
            self.field_propagated = self.propagate(
                field=field,
                distance=prop_dist,
                propagation_method=propagation_method,
                padding=propagation_padding,
            )
        else:
            self.field_propagated = field

        # Filtering
        # ToDo: Filtering needs implementation of algorithms
        if filtering:
            if filter_applied is None:
                warnings.warn(
                    "No filter selected! "
                    "Please select filter using the variable 'filter_applied'.",
                    stacklevel=2,
                )
                self.logger.info("No filter applied.")
                filter_applied = []
            else:
                self.logger.info("Start applying filters.")
            for filter_name in filter_applied:
                if filter_name not in self.available_filters:
                    raise NotImplementedError(f"Filter {filter_name} not implemented.")
        else:
            self.logger.info("No filter applied.")

        self.field_reconstructed = self.field_propagated

        # 4. Phase unwrapping
        self.phase_wrapped = self.phase(self.field_reconstructed)
        self.intensity_reconstructed = self.intensity(
            self.field_reconstructed, mode="db"
        )
        self.logger.info(
            f"Starting phase unwrapping with {phase_unwrapping_method} method."
        )
        self.phase_map = self.phase_unwrapping(
            self.phase_wrapped, method=phase_unwrapping_method
        )

        # 5. Height profile
        self.logger.info(
            f"Converting unwrapped phase to height map using refractive index "
            f"{refractive_index} (structure) and {n_medium} (medium)."
        )
        self.height_profile = self.phase_to_height(
            self.phase_map, n_resin=refractive_index, n_medium=n_medium
        )

        self.reconstruction_dict = {
            "Field propagated": propagate,
            "Field compensated": compensate,
            "Compensation method": (
                "Reference conjugated hologram (background image)"
                if compensate
                else "None"
            ),
            "propagationDistance": prop_dist,
            "propagationMethod": propagation_method,
            "propagationPadding": propagation_padding,
            "phaseUnwrappingMethod": phase_unwrapping_method,
            "structureMode": mode,
            "refractiveIndex_used": refractive_index,
            "refractiveIndexMedium_used": n_medium,
            "usedFilter": "Filtering not yet implemented.",
            "DHM Dictionary": self.params,
        }

    def propagate(
        self,
        field,
        distance,
        pixel_pitch: list[float] = None,
        propagation_method: Literal["angularSpectrum", "Fresnel"] = "angularSpectrum",
        padding=False,
    ):
        """
        Propagate a complex field by the given distance (metres).

        The pixel pitch has to be the pitch in the object plane (camera pixel size
        divided by the magnification). The refractive index of the propagation
        medium can be set with the DHM parameter "propagationMediumIndex"
        (default 1.0). See numericalPropagation.angularSpectrum for the theory.
        """
        if distance is None:
            distance = self.params["propagationDistance"]
        if float(distance) == 0.0:
            return field
        if pixel_pitch is None:
            dx = self.params["pixelPitch"][0]
            dy = self.params["pixelPitch"][1]
        elif isinstance(pixel_pitch, (float, int)):
            dx = dy = pixel_pitch
        else:
            dx = pixel_pitch[0]
            dy = pixel_pitch[1]

        if propagation_method == "angularSpectrum":
            propagated = angularSpectrum(
                field=field,
                z=distance,
                wavelength=self.params["wavelength"],
                dx=dx,
                dy=dy,
                n_medium=self.params.get("propagationMediumIndex", 1.0),
                padding=padding,
            )
        elif propagation_method == "Fresnel":
            raise NotImplementedError("Fresnel propagation not yet implemented.")
        else:
            raise NotImplementedError(
                f"No propagation method {propagation_method} implemented."
            )

        self.logger.info("Propagation finished ...")
        return propagated

    def compensate(self, original: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Aberration compensation with a reference (background) hologram.

        The background hologram is recorded with the same optical setup but
        without the specimen. Its reconstructed field E_ref contains all phase
        aberrations (wavefront curvature, tilt, higher order aberrations) and the
        inhomogeneous illumination. Dividing the object field by the reference
        field removes them:

            E_comp = E_obj / E_ref = E_obj * conj(E_ref) / |E_ref|^2

        This yields
            phase:      phi_comp = arg(E_obj * conj(E_ref)) = phi_obj - phi_ref
                        (wrapped to (-pi, pi])
            intensity:  I_comp   = |E_comp|^2 = I_obj / I_ref
            in dB:      I_comp,dB = I_obj,dB - I_ref,dB

        i.e. the compensated intensity is the relative intensity transmission of
        the specimen. A subtraction of linear intensities is not physically
        meaningful and is therefore not used. To avoid division by zero, a small
        regularisation (machine epsilon times the maximum reference intensity) is
        added to |E_ref|^2.

        References:
            Pietro Ferraro, Sergio De Nicola, Andrea Finizio, Giuseppe Coppola,
            Simonetta Grilli, Carlo Magro, and Giovanni Pierattini, "Compensation of
            the inherent wave front curvature in digital holographic coherent
            microscopy for quantitative phase-contrast imaging,"
            Appl. Opt. 42, 1938-1946 (2003). https://doi.org/10.1364/AO.42.001938

            Tristan Colomb, Jonas Kuehn, Frederic Charriere, Christian Depeursinge,
            Pierre Marquet, and Nicolas Aspert, "Total aberrations compensation in
            digital holographic microscopy with a reference conjugated hologram,"
            Opt. Express 14, 4300-4306 (2006). https://doi.org/10.1364/OE.14.004300
        """
        reference_intensity = np.abs(reference) ** 2
        eps = np.finfo(float).eps * np.max(reference_intensity)
        self.field_compensated = (
            original * np.conj(reference) / (reference_intensity + eps)
        )
        self.phase_compensated = self.phase(self.field_compensated)
        self.intensity_compensated_linear = self.intensity(
            self.field_compensated, mode="linear"
        )
        self.intensity_compensated_db = self.intensity(
            self.field_compensated, mode="db"
        )
        self.logger.info("Compensation finished ...")
        return self.field_compensated

    def phase_unwrapping(self, phase_wrapped, method: Literal["Fast 2D"] = "Fast 2D"):
        """
        2D phase unwrapping. See phaseUnwrapping.phase_unwrapping_fast2d for the
        algorithm and reference.
        """
        if method == "Fast 2D":
            phase_unwrapped = phase_unwrapping_fast2d(phase_wrapped)
        # elif method == "Kamui":
        #     phase_unwrapped = phase_unwrapping_kamui_normal(phase_wrapped)
        else:
            raise ValueError(
                f"Unsupported phase unwrapping method: {method}. "
                f"Possible unwrapping method is 'Fast 2D'."
            )
        self.logger.info("Phase unwrapping finished ...")
        return phase_unwrapped

    def phase_to_height(self, phase, n_resin=None, n_medium=N_AIR):
        """
        Convert the unwrapped phase (transmission measurement) into a height map.

        Light passing a structure of height h and refractive index n_s, surrounded
        by a medium of refractive index n_m, accumulates the optical path
        difference OPD = (n_s - n_m) * h relative to light passing only the
        medium. The corresponding phase difference is

            delta_phi = 2 * pi / lambda0 * (n_s - n_m) * h

        and therefore

            h = lambda0 * delta_phi / (2 * pi * (n_s - n_m)),

        with the vacuum wavelength lambda0. This holds for a transmission setup
        with (approximately) normal incidence and a homogeneous structure. The
        height is relative: the phase offset (e.g. of the substrate) has to be
        removed by the user, as the unwrapped phase has an arbitrary constant.

        References:
            Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and
            Rola Aylo, "Accurate quantitative phase digital holographic microscopy
            with single- and multiple-wavelength telecentric and nontelecentric
            configurations," Appl. Opt. 55, 5666-5683 (2016).
            http://dx.doi.org/10.1364/AO.55.005666

            Benjamin Rappaz, Pierre Marquet, Etienne Cuche, Yves Emery, Christian
            Depeursinge, and Pierre Magistretti, "Measurement of the integral
            refractive index and dynamic cell morphometry of living cells with
            digital holographic microscopy," Opt. Express 13, 9361-9373 (2005).
            https://doi.org/10.1364/OPEX.13.009361
        """
        if n_resin is None:
            n_resin = self.refractive_index_default
        if n_medium is None:
            n_medium = N_AIR
        delta_n = n_resin - n_medium
        if delta_n == 0:
            raise ValueError(
                "Refractive index of structure and medium are identical; "
                "height cannot be calculated."
            )
        return self.params["wavelength"] * phase / (2 * np.pi * delta_n)

    def get_dict(self):
        return self.reconstruction_dict if self.reconstruction_dict else self.params
