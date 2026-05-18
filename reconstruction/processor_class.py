import warnings
from typing import Any, Literal
import numpy as np
import os

from .plotter_class import DHMPlotter
from .phaseUnwrapping import phase_unwrapping_fast2d, phase_unwrapping_numpy # , phase_unwrapping_kamui_normal
from .numericalPropagation import angularSpectrum

from .base_class import HologramCore
from .hologram_class import Hologram, ReferenceHologram

from .. import get_logger

"""
Coordinates the entire reconstruction process, ensuring that all necessary steps (FFT, filtering, compensation) are 
properly executed.
"""


class HologramProcessor(HologramCore):
    hologram: Hologram
    background: ReferenceHologram = None
    required_dhm_keys = {'propagationDistance', 'name', 'pixelPitch', 'wavelength'}
    refractive_index_default = 1.5  # default value for refractive index
    available_filters = []  # todo implement filter in the py file and add it here

    # All possible fields
    # propagated field
    field_propagated = None  # field after propagation

    # compensated fields
    phase_compensated = None  # phase after compensation
    intensity_compensated_linear = None  # intensity after compensation
    intensity_compensated_db = None  # intensity after compensation
    field_compensated = None  # combined field of the compensated intensity and phase

    # filtered - to be done!
    field_filtered = None  # field after filtering (not yet implemented)

    # finished fields
    intensity_reconstructed = None  # intensity distribution of reconstructed field (same as compensated if no filtering is done
    field_reconstructed = None  # final reconstructed electromagnetic field (compensated intensity + phase)
    phase_map = None  # unwrapped phase map of the compensated, propagated hologram
    height_profile = None  # final height profile of the Hologram (reconstructed field)

    def __init__(self, hologram: Hologram, reference: ReferenceHologram = None,
                 dhm_parameter: dict = None, material_parameter: dict = None,
                 logger=None, saving_path=None):
        if logger is not None:
            super().__init__(logger)
        else:
            super().__init__()
            self.logger = get_logger()
        self.hologram = hologram
        self.params = dhm_parameter if dhm_parameter is not None else {}
        self.params.update({"MaterialDictionary": material_parameter} if material_parameter is not None else {})
        self.reconstruction_dict = {}
        try:
            self.refractiveIndex = material_parameter[
                "refractiveIndex"] if material_parameter is not None else self.refractive_index_default
        except Exception as e:
            self.logger.error(f"Error {e} occurred while trying to access 'refractiveIndex' of the material.")
            raise e

        if reference is not None:
            self.background = reference
            self.background_available = True
        else:
            self.background_available = False

        self._validate_input()
        self.plotter = DHMPlotter(img_path=saving_path)

    def plot_reconstruction(self, show_plot=False, save_single=False, cmap="gray", title=None,
                            mode: Literal["short", "full"] = "short",
                            compensation=True):
        if not self.plotter.img_save_path:
            show_plot = True

        if mode.lower() == "short":
            self.plotter.plot_reconstruction_short(self, show_plot=show_plot, save_single=save_single, cmap=cmap,
                                                   save_title=title, compensation=compensation)
        elif mode.lower() == "full":
            self.plotter.plot_full_reconstruction_process(self, show_plot=show_plot, save_single=save_single, cmap=cmap,
                                                          save_title=title)
        else:
            raise ValueError("Plotting mode can only be 'short' reconstruction or 'full' reconstruction.")

    @property
    def pixel_pitch(self):
        return self.params['pixelPitch']

    def _validate_input(self):
        self.logger.debug(f"Validating input for Reconstruction process")  # ToDo name ändern.
        assert isinstance(self.hologram, Hologram), "Hologram must be of type Hologram."
        # todo what to do if there is not background and no compensation should be done? - hotfix 13.01.26 comment
        # assert isinstance(self.background, ReferenceHologram), "ReferenceHologram must be of type Hologram"
        assert all(key in self.params for key in self.required_dhm_keys), \
            f"Missing required parameter(s): {self.required_dhm_keys - self.params.keys()}"

    def run(self, hologram: Hologram = None, background_hologram: ReferenceHologram = None, *,
            prop_dist=None, propagate=True,
            compensate=True, compensation_mode: Literal["Background", "ZernikePolynomial"] = "Background",
            filtering=False, filter_applied: list = None,  # todo think of a better name for the list
            refractive_index=None,
            mode="print",
            phase_unwrapping_method="Fast 2D",
            propagation_method="angularSpectrum") -> np.ndarray | tuple[Any, Any]:
        # ToDo: check if all e-fields are available even if propagate or compensation = False
        """
        Full reconstruction:

        Reconstruction Hologram + Reconstruction Background
        Propagation Hologram
                -- if no propagation -> set propagate = False !
        Compensation for Aberrations (intensity and phase independent)
                -- if no compensation -> set compensate = False
        Phase unwrapping
                -- different methods are available. Default - Fast 2D phase unwrapping using non-continuous path
        Filtering (not yet implemented)
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
            if hasattr(self, 'refractiveIndex'):
                refractive_index = self.refractiveIndex
            else:
                refractive_index = self.refractive_index_default

        if prop_dist is None and propagate:
            prop_dist = self.params['propagationDistance']

        self.logger.debug(f"Hologram processing with {mode} mode."
                          f"Refractive index: {refractive_index}"
                          f"Propagation Distance: {prop_dist}"
                          f"Compensation: {compensate}")

        # Reconstruction of Hologram
        holo_field = hologram.reconstruct(force=True)

        # Propagation of the electrical field to the focal plane
        if propagate:
            self.logger.info(f"Starting propagation of reconstructed field with {propagation_method} method.")
            self.field_propagated = self.propagate(field=holo_field, distance=prop_dist,
                                                   propagation_method=propagation_method)
            # todo if propagate field propagated cannot be accessed
        else:
            self.field_propagated = holo_field

        # Future ToDo: Aberration compensation using zernike polynom or other numerical methods
        # Aberration Compensation of Optics with Background image
        if compensate:
            if compensation_mode == "Background":
                self.logger.info(f"Starting compensation with the background field...")
                # Reconstruction of Background
                background_field = background_hologram.reconstruct()

                self.compensate(original=self.field_propagated, reference=background_field)
                # Note: recalculated field is not correct (while printing)

                self.logger.info(f"Starting phase unwrapping with {phase_unwrapping_method} method.")
                if mode.lower() == "print":
                    self.phase_map = self.phase_unwrapping(self.phase_compensated, method=phase_unwrapping_method)
                    self.intensity_reconstructed = self.intensity_compensated_db
                elif mode.lower() == "developed":
                    self.phase_map = self.phase_unwrapping(self.phase_compensated, method=phase_unwrapping_method)
                    self.intensity_reconstructed = self.intensity_compensated_db
            elif compensation_mode == "ZernikePolynomial":
                self.logger.warning("ZernikePolynomial not implemented yet.")
                # self.logger.info(f"Starting compensation with the zernike polynomials...") # todo future
                raise NotImplementedError("ZernikePolynomial not yet implemented.")
            else:
                raise NotImplementedError(f"Compensation method {compensation_mode} not implemented.")
        else:
            self.phase_map = self.phase_unwrapping(self.phase(self.field_propagated), method=phase_unwrapping_method)
            self.intensity_reconstructed = self.intensity(self.field_propagated, mode="db")

        # Filtering
        # ToDo: Filtering needs implementation of algorithms
        if filtering:
            if filter_applied is None:
                warnings.warn("No filter selected! Please select filter using the variable 'filter_applied'.")
                self.logger.info("No filter applied.")
                return
            self.logger.info("Start applying filters.")
            for i in range(len(filter_applied)):
                filter_name = filter_applied[i]
                if filter_name not in self.available_filters:
                    raise NotImplementedError(f"Filter {filter_name} not implemented.")
                if filter_name.lower() == "gauss" or filter_name.lower() == "gaussian":
                    pass
                elif filter_name.lower() == "median":
                    pass
                elif filter_name.lower() == "hm2f" or filter_name.lower() == "hybrid mean-median filter":
                    pass
                elif filter_name.lower() == "butterworth":
                    pass
        else:
            self.logger.info("No filter applied.")

        self.logger.info(f"Converting unwrapped phase to height map using refractive index=\
                        {refractive_index if not None else self.refractive_index_default}.")
        self.height_profile = self.phase_to_height(self.phase_map, n_resin=refractive_index)

        self.reconstruction_dict = {
            "Field propagated": propagate,
            "Field compensated": compensate,
            "Compensation method": "Background image",  # no other method until now.
            "propagationDistance": prop_dist,
            "propagationMethod": propagation_method,
            "phaseUnwrappingMethod": phase_unwrapping_method,
            "refractiveIndex_used": refractive_index if not None else self.refractive_index_default,
            "usedFilter": "Filtering not yet implemented.",
            "DHM Dictionary": self.params
        }

    def propagate(self, field, distance, pixel_pitch: list[float] = None,
                  propagation_method: Literal["angularSpectrum", "Fresnel"] = "angularSpectrum"):
        if float(distance) == 0.0 or distance is None:
            return field
        if pixel_pitch is None:
            dx = self.params['pixelPitch'][0]
            dy = self.params['pixelPitch'][1]
        else:
            if isinstance(pixel_pitch, float) or isinstance(pixel_pitch, int):
                dx = dy = pixel_pitch
            else:
                dx = pixel_pitch[0]
                dy = pixel_pitch[1]
        if distance is None:
            distance = self.params['propagationDistance']

        if propagation_method == "angularSpectrum" and float(distance) != 0.0:
            propagated = angularSpectrum(field=field, z=distance, wavelength=self.params['wavelength'], dx=dx, dy=dy)
        elif propagation_method == "Fresnel":
            raise NotImplementedError("Fresnel propagation not yet implemented.")
        else:
            raise NotImplementedError(f"No propagation method {propagation_method} implemented.")

        self.logger.info("Propagation finished ...")
        return propagated

    def compensate_phase(self, original: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Compensation of spherical phase aberrations are possible by capturing a background image with the same imaging
        system and subtraction of the background from the image with the specimen in it.
        Research done by:
        Pietro Ferraro, Sergio De Nicola, Andrea Finizio, Giuseppe Coppola, Simonetta Grilli, Carlo Magro, and Giovanni
        Pierattini, "Compensation of the inherent wave front curvature in digital holographic coherent microscopy for
        quantitative phase-contrast imaging," Appl. Opt. 42, 1938-1946 (2003)
        https://doi.org/10.1364/AO.42.001938
        """
        compensated = original - reference
        return compensated

    def compensate(self, original: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Compensation of spherical phase aberrations are possible by capturing a background image with the same imaging
        system and subtraction of the background from the image with the specimen in it.
        Research done by:
        Pietro Ferraro, Sergio De Nicola, Andrea Finizio, Giuseppe Coppola, Simonetta Grilli, Carlo Magro, and Giovanni
        Pierattini, "Compensation of the inherent wave front curvature in digital holographic coherent microscopy for
        quantitative phase-contrast imaging," Appl. Opt. 42, 1938-1946 (2003)
        https://doi.org/10.1364/AO.42.001938
        """
        self.phase_compensated = self.phase(original) - self.phase(reference)
        self.intensity_compensated_linear = self.intensity(original, mode="linear") - self.intensity(reference,
                                                                                                     mode="linear")
        self.intensity_compensated_db = self.intensity(original, mode="db") - self.intensity(reference, mode="db")
        self.field_compensated = self.calculate_efield(intensity=self.intensity_compensated_linear,
                                                       intensity_is_db=False,
                                                       phase=self.phase_compensated)
        self.logger.info("Compensation finished ...")
        return self.field_compensated

    def phase_unwrapping(self, phase_wrapped, method: Literal["Fast 2D", "Kamui", "Numpy"] = "Fast 2D"):
        """
        Phase Unwrapping with different phase unwrapping algorithms.
        """
        if method == "Fast 2D":
            phase_unwrapped = phase_unwrapping_fast2d(phase_wrapped)
        # elif method == "Kamui":
        #     phase_unwrapped = phase_unwrapping_kamui_normal(phase_wrapped)
        elif method == "Numpy":
            phase_unwrapped = phase_unwrapping_numpy(phase_wrapped)
        else:
            raise ValueError(f"Unsupported phase unwrapping method: {method}. "
                             f"Possible unwrapping methods are 'Fast 2D' and 'Kamui', 'Numpy'.")
        self.logger.info("Phase unwrapping finished ...")
        return phase_unwrapped

    def phase_to_height(self, phase, n_resin=None):
        """
        Height reconstruction based on the theoretical investigation by Nguyen et al. (2016)
        Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and Rola Aylo, "Accurate quantitative phase
        digital holographic microscopy with single- and multiple-wavelength telecentric and nontelecentric
        configurations,"
        Appl. Opt. 55, 5666-5683 (2016)
        http://dx.doi.org/10.1364/AO.55.005666
        """
        n_air = 1
        if n_resin is None:
            n_resin = self.refractive_index_default
        delta_n = n_resin - n_air  # change in refractive index - only approximate values
        height_profile = self.params['wavelength'] * phase / (2 * np.pi * delta_n)
        return height_profile

    def get_dict(self):
        return self.reconstruction_dict if not {} else self.params
