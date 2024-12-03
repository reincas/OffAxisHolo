import os.path
from typing import Any
import json
import numpy as np
from numpy import save
from numpy.lib.scimath import sqrt

from skimage.restoration import unwrap_phase
from .postprocessor import HologramPostProcessor
from .hologram_class import Hologram, ReferenceHologram
from .plotter import DHMPlotter

"""
Coordinates the entire reconstruction process, ensuring that all necessary steps (FFT, filtering, compensation) are 
properly executed.
"""


class HologramReconstructor(DHMPlotter):
    def __init__(self, hologram: Hologram, reference: ReferenceHologram = None,
                 processor: HologramPostProcessor = None):
        super().__init__()
        self.hologram = hologram
        self.processor = processor
        if reference is not None:
            self.background = reference
            self.background_available = True
        else:
            self.background_available = False

        # self.wavelength = self.hologram.wavelength # ToDo: ÄNDERN, da es falsch ist
        self.wavelength = 0.6749e-6
        self.pixel_pitch = self.hologram.pixel_pitch
        self.propagation_distance = self.hologram.propagation_distance
        self.n_resin = self.hologram.n_resin

        # All possible fields
        self.initial_field = None  # initial field after fft - shift 1st order - ifft
        self.field_propagated = None  # field after propagation
        self.phase_compensated = None  # phase after compensation
        self.intensity_compensated_linear = None  # intensity after compensation
        self.intensity_compensated_db = None  # intensity after compensation
        self.field_compensated = None  # combined field of the compensated intensity and phase
        self.field_filtered = None  # field after filtering (not yet implemented)
        self.intensity_reconstructed = None  # intensity distribution of reconstructed field (same as compensated if no filtering is done
        self.field_reconstructed = None  # final reconstructed electromagnetic field (compensated intensity + phase)
        self.phase_map = None  # unwrapped phase map of the compensated, propagated hologram
        self.height_profile = None  # final height profile of the Hologram (reconstructed field)

    def run(self, hologram: Hologram = None, background_hologram: ReferenceHologram = None, prop_dist=None,
            propagate=True,
            compensate=True) -> np.ndarray | tuple[Any, Any]:
        """
        Full reconstruction:

        Reconstruction Hologram + Reconstruction Background
        Propagation Hologram
                -- if no propagation -> set propagate = False or prop_dist = 0 !
        Compensation for Aberrations (intensity and phase independent)
                -- if no compensation -> set compensate = False
        Phase unwrapping
        Filtering (not yet implemented)
        """

        return_field = None  # field which will be returned
        if hologram is None:
            hologram = self.hologram
        if background_hologram is None:
            if self.background_available:
                background_hologram = self.background
        if prop_dist is None:
            prop_dist = self.propagation_distance

        # Reconstruction of Hologram
        holo_field = hologram.reconstruct()
        return_field = holo_field

        # Propagation of the electrical field to the focal plane
        if propagate:
            self.field_propagated = self.propagate(field=return_field, distance=prop_dist)
            return_field = self.field_propagated

        # Aberration Compensation of Optics with Background image
        if compensate:
            # Reconstruction of Background
            if background_hologram is not None:
                background_field = background_hologram.reconstruct()
            else:
                raise NotImplementedError("No Background image found.")

            return_field = self.compensate(original=return_field, reference=background_field)

        # Filtering
        # ToDo: Filtering needs rework or postprocessor needs rework
        # return_field = self.processor.filter(return_field)

        self.phase_map = self.phase_unwrapping(self.phase(return_field))
        self.intensity_reconstructed = self.intensity(return_field)
        # save necessary fields in hologram
        hologram.set_full_reconstruction(re_field=return_field,
                                         propagated_field=self.field_propagated if propagate else None,
                                         phase_unwrapped=self.phase_map)

        self.height_profile = self.phase_to_height(return_field)
        self.field_reconstructed = return_field
        return return_field

    def propagate(self, field, distance, pixel_pitch: list[float] = None):
        PROPAGATION_ALGORITHM = "Angular Spectrum"  # todo: follow up implementation of different algorithms

        if distance == 0 or distance is None:
            return field
        if pixel_pitch is None:
            if isinstance(self.pixel_pitch, list) or isinstance(self.pixel_pitch, tuple):
                dx = self.pixel_pitch[0]
                dy = self.pixel_pitch[1]
            else:
                dx = dy = self.pixel_pitch
        else:
            if isinstance(pixel_pitch, float) or isinstance(pixel_pitch, int):
                dx = dy = pixel_pitch
            else:
                dx = pixel_pitch[0]
                dy = pixel_pitch[1]
        if distance is None:
            distance = self.propagation_distance
        wv = self.wavelength
        if PROPAGATION_ALGORITHM.lower() == "angular spectrum" and int(distance) != 0:
            propagated = self.angularSpectrum(field=field, z=distance, wavelength=wv, dx=dx, dy=dy)
        else:
            propagated = field

        return propagated

    def angularSpectrum(self, field, z, wavelength, dx, dy):
        '''
        # Function to diffract a complex field using the angular spectrum approximation
        # Inputs:
        # field - complex field
        # z - propagation distance
        # wavelength - wavelength
        # dx,dy - sampling pitches
        '''

        field = np.array(field)
        M, N = field.shape
        x = np.arange(0, N, 1)  # array x
        y = np.arange(0, M, 1)  # array y
        X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

        dfx = 1 / (dx * M)
        dfy = 1 / (dy * N)

        field_spec = np.fft.fftshift(field)
        field_spec = np.fft.fft2(field_spec)
        field_spec = np.fft.fftshift(field_spec)

        phase = np.exp(
            1j * z * np.pi * sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

        tmp = field_spec * phase

        out = np.fft.ifftshift(tmp)
        out = np.fft.ifft2(out)
        out = np.fft.ifftshift(out)
        return out

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
        # ToDo change compensate, so it takes the e-field and calculate the correct compensation and returns a field
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
                                                       phase=self.phase_compensated)
        self.plotImage(self.phase_compensated)
        self.plotImage(self.phase(self.field_compensated))
        self.plotImage(self.intensity_compensated_linear)
        self.plotImage(self.intensity_compensated_db)
        return self.field_compensated

    def phase_unwrapping(self, phase_wrapped):
        """
        This phase unwrapping algorithm is based on:
        Miguel Arevallilo Herráez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat, "Fast two-dimensional
        phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path,"
        Appl. Opt. 41, 7437-7444 (2002)
        https://doi.org/10.1364/AO.41.007437
        """
        phase_unwrapped = unwrap_phase(phase_wrapped)
        return phase_unwrapped

    def phase_to_height(self, phase, n_resin=None):
        """
        Height reconstruction based on the theoretical investigation by Nguyen et al.
        Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and Rola Aylo, "Accurate quantitative phase
        digital holographic microscopy with single- and multiple-wavelength telecentric and nontelecentric
        configurations,"
        Appl. Opt. 55, 5666-5683 (2016)
        http://dx.doi.org/10.1364/AO.55.005666
        """
        n_air = 1
        if n_resin is None:
            n_resin = self.n_resin
        delta_n = n_resin - n_air  # change in refractive index - only approximate values
        height_profile = self.wavelength * phase / (2 * np.pi * delta_n)
        return height_profile

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ToDo Überarbeitung dieses Abschnittes

    def evaluate(self, hologram=None, background_hologram=None, save_img=False, compensate=True, propagate=True,
                 prop_dist=None):
        """
        All-in-one method for investigating a taken hologram.
        If save_img = True , then the plotted images will be safed to self.img_save_path
                                - set the path with set_save_path(path)

        Plotting of all steps:
        recording of hologram
        holo = hologram.data
        field to spectrum
        spectrum_normal = hologram.spectrum_not_shifted
        shift spectrum
        spectrum_shifted = hologram.spectrum_shifted
        mask spectrum (to get the -1 diffraction order)
        spectrum_shifted_masked = hologram.spectrum_masked
        inverse fouriertransform
        reconstructed_int = hologram.reconstructed_intensity
        reconstructed_phase = hologram.reconstructed_phase
        propagation
        propagated_int = hologram.propagated_intensity
        propagated_phase = hologram.propagated_phase
        phase unwrapping
        unwrapped_phase = hologram.phase_unwrapped
        aberration compensation
        compensated_phase = self.phase_compensated
        height_profile = self.phase_to_height(compensated_phase)
        """
        # ToDo Title der Auswertungen ändern.
        # ToDo save_img und savepath überarbeiten

        if save_img and self.img_save_path is None:
            raise Exception("A path for saving the images is needed! \nUse the method set_save_path for this purpose.")
        if background_hologram is None and compensate is True:
            assert self.background_available

        # make sure that an instance of hologram does exist
        if hologram is None:
            hologram = self.hologram
        if background_hologram is None:
            background_hologram = self.background

        self.run(hologram=hologram, background_hologram=background_hologram, propagate=propagate, compensate=compensate,
                 prop_dist=prop_dist)

        ######### ---- PLOTTING ---- ###########
        self.plotImage(hologram.data, "Recorded Hologram", save=save_img)
        # Plotting of spectrum - normal, shifted, masked
        self.plotImage(self.hologram.intensity(hologram.spectrum_not_shifted), "Angular spectrum not shifted",
                       save=save_img)
        self.plotImage(self.hologram.intensity(hologram.spectrum_shifted), "Angular spectrum shifted",
                       save=save_img)
        self.plotImage(self.hologram.intensity(hologram.spectrum_masked), "Angular spectrum shifted and masked",
                       save=save_img)

        # Plotting of field and phase after masking and iFFT
        self.plotImage(hologram.reconstructed_intensity, "Intensity before propagation", save=save_img)
        self.plotImage(hologram.reconstructed_phase, "Phase before propagation", save=save_img)

        # Plotting of field and phase after propagation
        self.plotImage(hologram.propagated_intensity, "Intensity after propagation", save=save_img)
        self.plotImage(hologram.propagated_phase, "Phase after propagation", save=save_img)

        tmp = hologram.reconstructed_intensity - hologram.propagated_intensity
        self.plotImage(tmp, "Intensity difference", save=save_img)
        # Plotting of unwrapped images
        self.plotImage(hologram.phase_unwrapped, "Phase after unwrapping", save=save_img)
        if compensate:
            self.plotImage(self.phase_compensated, "Phase after compensation", save=save_img)
            self.plot_height(self.phase_to_height(self.phase_compensated), title="Height profile of the Structure",
                             save=save_img)
