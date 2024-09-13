import numpy as np
from skimage.restoration import unwrap_phase
from offaxisholo.postprocessor import HologramPostProcessor
from offaxisholo.hologram_class import Hologram
from offaxisholo.plotter import DHMPlotter

"""
Coordinates the entire reconstruction process, ensuring that all necessary steps (FFT, filtering, compensation) are 
properly executed.
"""

class HologramReconstructor(DHMPlotter):
    def __init__(self, hologram: Hologram, processor: HologramPostProcessor, reference: Hologram = None,
                 propagation_distance=0):
        super().__init__()
        self.hologram = hologram
        self.processor = processor
        if reference is not None:
            self.background = reference
            self.background_available = True
        else:
            self.background_available = False

        self.wavelength = self.hologram.wavelength
        self.pixel_pitch = self.hologram.pixel_pitch
        self.propagation_distance = propagation_distance  #Noch ändern, wenn man herausgefunden hat was diese distanz ist
        self.n_resin = 1.5  # ToDO hier import nochmal überarbeiten

        self.phase_compensated = None
        self.height_profile = None

    def run(self, prop_dist=None) -> np.ndarray:
        """Full reconstruction pipeline including filtering and compensation."""
        # Get the unwrapped phase
        if not self.hologram.finished_reconstruction:
            phase_unwrapped = self.reconstruct_phase(hologram=self.hologram, propagation_distance=prop_dist)
        else:
            phase_unwrapped = self.hologram.phase_unwrapped
        if not self.background.finished_reconstruction:
            phase_unwrapped_reference = self.reconstruct_phase(hologram=self.background, propagation_distance=prop_dist)
        else:
            phase_unwrapped_reference = self.background.phase_unwrapped
        # Aberration Compensation of Optics with Background image
        compensated = self.compensate(original=phase_unwrapped, reference=phase_unwrapped_reference)
        # Filering of the phase
        # ToDo: Filtering needs rework or postprocessor needs rework
        # filtered = self.processor.filter(compensated)
        # self.height_profile = self.phase_to_height(filtered)
        # return filtered
        return compensated

    def reconstruct_phase(self, hologram, propagation_distance=None):
        # Step 1: Reconstruction of electrical field
        reconstructed = hologram.reconstruct()
        # Step 2: Propagation of E-field into focal plane
        if propagation_distance is None:
            propagated = self.propagate(field=reconstructed, distance=self.propagation_distance)
        else:
            propagated = self.propagate(field=reconstructed, distance=propagation_distance)
        # Step 3: Unwrapping phase
        propagated_phase = hologram.phase(propagated)
        phase_unwrapped = self.phase_unwrapping(propagated_phase)
        self.hologram.set_fullreconstruction(re_field=propagated, phase_unwrapped=phase_unwrapped)
        return phase_unwrapped

    def compensate(self, original: np.ndarray, reference: np.ndarray) -> np.ndarray:
        assert self.background_available == True
        self.phase_compensated = original - reference
        return self.phase_compensated

    def propagate(self, field, distance, pixel_pitch=None):
        if pixel_pitch is None:
            dx=self.pixel_pitch[0]
            dy=self.pixel_pitch[1]
        else:
            dx=pixel_pitch[0]
            dy=pixel_pitch[1]
        wv = self.wavelength
        propagated = self.angularSpectrum(field=field, z=distance, wavelength=wv, dx=dx, dy=dy)
        return propagated

    def angularSpectrum(self, field, z, wavelength, dx, dy):
        """
        # Function to diffract a complex field using the angular spectrum approximation
        # Inputs:
        # field - complex field
        # z - propagation distance
        # wavelength - wavelength
        # dx,dy - pixel pitch
        """
        field = np.array(field)
        # sanity check
        assert len(field.shape) == 2, "2D hologram image required!"
        assert field.shape[0] == field.shape[1], "Quadratic hologram image required!"
        assert field.shape[0] % 2 == 0, "Hologram image with even dimensions required!"

        M, N = field.shape
        x = np.arange(0, N, 1)  # array x
        y = np.arange(0, M, 1)  # array y
        X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

        dfx = 1 / (dx * M)
        dfy = 1 / (dy * N)

        spectrum = np.fft.fftshift(field)
        spectrum = np.fft.fft2(spectrum)
        spectrum = np.fft.fftshift(spectrum)

        phase = np.exp2(
            1j * z * np.pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

        tmp = spectrum * phase

        field_prop = np.fft.ifftshift(tmp)
        field_prop = np.fft.ifft2(field_prop)
        field_prop = np.fft.ifftshift(field_prop)

        return field_prop

    def phase_unwrapping(self, phase_wrapped):
        """
        This phase unwrapping algorithm is based on:
        Miguel Arevallilo Herráez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat, "Fast two-dimensional
        phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path,"
        Appl. Opt. 41, 7437-7444 (2002)
        https://opg.optica.org/ao/abstract.cfm?URI=ao-41-35-7437
        """
        phase_unwrapped = unwrap_phase(phase_wrapped)
        return phase_unwrapped


    def phase_to_height(self, phase):
        """
        Height reconstruction based on the theoretical investigation by Nguyen et al.
        Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and Rola Aylo, "Accurate quantitative phase
        digital holographic microscopy with single- and multiple-wavelength telecentric and nontelecentric
        configurations,"
        Appl. Opt. 55, 5666-5683 (2016)
        http://dx.doi.org/10.1364/AO.55.005666
        """
        n_air = 1
        delta_n = n_air - self.n_resin  # change in refractive index - only approximate values
        height_profile = self.wavelength * phase / (2 * np.pi * delta_n)
        return height_profile

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ToDo Überarbeitung dieses Abschnittes

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically
        #  determine a subfolder /img/ for the images - maybe done in the future for the complete structure class

    def evaluate(self, background_hologram=None, save_img=False, compensate=True):
        """
        All-in-one method for investigating a taken hologram.
        If save_img = True , then the plotted images will be safed to self.img_save_path
                                - set the path with set_save_path(path)
        """
        # ToDo Title der Auswertungen ändern.
        # ToDo save_img und savepath überarbeiten

        if save_img and self.img_save_path is None:
            raise Exception("A path for saving the images is needed! \nUse the method set_save_path for this purpose.")
        if background_hologram is None and compensate is True:
            assert self.background_available == True
        else:
            self.calc_background(background_hologram)

        self.run(compensation=compensate)

        # Plotting of spectrum - normal, shifted, masked (2 different r)
        self.plotImage(self.intensity(self.spectrum_not_shifted), "Intensity not shifted", save=save_img)
        self.plotImage(self.intensity(self.spectrum_shifted), "Intensity shifted", save=save_img)
        self.plotImage(self.intensity(self.spectrum_masked), "Intensity shifted and masked", save=save_img)
        # Plotting of field and phase after masking and iFFT
        # Plotting of field and phase after compensation
        self.plotImage(self.intensity(self.reconstructed_field), "intensity after prop", save=save_img)
        self.plotImage(self.phase(self.reconstructed_field), "phase after prop, before unwrapping", save=save_img)
        # Plotting of unwrapped images
        self.plotImage(self.phase_unwrapped, "Phase after unwrapping", save=save_img)
        if compensate:
            self.plotImage(self.phase_compensated, "Phase after compensation", save=save_img)
            self.plot_height(-self.height_profile, title="Height profile of the Structure", save=save_img)

        # ToDo Add variable for this evaluation
        # self.plotImage(self.intensity(self.reconstructed_field), "intensity before prop", save=save_img)
        # self.plotImage(self.phase(self.reconstructed_field), "phase before prop and unwrapping", save=save_img)
        # Plotting of unwrapped images BEFORE PROPAGATION
        # self.plotImage(phase_unwrapped_rmax_not_prop, "phase after unwrapping BEFORE prop (correct r)", save=save_img)
        # self.plotImage(phase_unwrapped_rcalc_not_prop, "phase after unwrapping BEFORE prop (false r)", save=save_img)
