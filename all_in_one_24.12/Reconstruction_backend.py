import cv2 as cv
import scidatacontainer
from scipy.ndimage import maximum_filter
import os
from matplotlib import pyplot as plt
from numpy import save

import os.path
from typing import Any
import json
import numpy as np
from numpy import save
from numpy.lib.scimath import sqrt

from skimage.restoration import unwrap_phase

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
            propagate=True, compensate=True, mode="print"
            ) -> np.ndarray | tuple[Any, Any]:
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
        holo_field = hologram.reconstruct(propagate=False, force=True)
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

            return_field = self.compensate(original=return_field, reference=background_field, tmp=prop_dist)
            # ToDo: recalculated field is not correct (while printing)
            # ToDo: saving the field as np.array has to be reworked! field is not correct
            if mode.lower() == "print":
                self.phase_map = self.phase_unwrapping(self.phase_compensated)
                self.intensity_reconstructed = self.intensity_compensated_db
            elif mode.lower() == "developed":
                self.phase_map = self.phase_unwrapping(self.phase(return_field))
                self.intensity_reconstructed = self.intensity(return_field)
        # Filtering
        # ToDo: Filtering needs rework or postprocessor needs rework
        # return_field = self.processor.filter(return_field)

        # save necessary fields in hologram
        # hologram.set_full_reconstruction(re_field=return_field,
        #      propagated_field=self.field_propagated if propagate else None, # field propagated is atm commented and therefore None
        #      phase_unwrapped=self.phase_map)

        self.height_profile = self.phase_to_height(self.phase_map)
        self.field_reconstructed = return_field

        # temporary plotting
        # path_base = r"C:\Users\hanne\Desktop\tmp\img_reconstruction"
        # folder = "spektrum"
        # eval_path = os.path.join(path_base, folder)
        # os.makedirs(eval_path, exist_ok=True)
        # self.plotImage(self.intensity(hologram.spectrum_not_shifted, mode="db"), title="1. Spektrum", save=True, save_path=eval_path)
        # self.plotImage(self.phase(hologram.spectrum_not_shifted), title="1. Spektrum - phase", save=True, save_path=eval_path)
        # self.plotImage(self.intensity(hologram.spectrum_shifted, mode="db"), title="2. Rolled spectrum", save=True, save_path=eval_path)
        # self.plotImage(self.phase(hologram.spectrum_shifted), title="2. Rolled spectrum - phase", save=True, save_path=eval_path)
        # self.plotImage(self.intensity(hologram.spectrum_masked, mode="db"), title="3. Masked spektrum", save=True, save_path=eval_path)
        # self.plotImage(self.phase(hologram.spectrum_masked), title="3. Masked spektrum - phase", save=True, save_path=eval_path)
        # self.plotImage(self.intensity(hologram.reconstructed_field, mode="db"), title="4.1 Reconstructed intensity", save=True, save_path=eval_path)
        # self.plotImage(self.phase(hologram.reconstructed_field), title="4.1 Reconstructed phase", save=True, save_path=eval_path)
        # path_base = r"C:\Users\hanne\Desktop\tmp\img_reconstruction"
        # folder = f"{prop_dist}"
        # eval_path = os.path.join(path_base, folder)
        # os.makedirs(eval_path, exist_ok=True)
        # self.plotImage(self.phase_map, title="Phase Unwrapped", save=True, save_path=eval_path)
        # self.plotImage(self.height_profile, title="Height Profile", save=True, save_path=eval_path, cmap="coolwarm")
        # self.plot_height(self.height_profile, title=f"Height profile 3d", save=True, save_path=eval_path)
        return return_field

    def propagate(self, field, distance, pixel_pitch: list[float] = None):
        PROPAGATION_ALGORITHM = "Angular Spectrum"  # todo: follow up implementation of different algorithms

        if float(distance) == 0.0 or distance is None:
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
        if PROPAGATION_ALGORITHM.lower() == "angular spectrum" and float(distance) != 0.0:
            propagated = self.angularSpectrum(field=field, z=distance, wavelength=wv, dx=dx, dy=dy)
        else:
            propagated = field

        # temporary plotting
        # path_base = r"C:\Users\hanne\Desktop\tmp\img_reconstruction"
        # folder = f"{distance}"
        # eval_path = os.path.join(path_base, folder)
        # os.makedirs(eval_path, exist_ok=True)
        # self.plotImage(self.intensity(propagated, mode="db"), title=f"2. Propagated field (d={distance} m)", save=True, save_path=eval_path)
        # self.plotImage(self.phase(propagated), title=f"2. Propagated field (d={distance} m) - phase", save=True, save_path=eval_path)
        return propagated

    def angularSpectrum(self, field, z, wavelength, dx, dy):
        """
        Angular spectrum propagation supporting both positive and negative distances
        """
        M, N = field.shape
        x = np.arange(0, N)
        y = np.arange(0, M)
        X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

        dfx = 1 / (dx * M)
        dfy = 1 / (dy * N)

        # Calculate frequency components
        fx = X * dfx
        fy = Y * dfy

        # Wave number
        k = 2 * np.pi / wavelength

        # Transfer function phase
        kz = sqrt(k ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2 + 0j)

        # Handle evanescent waves
        kz = np.real(kz) + 1j * np.abs(np.imag(kz))

        # Fourier transform and apply propagator
        field_spec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
        field_spec *= np.exp(1j * kz * z)

        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_spec)))

    def angularSpectrum_old(self, field, z, wavelength, dx, dy):
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
        compensated = original - reference
        return compensated

    def compensate(self, original: np.ndarray, reference: np.ndarray, tmp=None) -> np.ndarray:
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
        # temporay reconstruction
        # path = r"C:\Users\hanne\Desktop\tmp\img_reconstruction"
        # eval_path = os.path.join(path, str(tmp), "compensate")
        # os.makedirs(eval_path, exist_ok=True)
        # self.plotImage(self.phase(original), title="Phase of original field", save=True, save_path=eval_path)
        # self.plotImage(self.phase(reference), title="Phase of background field", save=True, save_path=eval_path)
        # self.plotImage(self.phase_compensated, title="Phase compensated", save=True, save_path=eval_path)
        # self.plotImage(self.phase(self.field_compensated), title="phase of field compensated (calculated)", save=True, save_path=eval_path)
        # self.plotImage(self.intensity_compensated_linear, title="intensity linear", save=True, save_path=eval_path)
        # self.plotImage(self.intensity_compensated_db, title="intensity db", save=True, save_path=eval_path)
        # #
        # phase1 = self.phase_unwrapping(self.phase_compensated)
        # phase2 = self.phase_unwrapping(self.phase(self.field_compensated))
        # test = phase1-phase2
        #
        # phase_original = self.phase_unwrapping(self.phase(original))
        # phase_reference = self.phase_unwrapping(self.phase(reference))
        # self.plotImage(phase_original, title="Unwrapped phase - original phase", save=True, save_path=eval_path)
        # self.plotImage(phase_reference, title="Unwrapped phase - reference phase", save=True, save_path=eval_path)
        # self.plotImage(phase_original-phase_reference, title="compensated phase after unwrapping", save=True, save_path=eval_path)
        # self.plotImage(phase1, title="Unwrapped phase - phase comp", save=True, save_path=eval_path)
        # self.plotImage(phase2, title="Unwrapped phase - phase(field)", save=True, save_path=eval_path)
        # self.plotImage(test, title="Difference Unwrapped phase", save=True, save_path=eval_path)
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

class DHMPlotter:
    def __init__(self, img_path=None):
        self.img_save_path = img_path
        # ToDo: Implement in the functions below what to do if img_ave_path is none

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically
        #  determine a subfolder /img/ for the images - maybe done in the future for the complete structure_dhm class

    def intensity(self, complex_field, mode='linear'):
        """
        Calculate intensity from complex field

        Parameters:
        -----------
        inp : ndarray
            Input complex field
        mode : str
            'linear' for |E|^2
            'db' for intensity in decibels

        Returns:
        --------
        intensity : ndarray
            Calculated intensity
        """
        # Calculate absolute square
        intensity = np.abs(complex_field) ** 2
        eps = 1e-12

        if mode.lower() == 'db':
            # Convert to dB scale (10 log10 for intensity)
            intensity = 10 * np.log10(intensity + eps)
        return intensity

    def phase(self, inp):
        """
        Calculate phase from complex field

        Parameters:
        -----------
        inp : ndarray
            Input complex field
        unwrap : bool
            Whether to unwrap the phase

        Returns:
        --------
        phase : ndarray
            Calculated phase in radians
        """
        phase = np.angle(inp)
        return phase

    def calculate_efield(self, intensity, phase, intensity_is_db=False):
        """
        Calculate complex electromagnetic field from intensity and phase.

        Parameters:
        -----------
        intensity : numpy.ndarray
            2D array of intensity values or intensity differences
        phase : numpy.ndarray
            2D array of phase values in radians
        intensity_is_db : bool
            Whether intensity is in dB scale

        Returns:
        --------
        numpy.ndarray
            Complex 2D array of E-field
        """
        if intensity.shape != phase.shape:
            raise ValueError("Intensity and phase arrays must have same shape")

        if np.any(np.isnan(intensity)) or np.any(np.isnan(phase)):
            raise ValueError("Input arrays contain NaN values")

        # if np.any(np.isinf(intensity)) or np.any(np.isinf(phase)):
        #     raise ValueError("Input arrays contain infinite values")

        if intensity_is_db:
            intensity = 10 ** (intensity / 10)

        # Calculate complex field (no negative intensity check)
        amplitude = np.sqrt(np.abs(intensity)) * np.sign(intensity)
        efield = amplitude * np.exp(1j * phase)

        return efield

    def intensity_stable(self, complex_field, mode='linear'):
        """Calculate intensity with numerical stability"""
        # Use log(abs()) instead of abs()^2 for better numerical stability
        intensity = np.log(np.abs(complex_field))
        intensity = np.exp(2 * intensity)  # Equivalent to abs()^2 but more stable

        eps = 1e-12
        if mode.lower() == 'db':
            intensity = 10 * np.log10(intensity + eps)
        return intensity

    def calculate_efield_stable(self, intensity, phase, intensity_is_db=False):
        """Calculate E-field with numerical stability"""
        # Handle NaN and Inf before calculations
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=1e6, neginf=-1e6)
        phase = np.nan_to_num(phase, nan=0.0, posinf=np.pi, neginf=-np.pi)

        if intensity_is_db:
            intensity = np.clip(intensity, -100, 100)  # Prevent extreme values
            intensity = 10 ** (intensity / 10)

        amplitude = np.sqrt(np.abs(intensity)) * np.sign(intensity)
        return amplitude * np.exp(1j * phase)

    def compensate_stable(self, original, reference):
        """Compensate with scaling to prevent overflow
        ToDo: Remove stable versions
        """
        scale = max(np.max(np.abs(original)), np.max(np.abs(reference)))
        original_scaled = original / scale
        reference_scaled = reference / scale

        self.phase_compensated = self.phase(original_scaled) - self.phase(reference_scaled)
        self.intensity_compensated_linear = self.intensity(original_scaled, mode="linear") - \
                                            self.intensity(reference_scaled, mode="linear")
        self.intensity_compensated_db = self.intensity(original_scaled, mode="db") - \
                                        self.intensity(reference_scaled, mode="db")

        return self.calculate_efield(intensity=self.intensity_compensated_linear,
                                     intensity_is_db=False,
                                     phase=self.phase_compensated)

    def plotImage(self, img, title=None, save=False, save_path=None, cmap='viridis'):
        if save_path is None:
            save_path = self.img_save_path
        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(save_path, name)
                plt.imsave(save_path, img, cmap=cmap)
            else:
                save_path = os.path.join(save_path, f"picture{self.var_4_saving}.png")
                plt.imsave(save_path, img, cmap=cmap)
                self.var_4_saving += 1
            plt.close()
        else:
            if title == None:
                plt.imshow(img, cmap=cmap)
            else:
                plt.imshow(img, cmap=cmap)
                plt.title(title)
            plt.show()  # show image
        return

    def plot_height(self, height_profile, title=None, save=False, save_path=None, legend_bar=True, cmap='coolwarm', pixel_pitch=None):
        """
        Plotting of the reconstructed height profile. Make sure the dimensions of the height profile matches the
        dimensions of the hologram.

        height_profile:  Height profile of the image
        title:           Title of the image. If save then this will also be the name of the saved image.
        save:            Boolean if it should be saved. If False then it will be shown.
        legend_bar:      Boolean if the color bar should be shown.
        """
        if save_path is None:
            save_path = self.img_save_path
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Using linspace to generate exactly 1024 points in each direction
        if pixel_pitch is None:
            if isinstance(self.pixel_pitch, float):
                x_px_sz = y_px_sz = self.pixel_pitch
            else:
                x_px_sz = self.pixel_pitch[0]
                y_px_sz = self.pixel_pitch[1]
        else:
            x_px_sz = pixel_pitch
            y_px_sz = pixel_pitch

        X = np.linspace(0, (self.hologram.shape[1] - 1) * y_px_sz, self.hologram.shape[1])
        Y = np.linspace(0, (self.hologram.shape[0] - 1) * x_px_sz, self.hologram.shape[0])
        # Creating the meshgrid
        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, height_profile, cmap=cmap,
                               linewidth=0, antialiased=False)
        if title is not None:
            plt.title(title)

        if legend_bar:
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(save_path, name)
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
            else:
                save_path = os.path.join(save_path, f"3D_plot_{self.var_4_saving}.png")
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
                self.var_4_saving += 1
        else:
            plt.show()

    def save(self, path, name: str = None, data: np.ndarray = None):
        """
        Saves the numpy array of the reconstructed field.
        """
        # ToDo rework position of implementation
        if name is None:
            name = "field_reconstructed"
        INFORMATION_FILE = name + "_information.txt"
        INFORMATION_PATH = os.path.join(path, INFORMATION_FILE)
        SAVE_NAME = os.path.join(path, name + ".npy")

        if data is None:
            data = self.field_reconstructed
        information = {
            "Description": f"Reconstructed field of a DHM image.",
            "Wavelength": self.wavelength,
            "Pixel size": self.pixel_pitch,
            "Propagation distance": self.propagation_distance,
            "Refractive index": self.n_resin,
            "format": data.dtype.name,
        }
        with open(INFORMATION_PATH, 'w', encoding='utf-8') as file:
            file.write(json.dumps(information, sort_keys=True, indent=4))
        save(SAVE_NAME, data)



class Holo_Dummy(DHMPlotter):
    def __init__(self, dummy_mode=True):
        if dummy_mode:
            raise Warning("Object is used in Dummy mode!")
        super().__init__()
        # Initialization of parameter
        self.first_diffraction_order_pos = [0, 0]

        self.radius_mask = None
        self.radius_0_order = 0
        self.wavelength = 0
        self.pixel_pitch = []

        self.spectrum_not_shifted = None
        self.spectrum_shifted = None
        self.spectrum_propagated = None
        self.spectrum_masked = None
        self.reconstructed_field = None  # np.complex128  # ToDo: How to initialize this as type
        self.reconstructed_intensity = None
        self.reconstructed_phase = None

    def getField(self, spectrum) -> np.ndarray:
        """ Return complex field from centered spectrum. """
        field = np.fft.ifft2(np.fft.fftshift(spectrum))
        return field

    def getSpectrum(self, field) -> np.ndarray:
        FT = np.fft.fft2(field)
        spectrum = np.fft.fftshift(FT)
        return spectrum

    def propagate(self, field, distance, pixel_pitch: list[float] = None):
        """
        TEMPORARY PROPAGATE -- ORIGINALLY IN RECONSTRUCTOR!!
        """
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
            distance = 0  # ToDo: default distance??

        wv = self.wavelength
        if PROPAGATION_ALGORITHM.lower() == "angular spectrum" and float(distance) != 0.0:
            propagated = self.angularSpectrum(field=field, z=distance, wavelength=wv, dx=dx, dy=dy)
        else:
            propagated = field

        return propagated

    def angularSpectrum(self, field, z, wavelength, dx, dy):
        """
        Angular spectrum propagation supporting both positive and negative distances
        """
        M, N = field.shape
        x = np.arange(0, N)
        y = np.arange(0, M)
        X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

        dfx = 1 / (dx * M)
        dfy = 1 / (dy * N)

        # Calculate frequency components
        fx = X * dfx
        fy = Y * dfy

        # Wave number
        k = 2 * np.pi / wavelength

        # Transfer function phase
        kz = sqrt(k ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2 + 0j)

        # Handle evanescent waves
        kz = np.real(kz) + 1j * np.abs(np.imag(kz))

        # Fourier transform and apply propagator
        field_spec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
        field_spec *= np.exp(1j * kz * z)

        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_spec)))

    def holo2Field_propagation(self, holo, fx, fy, r, return_spectrum=False, propagation_distance=None,
                               pixel_pitch=None):
        """ Calculate the wave field from a given positive real valued hologram
        image based on the given spectral position of the first diffraction order
        relative to the zero order. A circular mask with the given radius is
        applied to the Fourier spectrum in order to extract the first order
        spectrum. """

        # Positive, real valued hologram
        if len(holo.shape) != 2:
            raise RuntimeError("2D hologram image required!")
        if np.min(holo) < 0:
            raise RuntimeError("Real positive hologram image required!")
        if holo.shape[0] % 2 != 0 or holo.shape[1] % 2 != 0:
            raise RuntimeError("Hologram image with even dimensions required!")
        # if holo.dtype != float
        # holo = holo.astype(np.float64)
        if propagation_distance is None:
            raise ValueError("Propagation distance has to be set!")

        # Spatial spectrum of the hologram
        spectrum = self.getSpectrum(holo)

        # Roll the given first order coordinates to the centre of the spectrum
        spectrum_rolled = self.rollImage(spectrum, fx, fy)

        # Propagate spectrum
        spectrum_propagated = self.propagate(field=spectrum_rolled, distance=propagation_distance,
                                             pixel_pitch=pixel_pitch)

        # Apply circular aperture with radius r
        spectrum_masked = self.circularMask(spectrum_propagated, r)

        # Calculate and return the wave field from the first order spectrum
        field = self.getField(spectrum_masked)

        if return_spectrum:
            return field, spectrum, spectrum_rolled, spectrum_propagated, spectrum_masked
        else:
            return field

    def holo2Field(self, holo, fx, fy, r, return_spectrum=False):
        """ Calculate the wave field from a given positive real valued hologram
        image based on the given spectral position of the first diffraction order
        relative to the zero order. A circular mask with the given radius is
        applied to the Fourier spectrum in order to extract the first order
        spectrum. """

        # Positive, real valued hologram
        if len(holo.shape) != 2:
            raise RuntimeError("2D hologram image required!")
        if np.min(holo) < 0:
            raise RuntimeError("Real positive hologram image required!")
        if holo.shape[0] % 2 != 0 or holo.shape[1] % 2 != 0:
            raise RuntimeError("Hologram image with even dimensions required!")
        # if holo.dtype != float
        # holo = holo.astype(np.float64)

        # Spatial spectrum of the hologram
        spectrum = self.getSpectrum(holo)

        # Roll the given first order coordinates to the centre of the spectrum
        spectrum_rolled = self.rollImage(spectrum, fx, fy)

        # Apply circular aperture with radius r
        spectrum_masked = self.circularMask(spectrum_rolled, r)

        # Calculate and return the wave field from the first order spectrum
        field = self.getField(spectrum_masked)

        if return_spectrum:
            return field, spectrum, spectrum_rolled, spectrum_masked
        else:
            return field

    def locateOrder(self, holo, size=16):
        """ Calculate the Fourier spectrum of the given positive real valued
        hologram image and return the spectral coordinates, the maximum spectral
        filter radius and the weight of the estimated first diffraction order
        peak. The global maximum after masking the zero and Nyquist frequencies
        is taken as first diffraction order. The size parameter is the smoothing
        radius and thus limits the density of local minima to be considered. The
        weight of the peak is between 0.0 and 1.0. """

        assert len(holo.shape) == 2, "2D hologram image required!"
        assert holo.shape[0] == holo.shape[1], "Quadratic hologram image required!"
        assert holo.shape[0] % 2 == 0, "Hologram image with even dimensions required!"
        assert np.min(holo) >= 0.0, "Real positive hologram image required!"

        # Get spectrum of the hologram image
        holo = holo.astype(np.float64)
        spectrum = np.fft.fftshift(np.fft.fft2(holo))
        N = spectrum.shape[0]

        # Blur and normalize the right half of the spectrum
        blurred = cv.GaussianBlur(np.abs(spectrum[:, :N // 2]), None, size)
        blurred /= blurred[0, 0]

        # Get indices of all local maxima in the spectrum
        maxmask = (maximum_filter(blurred, size=size) == blurred)
        points = np.unravel_index(np.nonzero(maxmask.ravel()), maxmask.shape)
        points = np.concatenate(points, axis=0).T

        # Strip all local maxima around the zero and the Nyquist frequency. This
        # strips the dominating zero order peak and many mirror artifacts.
        s = N // 4
        points = [(y, x) for y, x in points if abs(x % (2 * s) - s) < s - size // 2 and \
                  abs(y % (2 * s) - s) < s - size // 2]
        if not points:
            x, y, weight = None, None, 0.0

        else:
            # Take global maximum of the remaining points
            weights = [blurred[y, x] for y, x in points]
            y, x = points[np.argmax(weights)]
            x -= N // 2
            y -= N // 2
            weight = np.max(weights)

        # Done.
        return spectrum, x, y, weight

    def rollImage(self, img, x, y):
        """ Roll given image content so that point (x, y) becomes (0, 0). Wrap
        pixels at the image edges. Therefore, no information is lost.  For
        x = w//2 and y = h//2, the function is equivalent to np.fft.fftshift(img).
        """
        return np.roll(img, (-y, -x), axis=(0, 1))

    def circularMask(self, spectrum, r):
        """ Apply circular mask with given radius to the centered spectrum. """

        N = spectrum.shape[0]
        y, x = np.indices((N, N), dtype=float)
        x -= N // 2
        y -= N // 2
        r2 = x * x + y * y
        return np.where(r2 <= r * r, spectrum, 0.0)

    def _calc_radius_mask(self, fx, fy, r0, h, w):
        rmax = np.sqrt(fx ** 2 + fy ** 2) - r0
        rmax = min(rmax, abs(fx), w // 2 - abs(fx), abs(fy), h // 2 - abs(fy))
        return rmax


class Hologram(Holo_Dummy):
    def __init__(self, data: np.ndarray, dhm_parameter, first_diffraction_order_pos=None):
        super().__init__(dummy_mode=False)
        if isinstance(data, scidatacontainer.fileimage.PngFile):
            self.data = data.data
        else:
            self.data = data
        if len(self.data.shape) != 2:
            raise Exception(f"An Error occurred. 2D Hologram image required! Check data.")
        if self.data.shape[0] != self.data.shape[1]:
            raise Exception("Quadratic hologram image required!")

        assert isinstance(dhm_parameter, dict)
        self.radius_0_order = dhm_parameter["DC radius"]
        self.propagation_distance = dhm_parameter["propagation distance"]
        self.wavelength = dhm_parameter["wavelength"]
        self.pixel_pitch = dhm_parameter["pixel pitch"]
        self.n_resin = dhm_parameter["refractive index"]

        self.first_diffraction_order_pos = first_diffraction_order_pos
        # Starting the necessary functions
        if self.first_diffraction_order_pos is None:
            self.__locate_order()
        else:
            self.calc_radius_mask()

        # Initialize the variable for field and phase after propagation
        self.finished_reconstruction = False  # flag for full reconstruction with propagation and unwrapping

        # ToDo : delete obsolote variables ! - compare holo_dummy
        self.reconstructed_field = None  # field after numerical reconstruction - before propagation
        self.int_reconstructed = None  # intensity after numerical reconstruction - before propagation
        self.phase_reconstructed = None  # phase after numerical reconstruction - before propagation

        # Attributes, which will only be set with a full reconstruction
        self.propagated_field = None  # field after propagation and numerical reconstruction
        self.propagated_intensity = None  # intensity after propagation and numerical reconstruction
        self.propagated_phase = None  # phase after propagation and numerical reconstruction

        self.phase_unwrapped = None  # phase of the propagated phase after unwrapping
        # self.height_profile = None          # height profile of the unwrapped phase - to be done in future

    @property
    def shape(self):
        return self.data.shape

    def run(self):
        self.calc_field()

    def reconstruct(self, force=False, propagate=False, prop_dist=None):
        if self.reconstructed_field is None or force is True:
            self.calc_field(propagate=propagate, prop_dist=prop_dist)
            return self.reconstructed_field
        else:
            return self.reconstructed_field

    def set_full_reconstruction(self, re_field, propagated_field, phase_unwrapped, height_profile=None):
        self.finished_reconstruction = True
        # if re_field.all()==propagated_field.all():
        #     print("Gleich")
        self.reconstructed_field = re_field
        self.phase_reconstructed = self.phase(re_field)
        self.int_reconstructed = self.intensity(re_field)
        self.propagated_field = propagated_field
        if propagated_field is None:
            self.finished_reconstruction = False
        else:
            self.propagated_intensity = self.intensity(propagated_field)
        self.propagated_phase = self.phase(propagated_field)
        self.phase_unwrapped = phase_unwrapped
        # self.height_profile = height_profile

    def __locate_order(self):
        """
        Internal Methode.
        """
        # ToDo: Check how good the locateOrder works with 63x objective
        spectrum, fx, fy, weight = self.locateOrder(holo=self.data)
        self.spectrum_not_shifted = spectrum
        self.first_diffraction_order_pos = [fx, fy]
        self.calc_radius_mask()

    def calc_field(self, propagate=False, prop_dist=None):
        """
        Calculate the field of the hologram. Only use this function if you want to use the hologram of the object itself.
        """
        if propagate:
            (self.reconstructed_field, self.spectrum_not_shifted, self.spectrum_shifted, self.spectrum_propagated,
             self.spectrum_masked) = self.holo2Field_propagation(holo=self.data, fx=self.first_diffraction_order_pos[0],
                                                                 fy=self.first_diffraction_order_pos[1],
                                                                 r=self.radius_mask, return_spectrum=True,
                                                                 propagation_distance=prop_dist,
                                                                 pixel_pitch=self.pixel_pitch)
        else:
            (self.reconstructed_field, self.spectrum_not_shifted, self.spectrum_shifted,
             self.spectrum_masked) = self.holo2Field(holo=self.data, fx=self.first_diffraction_order_pos[0],
                                                     fy=self.first_diffraction_order_pos[1], r=self.radius_mask,
                                                     return_spectrum=True)

        self.reconstructed_phase = self.phase(self.reconstructed_field)
        self.reconstructed_intensity = self.intensity(self.reconstructed_field)
        self.finished_reconstruction = False  # Reset the finished reconstruction flag

    def calc_radius_mask(self):
        if self.radius_0_order is None:
            raise Exception(f"Dc Radius not implemented for objective {self.dhm.objective}")
        h, w = self.data.shape
        fx = self.first_diffraction_order_pos[0]
        fy = self.first_diffraction_order_pos[1]
        rmax = self._calc_radius_mask(fx=fx, fy=fy, r0=self.radius_0_order, h=h, w=w)
        self.radius_mask = rmax
        return rmax


class ReferenceHologram(Hologram):
    def __init__(self, data: np.ndarray, first_diffraction_order_pos, dhm_parameter):
        super().__init__(data=data, dhm_parameter=dhm_parameter,
                         first_diffraction_order_pos=first_diffraction_order_pos)
        self.calc_radius_mask()

