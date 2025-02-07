from typing import Tuple
from datetime import datetime
import numpy as np
import cv2 as cv
from scipy.ndimage import maximum_filter

from .base_class import HologramCore

# ToDo:
#   - deal with the set full reconstruction -- is it necessary?


class Hologram(HologramCore):
    data: np.array  # original Hologram data
    # all possible data arrays
    spectrum: np.array = None
    spectrum_shifted: np.array = None
    spectrum_masked: np.array = None
    reconstructed_field: np.array = None
    reconstructed_phase: np.array = None  # wrapped phase
    reconstructed_intensity: np.array = None
    # other variables
    logger = None
    first_diffraction_order_pos: Tuple[float, float] | Tuple[int, int] = None
    finished_reconstruction: bool = False  # flag for successful completed reconstruction
    required_dhm_keys = {'dcRadius', 'name'}  # required parameters for this class
            # PROBLEM: #todo dcRadius is also called 'DC Radius' and name == obejctive name

    def __init__(self, data: np.ndarray, dhm_parameter, first_diffraction_order_pos=None, logger=None):
        self.data = data
        self.params = dhm_parameter
        self.first_diffraction_order_pos = first_diffraction_order_pos
        self.radius_mask = None

        if logger is not None:
            super().__init__(logger)

        self._validate_input()

        # Starting the necessary functions
        if self.first_diffraction_order_pos is None:
            self._locate_order()
        else:
            self._calc_radius_mask()

    def _validate_input(self):
        if self.logger:
            self.logger.DEBUG("Validating input")
        if len(self.data.shape) != 2:
            raise Exception(f"An Error occurred. 2D Hologram image required! Check data.")
        if self.data.shape[0] != self.data.shape[1]:
            raise Exception("Quadratic hologram image required!")

        assert isinstance(self.params, dict)  # dhm parameter has to be given as dictionary
        assert all(key in self.params for key in self.required_dhm_keys), \
            f"Missing required parameter(s): {self.required_dhm_keys - self.params.keys()}"
        if self.logger:
            self.logger.DEBUG("Input successfully validated")

    @property
    def shape(self):
        return self.data.shape

    def reconstruct(self, force=False):
        t1, t2 = None, None
        if self.logger:
            t1 = datetime.now()
            self.logger.INFO("Starting hologram reconstruction...")
        if force or (not self.finished_reconstruction):
            self.calc_field()
            t2 = datetime.now()
        if self.logger:
            if t1 is not None and t2 is not None:
                self.logger.INFO(f"Reconstruction completed in {(t2 - t1).total_seconds()} seconds")
            self.logger.DEBUG(
                f"Reconstruction of Hologram captured with {self.params['name']} (DC radius {self.params['dcRadius']}) successful.")
        return self.reconstructed_field

    # todo - how to deal with this? - remove??
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

    def holo2field(self, holo=None, fx=None, fy=None, r=None, return_spectrum=False):
        """
        Calculate the wave field from a given positive real valued hologram
        image based on the given spectral position of the first diffraction order
        relative to the zero order. A circular mask with the given radius is
        applied to the Fourier spectrum in order to extract the first order
        spectrum. """
        if holo is None:
            holo = self.data
        if (fx and fy) is None:
            fx = self.first_diffraction_order_pos[0]
            fy = self.first_diffraction_order_pos[1]
        if r is None:
            r = self.radius_mask

        # Positive, real valued hologram
        if len(holo.shape) != 2:
            raise RuntimeError("2D hologram image required!")
        if np.min(holo) < 0:
            raise RuntimeError("Real positive hologram image required!")
        if holo.shape[0] % 2 != 0 or holo.shape[1] % 2 != 0:
            raise RuntimeError("Hologram image with even dimensions required!")

        # Spatial spectrum of the hologram
        spectrum = self.get_spectrum(holo)

        # Roll the given first order coordinates to the centre of the spectrum
        spectrum_shifted = self.roll_image(spectrum, fx, fy)

        # Apply circular aperture with radius r
        spectrum_masked = self.circularMask(spectrum_shifted, r)

        # Calculate and return the wave field from the first order spectrum
        field = self.get_field(spectrum_masked)

        if return_spectrum:
            return field, spectrum, spectrum_shifted, spectrum_masked
        else:
            return field

    def locate_order(self, holo=None, size=16):
        """ Calculate the Fourier spectrum of the given positive real valued
        hologram image and return the spectral coordinates, the maximum spectral
        filter radius and the weight of the estimated first diffraction order
        peak. The global maximum after masking the zero and Nyquist frequencies
        is taken as first diffraction order. The size parameter is the smoothing
        radius and thus limits the density of local minima to be considered. The
        weight of the peak is between 0.0 and 1.0. """

        if holo is None:
            holo = self.data

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

        if self.logger:
            self.logger.DEBUG(f"Calculated position of first diffraction order: {x}, {y}")
        # Done.
        return spectrum, x, y, weight

    def roll_image(self, img, x, y):
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

    def _locate_order(self):
        """
        Internal Methode.
        """
        if self.logger:
            self.logger.INFO("Starting to locate position of first order diffraction.")
        # ToDo: Check how good the locateOrder works with 63x objective
        spectrum, fx, fy, weight = self.locate_order()
        self.spectrum = spectrum
        self.first_diffraction_order_pos = (fx, fy)
        self._calc_radius_mask()

    def calc_field(self):
        """
        Calculate the field of the hologram. Only use this function if you want to use the hologram of the object itself.
        """
        (self.reconstructed_field, self.spectrum, self.spectrum_shifted,
         self.spectrum_masked) = self.holo2field(return_spectrum=True)

        self.reconstructed_phase = self.phase(self.reconstructed_field)
        self.reconstructed_intensity = self.intensity(self.reconstructed_field)
        self.finished_reconstruction = True  # Set the reconstruction flag

    def _calc_radius_mask(self):
        if self.params['dcRadius'] is None:
            raise Exception(f"Dc Radius not implemented for objective {self.params.name}")
        h, w = self.data.shape
        fx = self.first_diffraction_order_pos[0]
        fy = self.first_diffraction_order_pos[1]
        rmax = np.sqrt(fx ** 2 + fy ** 2) - self.params['dcRadius']
        rmax = min(rmax, abs(fx), w // 2 - abs(fx), abs(fy), h // 2 - abs(fy))
        self.radius_mask = rmax
        if self.logger:
            self.logger.DEBUG(f"Calculated radius mask radius: {rmax}")
        return rmax


class ReferenceHologram(Hologram):
    def __init__(self, data: np.ndarray, first_diffraction_order_pos, dhm_parameter, logger=None):
        super().__init__(data=data, dhm_parameter=dhm_parameter,
                         first_diffraction_order_pos=first_diffraction_order_pos, logger=logger)
        self._calc_radius_mask()
