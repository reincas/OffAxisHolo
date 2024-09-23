import numpy as np
import cv2 as cv
import scidatacontainer
from scipy.ndimage import maximum_filter

from .plotter import DHMPlotter

"""
Handles individual holograms, including their data and reconstruction through FFT.
Provides methods for compensating aberrations using reference holograms.
"""

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
        self.spectrum_masked = None
        self.reconstructed_field_before_propagation = None  # np.complex128  # ToDo: How to initialize this as type
        self.reconstructed_intensity = None
        self.reconstructed_phase = None

    def intensity(self, inp, log=True):
        out = np.abs(inp)
        if not log:
            out = out * out
        else:
            out = 20 * np.log(out)
            out[out == np.inf] = 0
            out[out == -np.inf] = 0
        return out

    def phase(self, inp):
        out = np.angle(inp)
        return out

    def getField(self, spectrum) -> np.ndarray:
        """ Return complex field from centered spectrum. """
        field = np.fft.ifft2(np.fft.fftshift(spectrum))
        return field

    def getSpectrum(self, field) -> np.ndarray:
        FT = np.fft.fft2(field)
        spectrum = np.fft.fftshift(FT)
        return spectrum

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
        holo = holo.astype(np.float64)

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
    def __init__(self, data: np.ndarray, dhm, first_diffraction_order_pos=None):
        super().__init__(dummy_mode=False)
        if isinstance(data, scidatacontainer.fileimage.PngFile):
            self.data = data.data
        else:
            self.data = data
        if len(self.data.shape) != 2:
            raise Exception(f"An Error occurred. 2D Hologram image required! Check data.")
        if self.data.shape[0] != self.data.shape[1]:
            raise Exception("Quadratic hologram image required!")

        # ToDo: change settings of DHM based information to a correct useage
        self.radius_0_order = dhm.r0
        self.propagation_distance = dhm.prop_dist
        self.wavelength = dhm.wavelength
        self.pixel_pitch = dhm.pixel_pitch

        self.first_diffraction_order_pos = first_diffraction_order_pos
        # Starting the necessary functions
        if self.first_diffraction_order_pos is None:
            self.__locate_order()

        # Initialize the variable for field and phase after propagation
        self.finished_reconstruction = False  # flag for full reconstruction with propagation and unwrapping

        self.reconstructed_field = None     # field after numerical reconstruction - before propagation
        self.int_reconstructed = None       # intensity after numerical reconstruction - before propagation
        self.phase_reconstructed = None     # phase after numerical reconstruction - before propagation

        # Attributes, which will only be set with a full reconstruction
        self.propagated_field = None        # field after propagation and numerical reconstruction
        self.propagated_intensity = None    # intensity after propagation and numerical reconstruction
        self.propagated_phase = None        # phase after propagation and numerical reconstruction

        self.phase_unwrapped = None         # phase of the propagated phase after unwrapping
        # self.height_profile = None          # height profile of the unwrapped phase - to be done in future

    @property
    def shape(self):
        return self.data.shape

    def run(self):
        self.calc_field()

    def reconstruct(self, force=False):
        if self.reconstructed_field_before_propagation is None or force is True:
            self.calc_field()
            return self.reconstructed_field_before_propagation
        else:
            return self.reconstructed_field_before_propagation

    def set_full_reconstruction(self, re_field, propagated_field, phase_unwrapped, height_profile=None):
        self.finished_reconstruction = True
        # if re_field.all()==propagated_field.all():
        #     print("Gleich")
        self.reconstructed_field = re_field
        self.phase_reconstructed = self.phase(re_field)
        self.int_reconstructed = self.intensity(re_field)
        self.propagated_field = propagated_field
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

    def calc_field(self):
        """
        Calculate the field of the hologram. Only use this function if you want to use the hologram of the object itself.
        """
        (self.reconstructed_field_before_propagation, self.spectrum_not_shifted, self.spectrum_shifted,
         self.spectrum_masked) = self.holo2Field(holo=self.data, fx=self.first_diffraction_order_pos[0],
                   fy=self.first_diffraction_order_pos[1], r=self.radius_mask, return_spectrum=True)
        self.reconstructed_phase = self.phase(self.reconstructed_field_before_propagation)
        self.reconstructed_intensity = self.intensity(self.reconstructed_field_before_propagation)
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
    def __init__(self, data: np.ndarray, first_diffraction_order_pos, dhm):
        super().__init__(data=data, dhm=dhm, first_diffraction_order_pos=first_diffraction_order_pos)
        self.calc_radius_mask()





