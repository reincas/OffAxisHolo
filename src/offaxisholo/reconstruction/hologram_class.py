import math
from datetime import datetime
from typing import Tuple

import cv2 as cv
import numpy as np

from .. import backend
from .base_class import HologramCore


class Hologram(HologramCore):
    # All arrays live on the active backend (NumPy or CuPy, see backend.py).
    data = None  # original Hologram data (original dtype)
    # all possible data arrays
    spectrum = None
    spectrum_shifted = None
    spectrum_masked = None
    reconstructed_field = None
    reconstructed_phase = None  # wrapped phase
    reconstructed_intensity = None
    # other variables
    logger = None
    first_diffraction_order_pos: Tuple[float, float] | Tuple[int, int] = None
    finished_reconstruction: bool = (
        False  # flag for successful completed reconstruction
    )
    required_dhm_keys = {"dcRadius", "name"}  # required parameters for this class
    # PROBLEM: #todo dcRadius is also called 'DC Radius' and name == obejctive name

    def __init__(
        self,
        data,
        dhm_parameter,
        first_diffraction_order_pos=None,
        logger=None,
    ):
        # move the hologram to the active backend (GPU for 'cupy'); the original
        # dtype is kept, the conversion to the working precision is done later
        self.data = backend.asarray(data)
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
            self.logger.debug("Validating input")
        if len(self.data.shape) != 2:
            raise Exception(
                "An Error occurred. 2D Hologram image required! Check data."
            )
        if self.data.shape[0] != self.data.shape[1]:
            raise Exception("Quadratic hologram image required!")

        assert isinstance(
            self.params, dict
        )  # dhm parameter has to be given as dictionary
        assert all(key in self.params for key in self.required_dhm_keys), (
            f"Missing required parameter(s): "
            f"{self.required_dhm_keys - self.params.keys()}"
        )
        if self.logger:
            self.logger.debug("Input successfully validated")

    @property
    def shape(self):
        return self.data.shape

    def reconstruct(self, force=False):
        t1, t2 = None, None
        if self.logger:
            t1 = datetime.now()
            self.logger.info("Starting hologram reconstruction...")
        if force or (not self.finished_reconstruction):
            self.calc_field()
            t2 = datetime.now()
        if self.logger:
            if t1 is not None and t2 is not None:
                self.logger.info(
                    f"Reconstruction completed in {(t2 - t1).total_seconds()} seconds"
                )
            self.logger.debug(
                f"Reconstruction of Hologram captured with {self.params['name']} "
                f"(DC radius {self.params['dcRadius']}) successful."
            )
        return self.reconstructed_field

    # todo - how to deal with this? - remove??
    def set_full_reconstruction(
        self, re_field, propagated_field, phase_unwrapped, height_profile=None
    ):
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
        Calculate the complex object field from a real, positive hologram image
        (Fourier transform method / spatial filtering).

        The recorded off-axis hologram is the intensity of the superposition of
        object wave O and tilted reference wave R:

            I_H = |O + R|^2 = |O|^2 + |R|^2 + O * conj(R) + conj(O) * R.

        Due to the tilt of R (carrier frequency f_c), the four terms are separated
        in the Fourier spectrum: the zero order (|O|^2 + |R|^2) at the centre and
        the two first orders O * conj(R) and conj(O) * R at +f_c and -f_c.

        Steps:
            1. Centred spectrum of the hologram (FFT).
            2. Shift the selected first order (fx, fy) to the spectrum centre. This
               removes the carrier frequency, i.e. the linear phase of the plane
               reference wave.
            3. Circular mask with radius r -> only the first order remains.
            4. Inverse FFT -> complex field proportional to O * conj(R) / |R|.

        Note: The two first orders are complex conjugates of each other. Selecting
        the other order inverts the sign of the phase (and thus of the height).
        The sign convention therefore depends on the geometry of the reference
        beam of the setup.

        References:
            Mitsuo Takeda, Hideki Ina, and Seiji Kobayashi, "Fourier-transform
            method of fringe-pattern analysis for computer-based topography and
            interferometry," J. Opt. Soc. Am. 72, 156-160 (1982).
            https://doi.org/10.1364/JOSA.72.000156

            Etienne Cuche, Pierre Marquet, and Christian Depeursinge, "Spatial
            filtering for zero-order and twin-image elimination in digital
            off-axis holography," Appl. Opt. 39, 4070-4075 (2000).
            https://doi.org/10.1364/AO.39.004070
        """
        if holo is None:
            holo = self.data
        holo = backend.asarray(holo, dtype=backend.real_dtype())
        xp = backend.get_namespace(holo)
        if fx is None or fy is None:
            fx = self.first_diffraction_order_pos[0]
            fy = self.first_diffraction_order_pos[1]
        if r is None:
            r = self.radius_mask

        # Positive, real valued hologram
        if len(holo.shape) != 2:
            raise RuntimeError("2D hologram image required!")
        if bool(xp.min(holo) < 0):
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
        """Calculate the Fourier spectrum of the given positive real valued
        hologram image and return the spectral coordinates, the maximum spectral
        filter radius and the weight of the estimated first diffraction order
        peak. The global maximum after masking the zero and Nyquist frequencies
        is taken as first diffraction order. The size parameter is the smoothing
        radius and thus limits the density of local minima to be considered. The
        weight of the peak is between 0.0 and 1.0."""

        if holo is None:
            holo = self.data
        holo = backend.asarray(holo, dtype=backend.real_dtype())
        xp = backend.get_namespace(holo)

        assert len(holo.shape) == 2, "2D hologram image required!"
        assert holo.shape[0] == holo.shape[1], "Quadratic hologram image required!"
        assert holo.shape[0] % 2 == 0, "Hologram image with even dimensions required!"
        assert bool(xp.min(holo) >= 0.0), "Real positive hologram image required!"

        # Get spectrum of the hologram image
        spectrum = self.get_spectrum(holo)
        N = spectrum.shape[0]

        # Blur and normalize the right half of the spectrum.
        # CPU: OpenCV (kernel radius 4 * sigma, border BORDER_REFLECT_101).
        # GPU: equivalent ndimage filter (truncate=4, mode 'mirror').
        magnitude = xp.abs(spectrum[:, : N // 2])
        if backend.is_gpu(magnitude):
            blurred = backend.gaussian_filter(
                magnitude, sigma=size, mode="mirror", truncate=4.0
            )
        else:
            blurred = cv.GaussianBlur(magnitude, None, size)
        blurred = blurred / blurred[0, 0]

        # Get indices of all local maxima in the spectrum (on the device), then
        # evaluate the few candidates on the CPU
        maxmask = backend.maximum_filter(blurred, size=size) == blurred
        candidates = xp.nonzero(maxmask)
        weights = backend.to_numpy(blurred[candidates])
        ys, xs = (backend.to_numpy(c) for c in candidates)

        # Strip all local maxima around the zero and the Nyquist frequency. This
        # strips the dominating zero order peak and many mirror artifacts.
        s = N // 4
        keep = (np.abs(xs % (2 * s) - s) < s - size // 2) & (
            np.abs(ys % (2 * s) - s) < s - size // 2
        )
        if not np.any(keep):
            x, y, weight = None, None, 0.0

        else:
            # Take global maximum of the remaining points
            ys, xs, weights = ys[keep], xs[keep], weights[keep]
            best = int(np.argmax(weights))
            x = int(xs[best]) - N // 2
            y = int(ys[best]) - N // 2
            weight = float(weights[best])

        if self.logger:
            self.logger.debug(
                f"Calculated position of first diffraction order: {x}, {y}"
            )
        # Done.
        return spectrum, x, y, weight

    def roll_image(self, img, x, y):
        """Roll given image content so that point (x, y) becomes (0, 0). Wrap
        pixels at the image edges. Therefore, no information is lost.  For
        x = w//2 and y = h//2, the function is equivalent to np.fft.fftshift(img).
        """
        xp = backend.get_namespace(img)
        return xp.roll(img, (-int(y), -int(x)), axis=(0, 1))

    def circularMask(self, spectrum, r):
        """Apply circular mask with given radius to the centered spectrum."""

        xp = backend.get_namespace(spectrum)
        N = spectrum.shape[0]
        # integer coordinates relative to the centre -> exact squared radius
        c = xp.arange(N, dtype=xp.int64) - N // 2
        x, y = xp.meshgrid(c, c, indexing="xy")
        r2 = x * x + y * y
        return xp.where(r2 <= r * r, spectrum, 0.0)

    def _locate_order(self):
        """
        Internal Methode.
        """
        if self.logger:
            self.logger.info("Starting to locate position of first order diffraction.")
        # ToDo: Check how good the locateOrder works with 63x objective
        spectrum, fx, fy, weight = self.locate_order()
        self.spectrum = spectrum
        self.first_diffraction_order_pos = (fx, fy)
        self._calc_radius_mask()

    def calc_field(self):
        """
        Calculate the field of the hologram. Only use this function if you want to use
        the hologram of the object itself.
        """
        (
            self.reconstructed_field,
            self.spectrum,
            self.spectrum_shifted,
            self.spectrum_masked,
        ) = self.holo2field(return_spectrum=True)

        self.reconstructed_phase = self.phase(self.reconstructed_field)
        self.reconstructed_intensity = self.intensity(self.reconstructed_field)
        self.finished_reconstruction = True  # Set the reconstruction flag

    def _calc_radius_mask(self):
        if self.params["dcRadius"] is None:
            raise Exception(
                f"Dc Radius not implemented for objective {self.params['name']}"
            )
        if self.first_diffraction_order_pos[0] is None:
            raise RuntimeError(
                "First diffraction order could not be located in the spectrum."
            )
        h, w = self.data.shape
        fx = self.first_diffraction_order_pos[0]
        fy = self.first_diffraction_order_pos[1]
        rmax = math.sqrt(fx**2 + fy**2) - self.params["dcRadius"]
        rmax = min(rmax, abs(fx), w // 2 - abs(fx), abs(fy), h // 2 - abs(fy))
        if rmax <= 0:
            raise RuntimeError(
                f"Invalid mask radius {rmax}: first diffraction order at ({fx}, {fy}) "
                f"is too close to the DC term (dcRadius={self.params['dcRadius']}) "
                f"or to the spectrum border."
            )
        self.radius_mask = rmax
        if self.logger:
            self.logger.debug(f"Calculated radius mask radius: {rmax}")
        return rmax


class ReferenceHologram(Hologram):
    def __init__(self, data, first_diffraction_order_pos, dhm_parameter, logger=None):
        super().__init__(
            data=data,
            dhm_parameter=dhm_parameter,
            first_diffraction_order_pos=first_diffraction_order_pos,
            logger=logger,
        )
        self._calc_radius_mask()
