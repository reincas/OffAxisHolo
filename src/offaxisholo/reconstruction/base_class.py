from .. import backend
from . import fieldops


class HologramCore:
    """
    Base class with the elementary conversions between complex fields, their
    spectra, intensity and phase.

    Physical background
    -------------------
    A monochromatic scalar wave field is described by its complex amplitude

        E(x, y) = |E(x, y)| * exp(i * phi(x, y)),

    where |E| is the (real, non-negative) amplitude and phi the phase. Detectors
    only measure the time-averaged intensity, which is proportional to the squared
    magnitude of the complex amplitude, I = |E|^2 (constant factors such as
    c * eps0 / 2 are omitted, as only relative intensities are used).

    References:
        B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 2nd ed.,
        Wiley (2007), Chapter 2.2 (monochromatic waves, complex amplitude,
        intensity I = |U|^2).
        J. W. Goodman, "Introduction to Fourier Optics", 3rd ed., Roberts & Company
        (2005), Chapter 3 (scalar diffraction theory, complex field representation).
    """

    def __init__(self, logger=None):
        self.logger = logger

    def to_cpu(self):
        """
        Convert all array attributes of this object to NumPy arrays (copy from
        the GPU if the CuPy backend was used). Returns self.
        """
        for name, value in vars(self).items():
            if backend.is_array(value):
                setattr(self, name, backend.to_numpy(value))
        return self

    def get_field(self, spectrum):
        """
        Return the complex field from a centred spectrum (zero frequency in the
        middle of the array).

        The centred spectrum is shifted back to the standard FFT layout (zero
        frequency at index [0, 0]) with ifftshift and then inverse Fourier
        transformed. ifftshift is the exact inverse of fftshift for both even and
        odd array sizes.

        Reference:
            NumPy documentation, numpy.fft.fftshift / numpy.fft.ifftshift.
        """
        xp = backend.get_namespace(spectrum)
        field = xp.fft.ifftn(xp.fft.ifftshift(spectrum), axes=(-2, -1))
        return field

    def get_spectrum(self, field):
        """
        Return the centred spatial frequency spectrum of a field (zero frequency
        in the middle of the array). Inverse operation of get_field.
        """
        xp = backend.get_namespace(field)
        spectrum = xp.fft.fftshift(xp.fft.fftn(field, axes=(-2, -1)))
        return spectrum

    def intensity(self, complex_field, mode="db"):
        """
        Calculate the intensity of a complex field.

        Linear intensity:
            I = |E|^2

        Logarithmic intensity (decibel). Intensity is a power quantity, therefore
        the decibel value uses the factor 10:
            I_dB = 10 * log10(I / I_0),   with I_0 = 1 (arbitrary units).
        A small epsilon is added to avoid log10(0).

        Parameters
        ----------
        complex_field : ndarray
            Complex field E.
        mode : str
            'linear' for |E|^2, 'db' for 10 * log10(|E|^2).

        Returns
        -------
        ndarray
            Intensity in arbitrary linear units or in dB.

        References:
            B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 2nd ed.,
            Wiley (2007), Sec. 2.2 (I = |U|^2).
            IEC 60027-3, "Letter symbols to be used in electrical technology -
            Part 3: Logarithmic and related quantities, and their units"
            (definition of the decibel for power quantities: 10 * log10).
        """
        return fieldops.intensity(complex_field, mode=mode)

    def phase(self, input):
        """
        Calculate the wrapped phase of a complex field.

            phi = arg(E) = atan2(Im(E), Re(E)),   phi in (-pi, pi]

        The result is wrapped; use a phase unwrapping algorithm to obtain the
        continuous phase.

        Parameters
        ----------
        input : ndarray
            Complex field E.

        Returns
        -------
        ndarray
            Wrapped phase in radians.

        Reference:
            J. W. Goodman, "Introduction to Fourier Optics", 3rd ed. (2005),
            Chapter 3 (phasor representation of monochromatic fields).
        """
        return fieldops.phase(input)

    def calculate_efield(self, intensity, phase, intensity_is_db=False):
        """
        Calculate the complex field from intensity and phase.

        Inverse of intensity() and phase():

            |E| = sqrt(I)
            E   = sqrt(I) * exp(i * phi)

        If the intensity is given in dB it is first converted back to linear
        units, I = 10^(I_dB / 10).

        The intensity has to be non-negative, as it is a squared magnitude.
        Negative values are physically meaningless and raise a ValueError.

        Parameters
        ----------
        intensity : ndarray
            Intensity (linear or dB).
        phase : ndarray
            Phase in radians.
        intensity_is_db : bool
            True if intensity is given in dB.

        Returns
        -------
        ndarray
            Complex field E.

        Reference:
            B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 2nd ed.,
            Wiley (2007), Sec. 2.2.
        """
        xp = backend.get_namespace()
        intensity = backend.asarray(intensity, dtype=backend.real_dtype(xp))
        phase = backend.asarray(phase, dtype=backend.real_dtype(xp))

        if intensity.shape != phase.shape:
            raise ValueError("Intensity and phase arrays must have same shape")

        if bool(xp.any(xp.isnan(intensity))) or bool(xp.any(xp.isnan(phase))):
            raise ValueError("Input arrays contain NaN values")

        if intensity_is_db:
            intensity = 10 ** (intensity / 10)

        if bool(xp.any(intensity < 0)):
            raise ValueError(
                "Negative intensity values are not physical. "
                "Intensity must be |E|^2 >= 0."
            )

        amplitude = xp.sqrt(intensity)
        return amplitude * xp.exp(1j * phase)
