import numpy as np


class HologramCore:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logger

    def get_field(self, spectrum) -> np.ndarray:
        """Return complex field from centered spectrum."""
        field = np.fft.ifft2(np.fft.fftshift(spectrum))
        return field

    def get_spectrum(self, field) -> np.ndarray:
        FT = np.fft.fft2(field)
        spectrum = np.fft.fftshift(FT)
        return spectrum

    def intensity(self, complex_field, mode="db"):
        """
        Calculate intensity from complex field

        Parameters:
        -----------
        complex_field : ndarray
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

        if mode.lower() == "db":
            # Convert to dB scale (10 log10 for intensity)
            intensity = 10 * np.log10(intensity + eps)
        return intensity

    def phase(self, input):
        """
        Calculate phase from complex field

        Parameters:
        -----------
        input : ndarray
            Input complex field
        unwrap : bool
            Whether to unwrap the phase

        Returns:
        --------
        phase : ndarray
            Calculated phase in radians
        """
        phase = np.angle(input)
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

    def intensity_stable(self, complex_field, mode="linear"):
        """Calculate intensity with numerical stability"""
        # Use log(abs()) instead of abs()^2 for better numerical stability
        intensity = np.log(np.abs(complex_field))
        intensity = np.exp(2 * intensity)  # Equivalent to abs()^2 but more stable

        eps = 1e-12
        if mode.lower() == "db":
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

        self.phase_compensated = self.phase(original_scaled) - self.phase(
            reference_scaled
        )
        self.intensity_compensated_linear = self.intensity(
            original_scaled, mode="linear"
        ) - self.intensity(reference_scaled, mode="linear")
        self.intensity_compensated_db = self.intensity(
            original_scaled, mode="db"
        ) - self.intensity(reference_scaled, mode="db")

        return self.calculate_efield(
            intensity=self.intensity_compensated_linear,
            intensity_is_db=False,
            phase=self.phase_compensated,
        )
