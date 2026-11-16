import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift


class HologramReconstructor:
    def __init__(self):
        self.epsilon = 1e-12  # Small constant to avoid division by zero

    def reconstruct_hologram(self, hologram, mask_radius=None, center_coords=None):
        """
        Reconstruct hologram using Fourier transform method

        Parameters:
        -----------
        hologram : ndarray
            Input hologram intensity pattern
        mask_radius : int, optional
            Radius of the circular mask for filtering
        center_coords : tuple, optional
            (x,y) coordinates of the first order maximum

        Returns:
        --------
        complex_field : ndarray
            Reconstructed complex field
        """
        # FFT of hologram
        spectrum = fft2(hologram)
        spectrum = fftshift(spectrum)

        # Apply mask to isolate first order
        if mask_radius is not None and center_coords is not None:
            mask = self._create_circular_mask(
                spectrum.shape, center_coords, mask_radius
            )
            spectrum = spectrum * mask

        # Center the first order
        if center_coords is not None:
            spectrum = self._center_first_order(spectrum, center_coords)

        # Inverse FFT to get reconstructed field
        complex_field = ifft2(ifftshift(spectrum))

        return complex_field

    def calculate_intensity(self, complex_field, mode="linear"):
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

        if mode.lower() == "db":
            # Convert to dB scale (10 log10 for intensity)
            intensity = 10 * np.log10(intensity + self.epsilon)

        return intensity

    def calculate_phase(self, complex_field, unwrap=False):
        """
        Calculate phase from complex field

        Parameters:
        -----------
        complex_field : ndarray
            Input complex field
        unwrap : bool
            Whether to unwrap the phase

        Returns:
        --------
        phase : ndarray
            Calculated phase in radians
        """
        phase = np.angle(complex_field)

        if unwrap:
            phase = np.unwrap(phase)

        return phase

    def _create_circular_mask(self, shape, center, radius):
        """Create circular mask for filtering"""
        y, x = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask.astype(float)

    def _center_first_order(self, spectrum, current_center):
        """Center the first order maximum"""
        ny, nx = spectrum.shape
        shift_y = ny // 2 - current_center[1]
        shift_x = nx // 2 - current_center[0]
        return np.roll(np.roll(spectrum, shift_y, axis=0), shift_x, axis=1)


# Example usage
def example_reconstruction():
    # Create sample hologram (this would be your actual hologram data)
    size = 512
    sample_hologram = np.random.random((size, size))

    # Initialize reconstructor
    reconstructor = HologramReconstructor()

    # Reconstruct hologram
    complex_field = reconstructor.reconstruct_hologram(
        sample_hologram, mask_radius=50, center_coords=(300, 300)
    )

    # Calculate intensity and phase
    intensity = reconstructor.calculate_intensity(complex_field, mode="linear")
    phase = reconstructor.calculate_phase(complex_field, unwrap=True)

    return intensity, phase
