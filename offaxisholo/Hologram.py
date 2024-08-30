import numpy as np


class Hologram:
    def __init__(self, data: np.ndarray):
        self.data = data

    def fft_reconstruct(self) -> np.ndarray:
        """Reconstruct the hologram using FFT."""
        # Perform FFT and return the reconstructed hologram
        return np.fft.fftshift(np.fft.fft2(self.data))
    