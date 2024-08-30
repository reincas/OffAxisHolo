import numpy as np
from offaxisholo.Hologram import Hologram
from offaxisholo.HologramPostProcessor import HologramPostProcessor


class HologramReconstructor:
    def __init__(self, hologram: Hologram, processor: HologramPostProcessor):
        self.hologram = hologram
        self.processor = processor

    def reconstruct(self) -> np.ndarray:
        """Full reconstruction pipeline including filtering and compensation."""
        # Step 1: FFT Reconstruction
        reconstructed = self.hologram.fft_reconstruct()
        # Step 2: Post-Processing (Filter)
        filtered = self.processor.filter(reconstructed)
        # Step 3: Aberration Compensation
        # Assuming a reference hologram is provided
        # compensated = self.processor.compensate_aberration(reference)
        return filtered
