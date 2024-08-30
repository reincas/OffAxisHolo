import numpy as np
from offaxisholo.Hologram import Hologram
from offaxisholo.Filter import *


class HologramPostProcessor:
    def __init__(self, hologram: Hologram):
        self.hologram = hologram

    def filter(self, filter_method: Filter) -> np.ndarray:
        """Apply a filter method to the hologram."""
        return filter_method.apply(self.hologram.data)

    def compensate_aberration(self, reference: 'ReferenceHologram') -> np.ndarray:
        """Compensate aberrations using a reference hologram."""
        # Implement aberration compensation logic here
        pass

