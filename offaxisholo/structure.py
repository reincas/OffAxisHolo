import numpy as np
from offaxisholo.HologramReconstructor import HologramReconstructor

class Structure3D:
    def __init__(self):
        self.layers = []

    def add_layer(self, hologram: HologramReconstructor):
        """Add a reconstructed hologram as a layer to the 3D structure."""
        reconstructed_layer = hologram.reconstruct()
        self.layers.append(reconstructed_layer)

    def reconstruct_3d_structure(self) -> np.ndarray:
        """Reconstruct the 3D structure from all layers."""
        # Implement the logic to combine layers into a 3D structure
        return np.dstack(self.layers)
