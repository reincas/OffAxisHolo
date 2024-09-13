import numpy as np
from offaxisholo import HologramReconstructor

"""
Manages the compilation of multiple holograms into a 3D structure, facilitating layer-by-layer reconstruction.
This documentation structure provides a clear and detailed overview of each class's purpose, attributes, and methods. It
serves as a guide for both users and developers to understand and effectively use the code for hologram reconstruction.
"""
class Structure3D:
    def __init__(self):
        self.layers = []

    def add_layer(self, hologram: HologramReconstructor):
        """Add a reconstructed hologram as a layer to the 3D structure."""
        # reconstructed_layer = hologram.reconstruct()
        # self.layers.append(reconstructed_layer)
        print("Reconstruction")

    def reconstruct_3d_structure(self) -> np.ndarray:
        """Reconstruct the 3D structure from all layers."""
        # Implement the logic to combine layers into a 3D structure
        return np.dstack(self.layers)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# From the other file:

'''
class DHM:
    def __init__(self):
        self.pixel_pitch = []
        self.wavelength = 0
        self.prop_dist = 20
        self.n_resin = 0


# ToDo: Umschreiben der gesamten Klasse sodass immer nur der Container erforderlich ist. Braucht man eine nicht-Container VErsion?

class HologramTomographic:

    def __init__(self, root_directory, logger=None):
        self.logger = logger

        # Initialisation
        self.layer_container = {
            'Position': [],  # should have all the global information available.
            'Layer Height': 0,  # specific height fpr this layer.
        }
        self.layer = None

        # ToDo: figure out how i want to use the information on where the holograms are saved.
        self.root_directory = root_directory
        self.working_directory = os.path.join(root_directory,
                                              'Reconstructed')  # should act as a saving directory for all the data and so on.

    def add_layer(self, layer):
        # adding a hologram to the list/ dict of layer.
        # layer.run have to be called
        # set background with this background here
        pass

    def set_background(self):
        # field, phase, int should be saved for giving it to the layers.
        pass

    def _get_hologram_layer_list(self):
        # returns a list of the holograms for the investigated structure
        # with fnmatch and glob
        pass

    def _getLoggerInformation(self):
        # acts like a catching point for all the information of the logger folder.
        # all the informations are dozen of times available. - make it one
        # add additional information regarding the structures
        # how many layers were printed?
        # what kind of structure
        # where
        # maximal dimension of those structures
        pass

'''