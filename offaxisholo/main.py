import numpy as np
from offaxisholo.DHM_DUMMY import DHM
from offaxisholo.hologram import Hologram
from offaxisholo.postprocessor import HologramPostProcessor
from offaxisholo.reconstructor import HologramReconstructor
from offaxisholo.structure import Structure3D

# Initialize the DHM object
dhm_machine = DHM(wavelength=632.8e-9, magnification=10, pixel_size=6.5e-6)

# Initialize and use other classes
hologram_data = np.random.rand(512, 512)
hologram = Hologram(hologram_data, dhm_machine)

processor = HologramProcessor(hologram, dhm_machine)
reconstructor = HologramReconstructor(hologram, processor, dhm_machine)

structure_3d = Structure3D(dhm_machine)
structure_3d.add_layer(reconstructor)
final_structure = structure_3d.reconstruct_3d_structure()

"""
Summary of Documentation Structure
------------------------------------
DHM Class:

Describes the DHM setup and its parameters.
Provides methods for fetching reconstruction parameters and handling reference holograms.
------------------------------------
Hologram Class:

Handles individual holograms, including their data and reconstruction through FFT.
Provides methods for compensating aberrations using reference holograms.
------------------------------------
HologramProcessor Class:

Focuses on post-processing steps, like filtering and image enhancement.
------------------------------------
HologramReconstructor Class:

Coordinates the entire reconstruction process, ensuring that all necessary steps (FFT, filtering, compensation) are 
properly executed.
------------------------------------
Structure3D Class:

Manages the compilation of multiple holograms into a 3D structure, facilitating layer-by-layer reconstruction.
This documentation structure provides a clear and detailed overview of each class's purpose, attributes, and methods. It
serves as a guide for both users and developers to understand and effectively use the code for hologram reconstruction.
------------------------------------

"""