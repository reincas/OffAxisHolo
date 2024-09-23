import numpy as np

from offaxisholo import *

zdc_path = ("C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/"
            "20240905_parameter_testprint_Zeiss 63x/structures/lens0_ABZ_h_0.1_l_0.15/dhm/"
            "dhm_lens0_ABZ_h_0.1_l_0.15.0.zdc")
back_path = ("C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/"
             "20240905_parameter_testprint_Zeiss 63x/structures/lens0_ABZ_h_0.1_l_0.15/"
             "dhm_lens0_ABZ_h_0.1_l_0.15_before.zdc")
save_path = "C:\\Users\\hanne\\Desktop\\Test4DHMReconstruction\\20240923"

# Initialize the DHM object
dhm_machine = DHM(objective="Zeiss 63x")

# Initialize and use other classes
hologram_data = get_hologram(path=zdc_path)
background_data = get_hologram(path=back_path)
hologram = Hologram(data=hologram_data, dhm=dhm_machine)
background = ReferenceHologram(data=background_data, dhm=dhm_machine,
                               first_diffraction_order_pos=hologram.first_diffraction_order_pos)

processor = HologramPostProcessor(hologram)
reconstructor = HologramReconstructor(hologram=hologram, processor=processor, reference=background)

reconstructor.set_save_path(path=save_path)
phase = reconstructor.evaluate(save_img=True)


# structure_3d = Structure3D()
# structure_3d.add_layer(reconstructor)
# final_structure = structure_3d.reconstruct_3d_structure()

"""


# Initialize the DHM object
dhm_machine = DHM_DUMMY(wavelength=632.8e-9, magnification=63, pixel_size=6.5e-6)

# Initialize and use other classes
hologram_data = np.random.rand(512, 512)
hologram = Hologram(hologram_data, dhm_machine)

processor = HologramPostProcessor(hologram)
reconstructor = HologramReconstructor(hologram, processor, dhm_machine)

structure_3d = Structure3D(dhm_machine)
structure_3d.add_layer(reconstructor)
final_structure = structure_3d.reconstruct_3d_structure()


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
