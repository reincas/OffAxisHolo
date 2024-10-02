"""
Should serve as a docker between Structure Container and the Hologram reconstruction class

Vorgehensweise:

1. Pfad des Experiments
"""

from offaxisholo import ReferenceHologram, Hologram, HologramReconstructor, HologramPostProcessor, Structure3D
from .collector import StructureCollector


class Docker:
    def __init__(self):
        pass

    def evaluate(self):
        description = ""
        for structure_path in self.structure_path_list:
            structure = StructureCollector(structure_path, exp_description=description)
            structure_dict = structure.get_container()

        # Initialize the DHM object
        dhm_machine = self.DHM

        # Initialize and use other classes
        i = 0
        hologram_data = structure_dict[f"meas/dhm/layer_{i}.png"].data
        background_data = structure_dict["meas/dhm/background.png"].data
        hologram = Hologram(data=hologram_data, dhm=dhm_machine)
        background = ReferenceHologram(data=background_data, dhm=dhm_machine,
                                       first_diffraction_order_pos=hologram.first_diffraction_order_pos)

        processor = HologramPostProcessor(hologram)
        reconstructor = HologramReconstructor(hologram=hologram, processor=processor, reference=background)

        phase = reconstructor.run()
        reconstructor.plotImage(img=phase)

        structure_3d = Structure3D()


if __name__ == "__main__":
    root = "C:\\Users\\hanne\\Documents\\Seafile\\Nanoproduction_Hannes\\Code\\NanoFactorySystem\\mains\\.output\\dhm_paper\\20240820_dhm_testprint_Zeiss 63x"
    Exp = ExperimentCollector(root_path=root)
    Exp.evaluate()
