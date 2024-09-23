"""
Should serve as a docker between Structure Container and the Hologram reconstruction class

Vorgehensweise:

1. Pfad des Experiments
"""
import os.path
import json

from offaxisholo import ReferenceHologram, Hologram, HologramReconstructor, HologramPostProcessor, Structure3D
from offaxisholo.Container_Structure import StructureCollector


class Docker:
    def __init__(self):
        pass

class Experiment:
    def __init__(self, root_path, dhm):
        self.root_path = root_path
        self.struct_info_dict = {}
        self.structure_path_list = []
        self.DHM = dhm # DHM(objective="Zeiss 63x") # ToDo: Hier überarbeiten wegen des DHMs - vielleicht einfach erstmal weiter den DHM Dummy nutzen und mit informationen aus den json dateien füllen.

        self.post_init()

    def post_init(self):
        self.get_structure_informations()

    def get_structure_informations(self):
        pfad = self.root_path + "\\structures.json"
        with open(pfad) as datei:
            dictionary = json.load(datei)
        self.struct_info_dict = dictionary
        for structure in dictionary:
            self.structure_path_list.append(os.path.join(self.root_path, structure["name"]))

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
    Exp = Experiment(root_path=root)
    Exp.evaluate()
