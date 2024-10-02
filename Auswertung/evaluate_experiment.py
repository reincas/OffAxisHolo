from scidatacontainer import Container

from Auswertung.container import StructureContainer
from .container.util import update_container

from offaxisholo import Hologram, HologramReconstructor, ReferenceHologram, HologramPostProcessor


def evaluate_structure(structure_path):
    if structure_path.endswith(".zdc"):
        data_container = StructureContainer(file=structure_path)
    else:
        path = structure_path + ".zdc"
        data_container = Container(file=path)

    tmp_dict = {}
    params = data_container.dhm_params
    background_hologram = data_container.background_hologram
    finished_structure = data_container.complete_hologram

    structure_complete = Hologram(data=finished_structure, dhm=params)
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=structure_complete.first_diffraction_order_pos,
                                   dhm=params)

    for i in range(data_container.number_of_layer):
        # get raw hologram data
        hologram = data_container[f"meas/dhm/raw/layer_{i}.png"].data

        # reconstruction of the hologram using the parameter of the structure
        # Reconstruction
        I, phi = temporary_reconstruction_method(params)

        # get intensity of reconstructed image
        intensity_img = I
        # get unwrapped phase without aberrations
        phase_img = phi

        # save the intensity and phase
        tmp_dict.update({f"eval/dhm/layer_reconstruction/layer_{i}/intensity.png": intensity_img,
                         f"eval/dhm/layer_reconstruction/layer_{i}/phase.png": phase_img})

    # update container after complete reconstruction
    # ToDo: Add the reconstruction of complete structure
    update_container(structure_path, tmp_dict)


def temporary_reconstruction_method(dhm_parameter):
    dhm_params = dhm_parameter

    ##################################################################################################################
    zdc_path = ("C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/"
                "20240905_parameter_testprint_Zeiss 63x/structures/lens0_ABZ_h_0.1_l_0.15/dhm/"
                "dhm_lens0_ABZ_h_0.1_l_0.15.0.zdc")
    back_path = (
        "C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/"
        "20240905_parameter_testprint_Zeiss 63x/structures/lens0_ABZ_h_0.1_l_0.15/"
        "dhm_lens0_ABZ_h_0.1_l_0.15_before.zdc")
    save_path = "C:\\Users\\hanne\\Desktop\\Test4DHMReconstruction\\20240923"

    # Initialize the DHM object
    dhm_machine = DHM(objective="Zeiss 63x")

    # Initialize and use other classes
    hologram_data = get_hologram(path=zdc_path)
    background_data = get_hologram(path=back_path)
    # Todo liste
    # todo 1 - dhm überall als dict ersetzten und die parameter dann mit einer neuen funktion in die variablen speichern
    # todo 2 - reconstruction einbauen in code oben
    # todo 3 - Rekonstruktion mittels simulation untersuchen
    # todo 4 -
    hologram = Hologram(data=hologram_data, dhm=dhm_machine)
    background = ReferenceHologram(data=background_data, dhm=dhm_machine,
                                   first_diffraction_order_pos=hologram.first_diffraction_order_pos)

    processor = HologramPostProcessor(hologram)
    reconstructor = HologramReconstructor(hologram=hologram, processor=processor, reference=background)

    reconstructor.set_save_path(path=save_path)
    phase = reconstructor.evaluate(save_img=True)
    # ________________________________________________________________________________
    intensity = 0
    phase = 1j
    return intensity, phase
