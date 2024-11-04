import numpy as np
from scidatacontainer import Container

from Auswertung.container import StructureContainer
from Auswertung.container import update_container

from offaxisholo import Hologram, HologramReconstructor, ReferenceHologram, HologramPostProcessor


def test_propagation_distance(structure_path):
    if structure_path.endswith(".zdc"):
        data_container = StructureContainer(file=structure_path)
    else:
        path = structure_path + ".zdc"
        data_container = Container(file=path)

    tmp_dict = {}
    params = data_container.dhm_params
    background_hologram = data_container.background_hologram
    finished_structure = data_container.complete_hologram

    structure_complete = Hologram(data=finished_structure, dhm_parameter=params)
    pos = structure_complete.first_diffraction_order_pos
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=pos,
                                   dhm_parameter=params)

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete, background)

    # testing the propagation distance with a loop
    prop_dist = np.arange(0, 20e-3, 10e-6)    # maximum 2 cm , step 10µm
    for i in range(len(prop_dist)):
        phase_img = reconstructor.run(prop_dist=prop_dist[i])
        intensity_img = structure_complete.propagated_intensity
        background.run()
        background_phase = background.reconstructed_phase
        background_intensity = background.reconstructed_intensity  # propagated_intensity

        tmp_dict.update({f"eval/propagation_test/distance_{prop_dist[i]}_m_intensity.png": intensity_img,
                         f"eval/propagation_test/distance_{prop_dist[i]}_m_phase.png": phase_img})

        # update container after complete reconstruction
        update_container(structure_path, tmp_dict)
        hologram_3d_modelling(structure_path)


def external_reconstruction_method(hologram, hologram_background):
    """
    Um die übersicht zu behalten
    Was used for an easy implementation of the current code. additional todos
    """
    # for a later investigation/ usage
    # processor = HologramPostProcessor(hologram)
    processor = None
    reconstructor = HologramReconstructor(hologram=hologram, reference=hologram_background, processor=processor)

    phase = reconstructor.run()
    intensity = hologram.propagated_intensity

    # Todo liste
    # DONE_todo 1 - dhm_parameter überall als dict ersetzten und die parameter dann mit einer neuen funktion in die variablen speichern
    # DONE_todo 2 - reconstruction einbauen in code oben
    # todo 3 - Rekonstruktion mittels simulation untersuchen
    # todo 4 - investigate Filtering methods and reconstruction data

    return intensity, phase


def hologram_3d_modelling(structure_path):
    # 1. get all the reconstructed phase information
    # 2. calculation
    # 3. savind in eval/reconstruction/3D_model
    # have fun
    print("3D modelling not yet implemented")
    pass


def problem_solver(structure_path, save_path):
    if structure_path.endswith(".zdc"):
        data_container = StructureContainer(file=structure_path)
    else:
        path = structure_path + ".zdc"
        data_container = Container(file=path)

    tmp_dict = {}
    params = data_container.dhm_params
    background_hologram = data_container.background_hologram
    finished_structure = data_container.complete_hologram

    structure_complete = Hologram(data=finished_structure, dhm_parameter=params)
    pos = structure_complete.first_diffraction_order_pos
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=pos,
                                   dhm_parameter=params)

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete, background)

    reconstructor.set_save_path(save_path)
    reconstructor.evaluate(propagate=False, save_img=True)
