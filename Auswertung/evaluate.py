import os.path

import cv2
from matplotlib import pyplot as plt
from scidatacontainer import Container

from Auswertung.container import StructureContainer
from Auswertung.container import update_container

from offaxisholo import Hologram, HologramReconstructor, ReferenceHologram, HologramPostProcessor


def evaluate_image(img_path, background_img_path=None, save_path=None, objective="Zeiss 63x"):
    if objective == "Zeiss 63x":
        dhm_params = {"pixel pitch": 0.0869e-6,
                      "wavelength": 0.6749e-6,
                      "refractive index": 1.5,
                      "propagation distance": 0,
                      "DC radius": 304}
    elif objective == "Zeiss 20x":
        dhm_params = {"pixel pitch": 0.27596e-6,
                      "wavelength": 0.6749e-6,
                      "refractive index": 1.5,
                      "propagation distance": 0,
                      "DC radius": 304}
    else:
        raise NotImplementedError(f"Unknown objective {objective}.")

    if img_path[-4:] == ".tif":
        img_name = img_path.split("\\")[-1][:-4]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(img_path)

    structure = Hologram(data=img, dhm_parameter=dhm_params)

    if background_img_path is not None:
        if background_img_path[-4:] == ".tif":
            back_img = cv2.imread(background_img_path, cv2.IMREAD_UNCHANGED)
            # Convert to grayscale
            back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        else:
            back_img = cv2.imread(background_img_path)

        background = ReferenceHologram(data=back_img, first_diffraction_order_pos=structure.first_diffraction_order_pos,
                                       dhm_parameter=dhm_params)
        reconstructor = HologramReconstructor(hologram=structure, reference=background)
        field_reconstructed = reconstructor.run()
    else:
        reconstructor = HologramReconstructor(hologram=structure)
        field_reconstructed = reconstructor.run(compensate=False)

    phase_structure = reconstructor.phase_unwrapping(structure.phase(field_reconstructed))
    height_structure = -reconstructor.phase_to_height(phase_structure)  # minus because the sample is upside down

    if save_path is not None:
        name_phase = img_name + "_reconstructed_phase" + ".png"
        name_height = img_name + "_height_profile" + ".png"
        path_phase = os.path.join(save_path, name_phase)
        path_height = os.path.join(save_path, name_phase)
        plt.imsave(path_phase, phase_structure)
        plt.imsave(path_height, height_structure)
    else:
        plt.imshow(phase_structure)
        plt.show()
        plt.imshow(height_structure)
        plt.show()


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

    structure_complete = Hologram(data=finished_structure, dhm_parameter=params)
    pos = structure_complete.first_diffraction_order_pos
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=pos,
                                   dhm_parameter=params)

    for i in range(data_container.number_of_layer):
        # get raw hologram data
        hologram_data = data_container[f"meas/dhm/raw/layer_{i}.png"].data

        # reconstruction of the hologram using the parameter of the structure
        # Reconstruction
        hologram_object = Hologram(data=hologram_data, dhm_parameter=params, first_diffraction_order_pos=pos)
        intensity_img, phase_img = external_reconstruction_method(hologram=hologram_object,
                                                                  hologram_background=background)

        # save the intensity and phase
        tmp_dict.update({f"eval/dhm/layer_reconstruction/layer_{i}/intensity.png": intensity_img,
                         f"eval/dhm/layer_reconstruction/layer_{i}/phase.png": phase_img})

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete, background)
    phase_img = reconstructor.run()
    intensity_img = structure_complete.propagated_intensity
    background.run()
    background_phase = background.reconstructed_phase
    background_intensity = background.reconstructed_intensity  # propagated_intensity

    tmp_dict.update({f"eval/dhm/layer_reconstruction/structure_intensity.png": intensity_img,
                     f"eval/dhm/layer_reconstruction/structure_phase.png": phase_img,
                     f"eval/dhm/layer_reconstruction/background_intensity.png": background_intensity,
                     f"eval/dhm/layer_reconstruction/background_phase.png": background_phase})

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
