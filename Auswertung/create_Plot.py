import numpy as np
from matplotlib import pyplot as plt
import os

from scidatacontainer import Container

from Auswertung import mkdir
from Auswertung.container import StructureContainer
from Auswertung.evaluate import evaluate_image
from DataClass_Sensofar import SensofarData
from DataClass_np_array import StructureDataClass, NumpyArrayDataClass, StructureAxis
from offaxisholo import Hologram, ReferenceHologram, HologramReconstructor


def plot_profile_comparison(data_dhm_developed, data_dhm_printing, data_sensofar, data_lsm=None, axis="x", save_path=None):
    """
    Plot all data into one plot. Data has to be an 1D array. Make sure the data is at the same place and has the same
    dimensions.
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Comparison of the {axis.lower()}-profile")
    ax.plot(data_sensofar, 'r--', label='CM data (Sensofar)')
    ax.plot(1e6*data_dhm_printing, 'c--', label='DHM data while printing')
    ax.plot(1e6*data_dhm_developed, 'b--', label='DHM data after development')
    ax.legend()
    if data_lsm is not None:
        ax.plot(data_lsm, 'g--', label='LSM data (Goslar)')

    name = "profile_comparison_axis_" + axis.lower() + ".png"
    if save_path is not None:
        save_path = os.path.join(save_path, name)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def docker_sensofar_to_data(sensofar_dat_path):
    """
    Docker for a filepath to extract the necessary data as a numpy array from the sensofar data.
    :return:
    """
    sensofar_data = SensofarData(sensofar_dat_path)
    senso_data = np.asarray(sensofar_data.as_matrix())
    sensofar_pixelsize = sensofar_data.pixel_size
    return senso_data, sensofar_pixelsize


def dhm_image_to_data(dhm_img_path, background_path):
    reconstructor = evaluate_image(dhm_img_path, background_img_path=background_path)
    return reconstructor


def dhm_container_to_data(dhm_container_path):
    """
    reconstruction of the hologram + extra steps to split up the data into arrays which will be visualized
    :param dhm_data:
    :return:
    """
    if dhm_container_path.endswith(".zdc"):
        data_container = StructureContainer(file=dhm_container_path)
    else:
        path = dhm_container_path + ".zdc"
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
    phase_img = reconstructor.run()
    return reconstructor


def dhm_plot_data(reconstructor: HologramReconstructor, save_path, structure_name=None, cmap='coolwarm'):
    """
    Plotting of hologram, background hologram, reconstructed intensity, reconstructed unwrapped phase
    """
    hologram_captured = reconstructor.hologram.data
    hologram_background = reconstructor.background.data
    int_reconstructed = reconstructor.intensity_compensated
    phase_unwrapped_reconstructed_rad = reconstructor.phase_compensated
    phase_unwrapped_reconstructed_degree = np.degrees(phase_unwrapped_reconstructed_rad)

    if structure_name is None:
        structure_name = "structure"
    path_holo = structure_name + "hologram_captured.png"
    path_back = structure_name + "hologram_background.png"
    path_int = structure_name + "int_reconstructed.png"
    path_phase_rad = structure_name + "phase_rad.png"
    path_phase_deg = structure_name + "phase_degree.png"

    path = os.path.join(save_path, path_holo)
    fig = plt.figure()
    plt.title("Captured hologram")
    plt.imsave(path, hologram_captured, cmap=cmap)

    path = os.path.join(save_path, path_back)
    fig = plt.figure()
    plt.title("Captured background-hologram")
    plt.imsave(path, hologram_background, cmap=cmap)

    path = os.path.join(save_path, path_int)
    plt.imsave(path, int_reconstructed, cmap=cmap)

    path = os.path.join(save_path, path_phase_rad)
    plt.imsave(path, phase_unwrapped_reconstructed_rad, cmap=cmap)

    path = os.path.join(save_path, path_phase_deg)
    plt.imsave(path, phase_unwrapped_reconstructed_degree, cmap=cmap)


def dhm_data_to_height_array(reconstructor: HologramReconstructor, n_resin=1.5):
    phase_unwrapped_reconstructed_rad = reconstructor.phase_compensated
    height_np_array = reconstructor.phase_to_height(-phase_unwrapped_reconstructed_rad, n_resin=n_resin)
    pixel_pitch = reconstructor.pixel_pitch
    return height_np_array, pixel_pitch


def main(dhm_zdc_path, dhm_image_developed_path, dhm_image_background_developed_path, sensofar_dat_path,
         BASE_PATH, structure_name):
    """
    combining all functions in the end
    """
    N_RESIN = 1.3
    C_MAP = 'viridis'

    # --------- LSM NOT YET READY ----------
    """
    px_size_lsm = 0.000368
    save_path_lsm = "lsm"
    lsm_np_array_path = r""
    lsm_structure_data = StructureDataClass(data=lsm_np_array_path, pixel_size=px_size_lsm)
    """

    rel_path_dhm_printing = r"dhm/printing/"
    save_path_dhm_printing = mkdir(os.path.join(BASE_PATH, rel_path_dhm_printing))
    rel_path_dhm_developed = r"dhm/developed/"
    save_path_dhm_developed = mkdir(os.path.join(BASE_PATH, rel_path_dhm_developed))
    rel_path_sensofar = 'sensofar'
    save_path_sensofar = mkdir(os.path.join(BASE_PATH, rel_path_sensofar))
    rel_path_compare = r"compare"
    save_path_compare = mkdir(os.path.join(BASE_PATH, rel_path_compare))

    sensofar_data, px_size_sensofar = docker_sensofar_to_data(sensofar_dat_path)

    dhm_printing_reconstructor = dhm_container_to_data(dhm_zdc_path)
    dhm_developed_reconstructor = dhm_image_to_data(dhm_img_path=dhm_image_developed_path,
                                                    background_path=dhm_image_background_developed_path)
    dhm_printing_nparray, dhm_pixelpitch_printing = dhm_data_to_height_array(reconstructor=dhm_printing_reconstructor,
                                                                             n_resin=N_RESIN)
    dhm_developed_nparray, dhm_pixelpitch_developed = dhm_data_to_height_array(
        reconstructor=dhm_developed_reconstructor, n_resin=N_RESIN)

    # ------------- STRUCTURE DATACLASS -------------
    sensofar_structure_data = StructureDataClass(data=sensofar_data, pixel_size=px_size_sensofar)
    dhm_printing_structure_data = StructureDataClass(data=dhm_printing_nparray, pixel_size=dhm_pixelpitch_printing)
    dhm_developed_structure_data = StructureDataClass(data=dhm_developed_nparray, pixel_size=dhm_pixelpitch_developed)

    # ------------- Plotting -------------
    # DHM Plotting
    dhm_plot_data(reconstructor=dhm_printing_reconstructor, save_path=save_path_dhm_printing,
                  structure_name=structure_name, cmap=C_MAP)
    dhm_plot_data(reconstructor=dhm_developed_reconstructor, save_path=save_path_dhm_developed,
                  structure_name=structure_name, cmap=C_MAP)
    # Height plotting
    sensofar_structure_data.plot_height(save_path=save_path_sensofar, structure_name=structure_name,
                                        meas_tech="Confocal Microscopy")
    dhm_printing_structure_data.plot_height(save_path=save_path_dhm_printing, structure_name=structure_name,
                                            meas_tech="Digital Holographic Microscopy")
    dhm_developed_structure_data.plot_height(save_path=save_path_dhm_developed, structure_name=structure_name,
                                             meas_tech="Digital Holographic Microscopy")
    # Comparison Plotting
    axis = StructureAxis.X
    plot_profile_comparison(data_dhm_developed=dhm_developed_structure_data.get_profile(axis=axis, location="mid"),
                            data_dhm_printing=dhm_printing_structure_data.get_profile(axis=axis, location="mid"),
                            data_sensofar=sensofar_structure_data.get_profile(axis=axis, location="mid"),
                            data_lsm=None,
                            axis="x",
                            save_path=save_path_compare)
    axis = StructureAxis.Y
    plot_profile_comparison(data_dhm_developed=dhm_developed_structure_data.get_profile(axis=axis, location="mid"),
                            data_dhm_printing=dhm_printing_structure_data.get_profile(axis=axis, location="mid"),
                            data_sensofar=sensofar_structure_data.get_profile(axis=axis, location="mid"),
                            data_lsm=None,
                            axis="y",
                            save_path=save_path_compare)


if __name__ == "__main__":
    base = r"C:\Users\hanne\Desktop\Test_Comparison_DHM_Paper"
    structure = "DOE2_n_resin_1_3"
    BASE_PATH = os.path.join(base, structure)
# EXAMPLE DOE 1
    DHM_ZDC = r"C:\Users\hanne\Documents\Seafile\Nanoproduction_Hannes\Code\NanoFactorySystem\mains\.output\dhm_paper\FINAL_print_20241029_Zeiss 63x\structures\DOE1_ABZ_Zeiss 63x.zdc"
    DHM_IMAGE_DEVELOPED = r"C:\Users\hanne\Documents\Seafile\Nanoproduction_Hannes\Code\holo_4_paper_after_development\DOE1.tif"
    DHM_IMAGE_BACKGROUND_DEVELOPED = r"C:\Users\hanne\Documents\Seafile\Nanoproduction_Hannes\Code\holo_4_paper_after_development\DOE_background.tif"
    SENSOFAR_DAT = r"C:\Users\hanne\Documents\Seafile\Nanoproduction_Hannes\Sensofar\Substrat DHMPaper\#Auswertungsdaten\DOE3.dat"

    main(DHM_ZDC, DHM_IMAGE_DEVELOPED, DHM_IMAGE_BACKGROUND_DEVELOPED, SENSOFAR_DAT, BASE_PATH,
         structure_name=structure)

