import fnmatch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from Lens_original import AsphericalLens
from DataClass_Sensofar import SensofarData
from utils import get_datafiles


def data_correction(data, compare_height, threshold = 0.75):
    data_struct = []
    for i in range(len(data)):
        if data[i] >= threshold:
            data_struct.append(data[i])

    mean_val = np.mean(data_struct)
    offset = mean_val - compare_height
    data = data - offset
    return data, offset

if __name__ == "__main__":
    base_path = 'C:\\Users\\hanne\\Documents\\Seafile\\Nanoproduction_Hannes\\Sensofar'
    experiment = "2024_06_25 DHM Paper Pillow test\Data"
    path = os.path.join(base_path, experiment)

    datafile_paths = get_datafiles(path)
    names = fnmatch.filter(os.listdir(path), '*.dat')

    # data_name = "Aspherical_left_structureonly.dat"
    # data_name = "Aspherical_left_good.dat"
    # data_path = os.path.join(base_path, experiment, data_name)

    for i in range(len(datafile_paths)):
        data_path = datafile_paths[i]
        fig = plt.figure(dpi=400)

        # ======  Data Aspherical lens   ======
        data_aspherical_left = SensofarData(data_path)
        px_size = data_aspherical_left.pixel_size
        mid_point = data_aspherical_left.structure_loc_mid

        data = data_aspherical_left.get_y_profile(index=mid_point[0])
        n_data_pnts = len(data)
        x_arr = np.arange(0, n_data_pnts * px_size, px_size)
        x_arr = x_arr - x_arr[mid_point[1]]  # verschiebung zurm 0-Punkt

        # ======  Plot Comparison   ======
        structure_height = 7
        length = 100
        width = 75
        lens = AsphericalLens((0, 0, 0), height=structure_height, length=length, width=width, radius_of_curvature=2030,
                              slice_size=.1, conic_constant_k=-2.3, alpha=0, hatch_size=0.1, velocity=40_000)
        # Plot the default
        prof, height = lens.calc_profile(axis='y', pos=0, step_size=px_size)
        prof = np.array(prof)
        height = -1 * np.array(height) + lens.height

        # ======  Plot Data   ======
        data_corrected, offset = data_correction(data, structure_height)
        plt.plot(x_arr, data_corrected)
        plt.plot(prof, height, color="orange")

        # ======  Save Figure   ======
        fig_name = f"Comparison {names[i][:-4]} and specified lens"
        save_path = os.path.join(path, fig_name)
        fig.savefig(save_path)
