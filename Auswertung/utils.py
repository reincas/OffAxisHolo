import pathlib as Path
import numpy as np
import pandas as pd
import os
import glob

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def get_datafiles(root, subdir=False, ending='.dat') -> list:
    if subdir:
        files='**\*'+ending
        path = os.path.join(root, files)
    else:
        files = '*' + ending
        path = os.path.join(root, files)
    try:
        return glob.glob(path)
    except Exception as e:
        raise FileNotFoundError(f"No directory {root}. Exception {e}")


def read_dat_file(file_path):
    """
    Liest eine .dat Datei ein und gibt den Inhalt als String zurück.

    :param file_path: Der Pfad zur .dat Datei
    :return: Der Inhalt der Datei als String
    """
    try:
        with open(file_path, 'r') as file:
            data = file.read()
        return data
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None


def read_dat_file_to_dataframe(file_path):
    """
    Liest eine .dat Datei ein und gibt den Inhalt als Pandas DataFrame zurück.

    :param file_path: Der Pfad zur .dat Datei
    :return: Ein Pandas DataFrame mit dem Inhalt der Datei
    """
    try:
        # Lesen der Datei mit Pandas, Annahme dass die Datei durch Semikolons getrennt ist
        df = pd.read_csv(file_path, delimiter=';', header=None, names=['x', 'y', 'z'])

        # Erkennen der Linienanfänge
        start_value = df['x'].iloc[0]  # Annahme: die Linie beginnt immer mit demselben x-Wert
        df['line'] = (df['x'] == start_value).cumsum() - 1
        return df

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None


# def test_plot(dataframe):
#     lines = []
#     for line_id, line_data in dataframe.groupby('line'):
#         lines.append(line_data)
#
#     lines = np.array(lines)
#
#     fig = plt.figure(dpi=400)
#     ax = fig.add_subplot(projection='3d')
#     # ax2 = fig.add_subplot(122, projection='3d')
#
#     # for ax in [ax1, ax2]:
#     plt_lines = Line3DCollection(lines, linewidths=1)
#     ax.add_collection3d(plt_lines)
#     return fig


def test_plot(dataframe):
    lines = []
    for line_id, line_data in dataframe.groupby('line'):
        # Ensure line_data is sorted by some coordinate if needed
        line_points = line_data[['x', 'y', 'z']].values
        lines.append(line_points)

    lines = np.array(lines, dtype=object)  # Keep lines as arrays of arrays

    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111, projection='3d')

    plt_lines = Line3DCollection(lines, linewidths=1)
    ax.add_collection3d(plt_lines)

    # Setting limits, labels, and other plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
    return fig


def fill_nan_values(dataframe, window_size=5):
    # Fill NaN values for 'x' and 'y' with mean of neighboring values
    dataframe['x'].fillna(dataframe['x'].rolling(window=2, min_periods=1, center=True).mean(), inplace=True)
    dataframe['y'].fillna(dataframe['y'].rolling(window=2, min_periods=1, center=True).mean(), inplace=True)

    # Fill NaN values for 'z' with mean of 5 neighboring values
    dataframe['z'].fillna(dataframe['z'].rolling(window=window_size, min_periods=1, center=True).mean(), inplace=True)

    return dataframe


def plot_surface_from_points(dataframe):
    # # Fill NaN values
    # dataframe = fill_nan_values(dataframe)

    # Extracting x, y, z coordinates from the dataframe
    x = dataframe['x'].values
    y = dataframe['y'].values
    z = dataframe['z'].values

    # Exclude NaN values from the plot by creating a mask
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)

    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(x[mask], y[mask], z[mask], cmap='viridis', edgecolor='none')

    # Setting labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Showing the plot
    plt.show()
    return fig


if __name__ == "__main__":
    base_path = 'C:\\Users\\hanne\\Documents\\Seafile\\Nanoproduction_Hannes\\Sensofar'
    path = os.path.join(base_path, "2024_06_25 DHM Paper Pillow test\Data")
    data_name = "Aspherical_left_good.dat"
    file = read_dat_file_to_dataframe(os.path.join(path, data_name))

    start = True
    lines = []
    y_val = []
    index_start = 0
    index_end = 0
    # ToDo: alles in dataframes umsetzen und nicht hier in sowas
    # getting values and start and end index
    for line_id, line_data in file.groupby('line'):
        # Ensure line_data is sorted by some coordinate if needed
        line_points = line_data[['x', 'z']].values
        lines.append(line_points)
        a = line_data['y'].drop_duplicates().values
        y_val.append(a)

        z_mean = np.array(lines[line_id], dtype=object).mean(axis=0)[1]  # mean of z value of line
        if z_mean >= 1:
            if start:
                start = False
                index_start = line_id
            index_end = line_id

    lines_w_structure = lines[index_start:index_end]
    mid_line = np.array(lines[np.ceil((index_start+index_end)/2).astype(int)]).transpose()


    # plt.plot(mid_line[0], mid_line[1])
    plt.plot(mid_line[1])
    plt.ylim([-2,50])
    plt.show()


