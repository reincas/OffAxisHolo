import os, glob
from dataclasses import dataclass
import numpy as np
from typing import Optional
import pandas as pd
from matplotlib import pyplot as plt

@dataclass
class SensofarData():
    def __init__(self, path):
        self.data_path = path
        self.df = pd.DataFrame()
        self.load_data()
        self.data_correction()
        self._add_matrix_information()

        self.calc_pixel_size()
        self.calc_structure_information()

    def load_data(self):
        try:
            # Lesen der Datei mit Pandas, Annahme dass die Datei durch Semikolons getrennt ist
            self.df = pd.read_csv(self.data_path, delimiter=';', header=None, names=['x', 'y', 'z'])
        except Exception as e:
            print(f"Error while loading data: {e}")

    def data_correction(self):
        # ToDo: NaN should usually not be filled with zeroes - has to be changed
        self.df['z_no_nan'] = self.df['z'].fillna(0)

    def _add_matrix_information(self):
        self.add_y_id()
        self.add_x_id()
        self._add_matrix()

    def add_y_id(self):
        # Hinzufügen der y-Koordinatennummer
        start_value = self.df['x'].iloc[0]  # Annahme: die Linie beginnt immer mit demselben x-Wert
        self.df['line_y'] = (self.df['x'] == start_value).cumsum() - 1

        y_id = self.df['line_y'].drop_duplicates()
        self.y_max_id = int(y_id.max() + 1)

    def add_x_id(self):
        x_max = self.df['x'].size / (self.y_max_id)
        try:
            self.x_max_id = int(x_max)
        except Exception as e:
            print(f"Number of Columns of the Matrix is not an integer.")  # ToDo: schönere Exception schreiben

    def get_pixel_size(self):
        return self.pixel_size  # Umrechnungsding

    def calc_pixel_size(self):
        y_pixel_size = self.df['y'].values[-1] / self.y_max_id
        x_pixel_size = self.df['x'].values[-1] / self.x_max_id
        diff = y_pixel_size - x_pixel_size
        if diff > 10e-3:
            print(f"Difference in pixel size: {diff}")
        self.pixel_size = (y_pixel_size + x_pixel_size) / 2

    def as_matrix(self) -> np.array:
        return self.df_matrix

    def _add_matrix(self):
        # ToDo: Kontrollieren warum erst y_max_id kommt und dann x - ich weiß dass es irgendwo vorher ein problem mit den dimensionen gab und das den fehler gelöst hat, aber ist das jetzt noch richtig von der orientierung?
        self.df_matrix = self.df['z_no_nan'].__array__().reshape(self.y_max_id, self.x_max_id)
        self.df_matrix_as_matrix = np.matrix(self.df_matrix)

    @property
    def __y_max_matrix(self):
        return self.y_max_id

    @property
    def __x_max_matrix(self):
        return self.x_max_id

    def plot_height_3D_complete(self) -> plt.figure:
        x = np.arange(0, self.x_max_id)
        y = np.arange(0, self.y_max_id)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X=X, Y=Y, Z=data_aspherical_left.as_matrix(), cmap='viridis')
        plt.show()
        return fig

    def plot_height_3D(self) -> plt.figure:
        x = np.arange(self.structure_loc[0][0], self.structure_loc[1][0])
        y = np.arange(self.structure_loc[0][1], self.structure_loc[1][1])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        matrix = data_aspherical_left.as_matrix()
        matrix = matrix[self.structure_loc[0][1]:self.structure_loc[1][1],
                 self.structure_loc[0][0]:self.structure_loc[1][0]]

        ax.plot_surface(X=X, Y=Y, Z=matrix, cmap='viridis')
        plt.show()
        return fig

    def get_x_profile(self, index) -> np.array:
        """
        get height-profile of the x axis at y-coordinate y_index
        """
        return self.df_matrix[index, :]

    def get_y_profile(self, index) -> np.array:
        """
        get height-profile of the y axis at x-coordinate x_index
        """
        return self.df_matrix[:, index]

    def plot_profile(self, axis, index, plot_axes: plt.axes = None, return_data=False) -> plt.figure:
        if axis.lower() == 'x':
            data = self.get_x_profile(index)
        elif axis.lower() == 'y':
            data = self.get_y_profile(index)
        else:
            raise NotImplementedError(f"Axis {axis} is not implemented.")

        fig = plt.figure()
        if plot_axes is not None:
            plot_axes.plot(data)
        else:
            ax = fig.add_subplot()
            ax.plot(data)
            plt.show()
        if return_data:
            return fig, data
        else:
            return fig

    def calc_structure_information(self, threshold=0.5):
        # ToDo: mean abfrage und dadurch auch threshold überdenken und ggf. ersetzen
        lower_val = np.mean(self.get_x_profile(index=0)) if np.mean(self.get_x_profile(index=0)) <= np.mean(
            self.get_y_profile(index=0)) else np.mean(self.get_y_profile(index=0))
        # high_val = np.mean(self.get_x_profile(index=int(np.ceil(self.x_max_id / 2)))) if np.mean(
        #     self.get_x_profile(index=int(np.ceil(self.x_max_id / 2)))) >= np.mean(
        #     self.get_y_profile(index=int(np.ceil(self.y_max_id / 2)))) else np.mean(
        #     self.get_y_profile(index=int(np.ceil(self.y_max_id / 2))))
        high_val_x = np.mean(self.get_x_profile(index=int(np.ceil(self.x_max_id / 2))))
        high_val_y = np.mean(self.get_y_profile(index=int(np.ceil(self.y_max_id / 2))))
        start = True
        for i in range(self.y_max_id):
            z_mean = self.get_x_profile(index=i).mean()
            if z_mean >= threshold:
                if start:
                    start = False
                    y_struct_index_start = i
                y_struct_index_end = i
        start = True
        for i in range(self.x_max_id):
            z_mean = self.get_y_profile(index=i).mean()
            if z_mean >= threshold:
                if start:
                    start = False
                    x_struct_index_start = i
                x_struct_index_end = i

        # structure locations
        loc = [[x_struct_index_start, y_struct_index_start], [x_struct_index_end, y_struct_index_end]]
        mid_x = int(np.round((loc[0][0] + loc[1][0]) / 2))
        mid_y = int(np.round((loc[0][1] + loc[1][1]) / 2))
        # structure information
        self.length = self.get_pixel_size() * (loc[1][0] - loc[0][0])  # length equals the path in x-direction
        self.width = self.get_pixel_size() * (loc[1][1] - loc[0][1])  # length equals the path in y-direction
        # saving structure location
        self.structure_loc = loc
        self.structure_loc_mid = np.array([mid_x, mid_y])


if __name__ == "__main__":
    base_path = 'C:\\Users\\hanne\\Documents\\Seafile\\Nanoproduction_Hannes\\Sensofar'
    experiment = "2024_06_25 DHM Paper Pillow test\Data"
    path = os.path.join(base_path, experiment)
    # data_name = "Aspherical_left_structureonly.dat"
    data_name = "Aspherical_left_good.dat"
    data_path = os.path.join(base_path, experiment, data_name)

    data_aspherical_left = SensofarData(data_path)

    # plotting the middle crosssection in x-direction
    fig, data = data_aspherical_left.plot_profile(axis="y", index=data_aspherical_left.structure_loc_mid[1],
                                                  return_data=True)
