import os
from enum import Enum
from pathlib import Path
from typing import Union, Literal

import numpy as np
from dataclasses import dataclass

import pandas as pd
from matplotlib import pyplot as plt


class StructureAxis(Enum):
    X = "X"
    Y = "Y"
    # def flip(self):


@dataclass
class NumpyArrayDataClass:
    data: np.ndarray
    dtype: np.dtype
    shape: tuple

    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def x_dim(self):
        return self.shape[1]

    @property
    def y_dim(self):
        return self.shape[0]

    @property
    def size(self):
        return self.data.size

    def locate_structure(self, threshold=0.5):
        """
        Locate the structure in a 2D data array by identifying regions where there is a
        significant change in values along the x and y axes.

        Parameters:
        threshold (float): The minimum difference between consecutive elements required to identify
                           the boundary of the structure.

        Returns:
        list: A 2D list with y and x start and end points of the structure: [[y_begin, y_end], [x_begin, x_end]]
        """
        # Precompute absolute values of data to avoid redundant calculations
        abs_data = np.abs(self.data)

        # Midpoints of the data array
        y_mid = self.shape[0] // 2
        x_mid = self.shape[1] // 2

        # Initialize boundaries
        y_begin, y_end = None, None
        x_begin, x_end = None, None

        # Find y_begin and y_end by scanning from both top and bottom towards the middle
        for i in range(y_mid + 1):
            if y_begin is None and abs_data[i, x_mid] - abs_data[i + 1, x_mid] >= threshold:
                y_begin = i
            if y_end is None and abs_data[self.shape[0] - 1 - i, x_mid] - abs_data[
                self.shape[0] - 2 - i, x_mid] >= threshold:
                y_end = self.shape[0] - 1 - i
            if y_begin is not None and y_end is not None:
                break  # Exit once both boundaries are found

        # Find x_begin and x_end by scanning from both left and right towards the middle
        for i in range(x_mid + 1):
            if x_begin is None and abs_data[y_mid, i] - abs_data[y_mid, i + 1] >= threshold:
                x_begin = i
            if x_end is None and abs_data[y_mid, self.shape[1] - 1 - i] - abs_data[
                y_mid, self.shape[1] - 2 - i] >= threshold:
                x_end = self.shape[1] - 1 - i
            if x_begin is not None and x_end is not None:
                break  # Exit once both boundaries are found

        # Return the located structure's bounds
        return [[y_begin, y_end], [x_begin, x_end]]


@dataclass
class StructureDataClass(NumpyArrayDataClass):
    def __init__(self, data, pixel_size):
        if isinstance(data, str) or isinstance(data, Path):
            try:
                data = open(data, mode="r+")
                data = np.asarray(data)
            except Exception as e:
                raise e
        elif isinstance(data, pd.DataFrame):
            data = pd.read_csv(data, delimiter=';')
            data = np.asarray(data)

        assert isinstance(data, np.ndarray)  # necessary for initialisation of the parent class
        super().__init__(data)

        self.quadratic_pixel = False
        if isinstance(pixel_size, tuple) or isinstance(pixel_size, list) or isinstance(pixel_size, np.ndarray):
            self.px_size_x = pixel_size[0]
            self.px_size_y = pixel_size[1]
            if self.px_size_x == self.px_size_y:
                self.pixel_size = self.px_size_x
                self.quadratic_pixel = True
        else:
            self.pixel_size = pixel_size
            self.quadratic_pixel = True

        # Detect and change the structure's bound to a variable
        self.structure_location = self.locate_structure()

    def get_data(self):
        return self.data

    def get_profile(self, axis: StructureAxis, location: Union[Literal['mid'], int]):
        if axis == StructureAxis.X:
            if isinstance(location, int):
                assert 0 <= location < self.x_dim
                data = self.data[:, location]
            else:  # location had to be set to 'mid'
                data = self.data[:, self.x_dim//2-1]
        elif axis == StructureAxis.Y:
            if isinstance(location, int):
                assert 0 <= location < self.y_dim
                data = self.data[:, location]
            else:  # location had to be set to 'mid'
                data = self.data[:, self.y_dim//2-1]
        else:
            raise NotImplementedError(f"No Axis {StructureAxis} defined.")

        # Return the profile
        return data

    def plot_height(self, save_path, structure_name: str, meas_tech: str, cmap: str = 'coolwarm'):
        """
        Plotting of the reconstructed height profile. Make sure the unit of pixel pitch and data are matching.
        """
        title = "3D height plot of " + structure_name + " measured with " + meas_tech
        title_2d = "Surface plot of " + structure_name + " measured with " + meas_tech
        x_label = "Length [µm]"
        y_label = "Width [µm]"
        z_label = "Height [µm]"
        x_label_2d = "Length of " + structure_name + " [px]"
        y_label_2d = "Width of " + structure_name + " [px]"

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Using linspace to generate exactly 1024 points in each direction
        if self.quadratic_pixel:
            X = np.linspace(0, (self.data.shape[1] - 1) * self.pixel_size, self.data.shape[1])
            Y = np.linspace(0, (self.data.shape[0] - 1) * self.pixel_size, self.data.shape[0])
        else:
            X = np.linspace(0, (self.data.shape[1] - 1) * self.px_size_x, self.data.shape[1])
            Y = np.linspace(0, (self.data.shape[0] - 1) * self.px_size_y, self.data.shape[0])
        # Creating the meshgrid
        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, self.data, cmap=cmap,
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.suptitle(title)
        # ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        name = title.replace(" ", "_") + '.png'
        save_path_3d = os.path.join(save_path, name)
        plt.savefig(fname=save_path_3d, dpi=400)  # transparent=True)

        fig_2d, ax_2d = plt.subplots()

        fig_2d.suptitle(title)
        # ax.set_title(title)
        ax_2d.set_xlabel(x_label_2d)
        ax_2d.set_ylabel(y_label_2d)

        fig_2d.colorbar(surf, shrink=0.5, aspect=5)
        name2d = title_2d.replace(" ", "_") + '.png'
        save_path2d = os.path.join(save_path, name2d)
        plt.imshow(self.data, cmap=cmap)
        plt.savefig(fname=save_path2d, dpi=400)  # transparent=True)




"""
REDUNDANT METHOD


def plot_3d_height(data, pixel_pitch, save_path, structure_name: str, meas_tech: str, cmap: str = 'coolwarm'):
    
    #Plotting of the reconstructed height profile. Make sure the unit of pixel pitch and data are matching.
    
    if isinstance(pixel_pitch, int) or isinstance(pixel_pitch, float):
        pixel_pitch = [pixel_pitch, pixel_pitch]
    title = "3D height plot of " + structure_name + " measured with " + meas_tech
    title_2d = "Surface plot of " + structure_name + " measured with " + meas_tech
    x_label = "Length [µm]"
    y_label = "Width [µm]"
    z_label = "Height [µm]"
    x_label_2d = "Length of" + structure_name + " [µm]"
    y_label_2d = "Width of" + structure_name + " [µm]"

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Using linspace to generate exactly 1024 points in each direction
    X = np.linspace(0, (data.shape[1] - 1) * pixel_pitch[1], data.shape[1])
    Y = np.linspace(0, (data.shape[0] - 1) * pixel_pitch[0], data.shape[0])
    # Creating the meshgrid
    X, Y = np.meshgrid(X, Y)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, data, cmap=cmap,
                           linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.subtitle(title)
    # ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    name = title.replace(" ", "_") + '.png'
    save_path_3d = os.path.join(save_path, name)
    plt.savefig(fname=save_path_3d, dpi=400)  # transparent=True)

    fig_2d, ax_2d = plt.subplots()

    fig_2d.subtitle(title)
    # ax.set_title(title)
    ax_2d.set_xlabel(x_label_2d)
    ax_2d.set_ylabel(y_label_2d)

    fig_2d.colorbar(surf, shrink=0.5, aspect=5)
    fig_2d.subtitle(title)
    name2d = title_2d.replace(" ", "_") + '.png'
    save_path2d = os.path.join(save_path, name2d)
    plt.savefig(fname=save_path2d, dpi=400)  # transparent=True)

"""