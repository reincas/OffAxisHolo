import json
import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import save


class DHMPlotter:
    def __init__(self, img_path=None):
        self.img_save_path = img_path
        # ToDo: Implement in the functions below what to do if img_ave_path is none

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically
        #  determine a subfolder /img/ for the images - maybe done in the future for the complete structure class

    def intensity(self, complex_field, mode='linear'):
        """
        Calculate intensity from complex field

        Parameters:
        -----------
        inp : ndarray
            Input complex field
        mode : str
            'linear' for |E|^2
            'db' for intensity in decibels

        Returns:
        --------
        intensity : ndarray
            Calculated intensity
        """
        # Calculate absolute square
        intensity = np.abs(complex_field) ** 2
        eps = 1e-12

        if mode.lower() == 'db':
            # Convert to dB scale (10 log10 for intensity)
            intensity = 10 * np.log10(intensity + eps)
        return intensity

    def phase(self, inp):
        """
        Calculate phase from complex field

        Parameters:
        -----------
        inp : ndarray
            Input complex field
        unwrap : bool
            Whether to unwrap the phase

        Returns:
        --------
        phase : ndarray
            Calculated phase in radians
        """
        phase = np.angle(inp)
        return phase

    def calculate_efield(self, intensity, phase, intensity_is_db=False):
        """
        Calculate complex electromagnetic field from intensity and phase.

        Parameters:
        -----------
        intensity : numpy.ndarray
            2D array of intensity values or intensity differences
        phase : numpy.ndarray
            2D array of phase values in radians
        intensity_is_db : bool
            Whether intensity is in dB scale

        Returns:
        --------
        numpy.ndarray
            Complex 2D array of E-field
        """
        if intensity.shape != phase.shape:
            raise ValueError("Intensity and phase arrays must have same shape")

        if np.any(np.isnan(intensity)) or np.any(np.isnan(phase)):
            raise ValueError("Input arrays contain NaN values")

        if np.any(np.isinf(intensity)) or np.any(np.isinf(phase)):
            raise ValueError("Input arrays contain infinite values")

        if intensity_is_db:
            intensity = 10 ** (intensity / 10)

        # Calculate complex field (no negative intensity check)
        amplitude = np.sqrt(np.abs(intensity)) * np.sign(intensity)
        efield = amplitude * np.exp(1j * phase)

        return efield

    def plotImage(self, img, title=None, save=False, cmap='viridis'):
        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(self.img_save_path, name)
                plt.imsave(save_path, img, cmap=cmap)
            else:
                save_path = os.path.join(self.img_save_path, f"picture{self.var_4_saving}.png")
                plt.imsave(save_path, img, cmap=cmap)
                self.var_4_saving += 1
        else:
            if title == None:
                plt.imshow(img, cmap=cmap)
            else:
                plt.imshow(img, cmap=cmap)
                plt.title(title)
            plt.show()  # show image
        return

    def plot_height(self, height_profile, title=None, save=False, legend_bar=True, cmap='coolwarm'):
        """
        Plotting of the reconstructed height profile. Make sure the dimensions of the height profile matches the
        dimensions of the hologram.

        height_profile:  Height profile of the image
        title:           Title of the image. If save then this will also be the name of the saved image.
        save:            Boolean if it should be saved. If False then it will be shown.
        legend_bar:      Boolean if the color bar should be shown.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Using linspace to generate exactly 1024 points in each direction
        X = np.linspace(0, (self.hologram.shape[1] - 1) * self.pixel_pitch[1], self.hologram.shape[1])
        Y = np.linspace(0, (self.hologram.shape[0] - 1) * self.pixel_pitch[0], self.hologram.shape[0])
        # Creating the meshgrid
        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, height_profile, cmap=cmap,
                               linewidth=0, antialiased=False)
        if title is not None:
            plt.title(title)

        if legend_bar:
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(self.img_save_path, name)
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
            else:
                save_path = os.path.join(self.img_save_path, f"3D_plot_{self.var_4_saving}.png")
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
                self.var_4_saving += 1
        else:
            plt.show()

    def save(self, path, name: str = None, data: np.ndarray = None):
        """
        Saves the numpy array of the reconstructed field.
        """
        # ToDo rework position of implementation
        if name is None:
            name = "field_reconstructed"
        INFORMATION_FILE = name + "_information.txt"
        INFORMATION_PATH = os.path.join(path, INFORMATION_FILE)
        SAVE_NAME = os.path.join(path, name+".npy")

        if data is None:
            data = self.field_reconstructed
        information = {
            "Description": f"Reconstructed field of a DHM image.",
            "Wavelength": self.wavelength,
            "Pixel size": self.pixel_pitch,
            "Propagation distance": self.propagation_distance,
            "Refractive index": self.n_resin,
            "format": data.dtype.name,
        }
        with open(INFORMATION_PATH, 'w', encoding='utf-8') as file:
            file.write(json.dumps(information, sort_keys=True, indent=4))
        save(SAVE_NAME, data)


# _____________________________________________________________________________________
# OLD IMPLEMENTATIONS
# def intensity(self, inp, log=True):
#     out = np.abs(inp)
#     if not log:
#         out = out * out
#     else:
#         out = 20 * np.log(out)
#         out[out == np.inf] = 0
#         out[out == -np.inf] = 0
#     return out

# def phase(self, inp):
#     out = np.angle(inp)
#     return out

# def calculate_efield(self, intensity, phase):
#     """
#     Calculate the complex electromagnetic field from intensity and phase information.
#
#     Parameters:
#     -----------
#     intensity : numpy.ndarray
#         2D array containing the intensity information
#     phase : numpy.ndarray
#         2D array containing the phase information in radians
#
#     Returns:
#     --------
#     numpy.ndarray
#         Complex 2D array representing the electromagnetic field
#
#     Notes:
#     ------
#     The electromagnetic field E is calculated using:
#     E = sqrt(I) * exp(iφ)
#     where I is the intensity and φ is the phase
#     """
#
#     # Verify inp arrays have same shape
#     if intensity.shape != phase.shape:
#         raise ValueError("Intensity and phase arrays must have the same shape")
#
#     # Check for negative intensity values
#     if np.any(intensity < 0):
#         raise ValueError("Intensity values cannot be negative")
#
#     # Calculate amplitude as square root of intensity
#     amplitude = np.sqrt(intensity)
#
#     # Calculate complex E-field
#     efield = amplitude * np.exp(1j * phase)
#
#     return efield
