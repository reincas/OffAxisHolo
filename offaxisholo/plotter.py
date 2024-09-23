import os
from matplotlib import pyplot as plt
import numpy as np


class DHMPlotter:
    def __init__(self, img_path=None):
        self.img_save_path = img_path
        # ToDo: Implement in the functions below what to do if img_ave_path is none

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically
        #  determine a subfolder /img/ for the images - maybe done in the future for the complete structure class

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
