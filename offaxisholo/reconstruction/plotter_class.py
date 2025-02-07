import json
import os
from typing import Literal

from matplotlib import pyplot as plt
import numpy as np
from numpy import save


# ToDo - rework single image and height plotting


class DHMPlotter:
    def __init__(self, img_path):
        self.img_save_path = img_path
        # ToDo: Implement in the functions below what to do if img_ave_path is none

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically
        #  determine a subfolder /img/ for the images - maybe done in the future for the complete structure_dhm class

    def plotImage_old(self, img, title=None, save=False, save_path=None, cmap='viridis'):
        if save_path is None:
            save_path = self.img_save_path
        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(save_path, name)
                plt.imsave(save_path, img, cmap=cmap)
            else:
                save_path = os.path.join(save_path, f"picture{self.var_4_saving}.png")
                plt.imsave(save_path, img, cmap=cmap)
                self.var_4_saving += 1
            plt.close()
        else:
            if title == None:
                plt.imshow(img, cmap=cmap)
            else:
                plt.imshow(img, cmap=cmap)
                plt.title(title)
            plt.show()  # show image
        return

    def plot_height_old(self, height_profile, title=None, save=False, save_path=None, legend_bar=True, cmap='coolwarm',
                        pixel_pitch=None):
        """
        Plotting of the reconstructed height profile. Make sure the dimensions of the height profile matches the
        dimensions of the hologram.

        height_profile:  Height profile of the image
        title:           Title of the image. If save then this will also be the name of the saved image.
        save:            Boolean if it should be saved. If False then it will be shown.
        legend_bar:      Boolean if the color bar should be shown.
        """
        if save_path is None:
            save_path = self.img_save_path
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Using linspace to generate exactly 1024 points in each direction
        if pixel_pitch is None:
            if isinstance(self.pixel_pitch, float):
                x_px_sz = y_px_sz = self.pixel_pitch
            else:
                x_px_sz = self.pixel_pitch[0]
                y_px_sz = self.pixel_pitch[1]
        else:
            if isinstance(self.pixel_pitch, float):
                x_px_sz = y_px_sz = pixel_pitch
            else:
                x_px_sz = pixel_pitch[0]
                y_px_sz = pixel_pitch[1]

        X = np.linspace(0, (self.hologram.shape[1] - 1) * y_px_sz, self.hologram.shape[1])
        Y = np.linspace(0, (self.hologram.shape[0] - 1) * x_px_sz, self.hologram.shape[0])
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
                save_path = os.path.join(save_path, name)
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
            else:
                save_path = os.path.join(save_path, f"3D_plot_{self.var_4_saving}.png")
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
        SAVE_NAME = os.path.join(path, name + ".npy")

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

    def check_reonstruction_flag(self, holo_class):
        # todo implement a query and check in hologram and processor implementation of reconstruction_flag
        # true for finished, false for not reconstructed
        pass

    def plot_spatial_filtering(self, holo_class, mode: Literal["Hologram", "Processor"] = "Processor",
                               show_plot=False, save_single=False, cmap='gray', save_title=None):
        if mode == "Hologram":
            class_object = holo_class
        elif mode == "Processor":
            class_object = holo_class.hologram
        else:
            raise ValueError("Mode must be either Hologram or Processor.")

        # check if reconstruction is already done
        if not self.check_reonstruction_flag(holo_class):
            holo_class.run()

        # plotting
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Image Reconstruction Steps', fontsize=16)
        axs[0, 0].imshow(class_object.data, cmap=cmap)
        axs[0, 0].set_title("Original Hologram Image")
        axs[1, 0].imshow(class_object.intensity(class_object.spectrum), cmap=cmap)
        axs[1, 0].set_title("Spectrum of the captured hologram")
        axs[0, 1].imshow(class_object.intensity(class_object.spectrum_shifted), cmap=cmap)
        axs[0, 1].set_title("Shifted spectrum")
        axs[1, 1].imshow(class_object.intensity(class_object.spectrum_masked), cmap=cmap)
        axs[1, 1].set_title("Masked spectrum")
        axs[0, 2].imshow(class_object.reconstructed_phase, cmap=cmap)
        axs[0, 2].set_title("Reconstructed phase image")
        axs[1, 2].imshow(class_object.reconstructed_intensity, cmap=cmap)
        axs[1, 2].set_title("Reconstructed intensity image")

        if show_plot:
            plt.show()

        if self.img_save_path is not None:
            if save_title is not None:
                name = "Spatial Filtering" + save_title  # todo: ok like this or should it only be the savetitle
            else:
                name = "Spatial Filtering"
            save_path = os.path.join(self.img_save_path, f"{name}.png")
            fig.savefig(save_path, dpi=600)

            # todo implement this with a for loop
            # if save_single:
            #     extent = axs[0,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            #     fig.savefig("ax1_figure.png", bbox_inches=extent)
            #     fig.savefig("ax1_figure.png", bbox_inches=extent.expanded(1.1,1.1))  # 10% extent in x and y direction

    def plot_reconstruction_short(self, holo_class, show_plot=False, save_single=False, cmap='gray', save_title=None):
        # check if reconstruction is already done
        if not self.check_reonstruction_flag(holo_class):
            holo_class.run()

        # plotting
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle('Hologram Reconstruction', fontsize=16)
        axs[0, 0].imshow(holo_class.hologram.data, cmap=cmap)
        axs[0, 0].set_title("Original Hologram Image")
        axs[1, 0].imshow(holo_class.background.data, cmap=cmap)
        axs[1, 0].set_title("Background hologram")
        axs[0, 1].imshow(holo_class.phase_map, cmap=cmap)
        axs[0, 1].set_title("Reconstructed and unwrapped phase image")
        axs[1, 1].imshow(holo_class.intensity_reconstructed, cmap=cmap)
        axs[1, 1].set_title("Reconstructed intensity image")

        if show_plot:
            plt.show()

        if self.img_save_path is not None:
            if save_title is not None:
                name = "Hologram reconstruction" + save_title  # todo: ok like this or should it only be the savetitle
            else:
                name = "Hologram reconstruction"
            save_path = os.path.join(self.img_save_path, f"{name}.png")
            fig.savefig(save_path, dpi=600)

            # todo implement this with a for loop
            # if save_single:
            #     extent = axs[0,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            #     fig.savefig("ax1_figure.png", bbox_inches=extent)
            #     fig.savefig("ax1_figure.png", bbox_inches=extent.expanded(1.1,1.1))  # 10% extent in x and y direction

    def plot_full_reconstruction_process(self, holo_class, show_plot=False, save_single=False, cmap='gray',
                                         save_title=None):
        # check if reconstruction is already done
        if not self.check_reonstruction_flag(holo_class):
            holo_class.run()

        # plotting
        fig, axs = plt.subplots(2, 6, figsize=(16, 9))
        fig.suptitle('Hologram Reconstruction', fontsize=16)
        # Holo and Background
        axs[0, 0].imshow(holo_class.hologram.data, cmap=cmap)
        axs[0, 0].set_title("Original \nhologram Image")
        axs[1, 0].imshow(holo_class.background.data, cmap=cmap)
        axs[1, 0].set_title("Background \nhologram")
        # Spectrum
        axs[0, 1].imshow(holo_class.intensity(holo_class.hologram.spectrum), cmap=cmap)
        axs[0, 1].set_title("Spectrum of \ncaptured hologram")
        axs[1, 1].imshow(holo_class.intensity(holo_class.hologram.spectrum_masked), cmap=cmap)
        axs[1, 1].set_title("Shifted and \nmasked spectrum")
        # Reconstruction after spatial filtering
        axs[0, 2].imshow(holo_class.hologram.reconstructed_intensity, cmap=cmap)
        axs[0, 2].set_title("Reconstructed \nIntensity")
        axs[1, 2].imshow(holo_class.hologram.reconstructed_phase, cmap=cmap)
        axs[1, 2].set_title("Reconstructed \nphase")
        # Reconstruction after propagation
        axs[0, 3].imshow(holo_class.intensity(holo_class.field_propagated), cmap=cmap)
        axs[0, 3].set_title("Intensity after \npropagation")
        axs[1, 3].imshow(holo_class.phase(holo_class.field_propagated), cmap=cmap)
        axs[1, 3].set_title("Phase after \npropagation")
        # Reconstruction after compensation
        axs[0, 4].imshow(holo_class.intensity_compensated_db, cmap=cmap)
        axs[0, 4].set_title("Intensity after \ncompensation")
        axs[1, 4].imshow(holo_class.phase_compensated, cmap=cmap)
        axs[1, 4].set_title("Phase after compensation \nand unwrapping")
        # Reconstruction after filtering - final step
        axs[0, 5].imshow(holo_class.intensity_reconstructed, cmap=cmap)
        axs[0, 5].set_title("Final intensity \nimage")
        axs[1, 5].imshow(holo_class.phase_map, cmap=cmap)
        axs[1, 5].set_title("Final phase \nimage")

        if self.img_save_path is not None:
            if save_title is not None:
                name = "Complete hologram reconstruction" + save_title  # todo: ok like this or should it only be the savetitle
            else:
                name = "Complete hologram reconstruction"
            save_path = os.path.join(self.img_save_path, f"{name}.png")
            fig.savefig(save_path, dpi=600)

            # todo implement this with a for loop - iteration over axes
            # if save_single:
            #     extent = axs[0,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            #     fig.savefig("ax1_figure.png", bbox_inches=extent)
            #     fig.savefig("ax1_figure.png", bbox_inches=extent.expanded(1.1,1.1))  # 10% extent in x and y direction

