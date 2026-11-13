"""
Plotting of holograms and of the individual reconstruction steps.

All plot methods take either a Hologram or a HologramProcessor object
("holo_class") and visualise its stored arrays. If the reconstruction has not
been done yet, it is started automatically.

Arrays of the GPU backend are converted to NumPy (to_numpy) before plotting or
saving.
"""

import json
import os
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

from ..backend import to_numpy


class DHMPlotter:
    def __init__(self, img_path=None):
        """
        img_path: directory in which the images are saved. If None, plots are
                  only shown and not saved.
        """
        self.img_save_path = img_path
        self._image_counter = 0  # used for file names of untitled images

    def set_save_path(self, path):
        """Set the directory in which the images are saved."""
        self.img_save_path = path

    # ------------------------------------------------------------------ helpers
    def _require_save_path(self, save_path=None):
        """Return a valid save directory or raise a ValueError."""
        path = save_path if save_path is not None else self.img_save_path
        if path is None:
            raise ValueError(
                "No save path given. Use set_save_path() or pass save_path."
            )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _ensure_reconstructed(holo_class):
        """
        Make sure that the reconstruction of the given object is finished.

        Hologram:          reconstruct() is called if no field is available.
        HologramProcessor: run() is called if no phase map is available.
        """
        if hasattr(holo_class, "phase_map"):  # HologramProcessor
            if holo_class.phase_map is None:
                holo_class.run()
        elif not holo_class.finished_reconstruction:  # Hologram
            holo_class.reconstruct()

    def _finish_figure(self, fig, file_name, show_plot, dpi=600):
        """Save the figure (if a save path is set), then show or close it."""
        if self.img_save_path is not None:
            os.makedirs(self.img_save_path, exist_ok=True)
            fig.savefig(
                os.path.join(self.img_save_path, f"{file_name}.png"),
                dpi=dpi,
                bbox_inches="tight",
            )
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # -------------------------------------------------------------- single plots
    def plot_image(self, img, title=None, save=False, save_path=None, cmap="viridis"):
        """
        Plot a single 2D array.

        save=True:  the array is saved as image (plt.imsave, one pixel per array
                    element). The file name is derived from the title.
        save=False: the image is shown in a matplotlib window.
        """
        if save:
            save_path = self._require_save_path(save_path)
            if title is not None:
                name = title.replace(" ", "_") + ".png"
            else:
                name = f"picture_{self._image_counter}.png"
                self._image_counter += 1
            plt.imsave(os.path.join(save_path, name), to_numpy(img), cmap=cmap)
        else:
            fig, ax = plt.subplots()
            ax.imshow(to_numpy(img), cmap=cmap)
            if title is not None:
                ax.set_title(title)
            plt.show()

    def plot_height(
        self,
        height_profile,
        pixel_pitch,
        title=None,
        save=False,
        save_path=None,
        legend_bar=True,
        cmap="coolwarm",
    ):
        """
        3D surface plot of a reconstructed height profile.

        height_profile: 2D array with the height in metres.
        pixel_pitch:    pixel pitch in the object plane in metres, either a single
                        value or [x pitch, y pitch].
        title:          title of the image. If save=True this is also the file name.
        save:           True saves the plot, False shows it.
        legend_bar:     show the colour bar.
        """
        if isinstance(pixel_pitch, (float, int)):
            x_px_sz = y_px_sz = pixel_pitch
        else:
            x_px_sz, y_px_sz = pixel_pitch[0], pixel_pitch[1]

        height_profile = to_numpy(height_profile)
        rows, cols = height_profile.shape
        X, Y = np.meshgrid(
            np.arange(cols) * x_px_sz,
            np.arange(rows) * y_px_sz,
        )

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            X, Y, height_profile, cmap=cmap, linewidth=0, antialiased=False
        )
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_zlabel("height / m")
        if title is not None:
            ax.set_title(title)
        if legend_bar:
            fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            save_path = self._require_save_path(save_path)
            if title is not None:
                name = title.replace(" ", "_") + ".png"
            else:
                name = f"3D_plot_{self._image_counter}.png"
                self._image_counter += 1
            fig.savefig(os.path.join(save_path, name), dpi=400)
            plt.close(fig)
        else:
            plt.show()

    def save_individual_plot(
        self,
        data,
        filename,
        cmap="gray",
        figsize=(8, 6),
        dpi=300,
        title=None,
        save=False,
        show=False,
    ):
        """Plot a single array in its own figure and save and/or show it."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(to_numpy(data), cmap=cmap)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        if save:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def save_array(self, path, data: np.ndarray, name: str = None, information=None):
        """
        Save a numpy array (.npy) together with a json text file containing
        additional information (e.g. wavelength, pixel pitch, propagation
        distance, refractive index).

        path:        target directory
        data:        array to be saved
        name:        file name without extension (default 'field_reconstructed')
        information: dictionary with additional information
        """
        if name is None:
            name = "field_reconstructed"
        data = to_numpy(data)
        os.makedirs(path, exist_ok=True)
        info = {"format": data.dtype.name, "shape": list(data.shape)}
        if information is not None:
            info.update(information)
        with open(
            os.path.join(path, name + "_information.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(json.dumps(info, sort_keys=True, indent=4, default=str))
        np.save(os.path.join(path, name + ".npy"), data)

    # --------------------------------------------------------- overview plots
    def plot_spatial_filtering(
        self,
        holo_class,
        mode: Literal["Hologram", "Processor"] = "Processor",
        show_plot=False,
        cmap="gray",
        save_title=None,
    ):
        """
        Overview of the spatial filtering (Fourier transform method):
        hologram, spectrum, shifted spectrum, masked spectrum and the resulting
        wrapped phase and intensity in the hologram plane.

        mode: 'Hologram' if holo_class is a Hologram, 'Processor' if it is a
              HologramProcessor.
        """
        if mode == "Hologram":
            hologram = holo_class
        elif mode == "Processor":
            hologram = holo_class.hologram
        else:
            raise ValueError("Mode must be either Hologram or Processor.")

        self._ensure_reconstructed(hologram)

        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Spatial filtering", fontsize=16)
        axs[0, 0].imshow(to_numpy(hologram.data), cmap=cmap)
        axs[0, 0].set_title("Original hologram image")
        axs[1, 0].imshow(to_numpy(hologram.intensity(hologram.spectrum)), cmap=cmap)
        axs[1, 0].set_title("Spectrum of the hologram (dB)")
        axs[0, 1].imshow(
            to_numpy(hologram.intensity(hologram.spectrum_shifted)), cmap=cmap
        )
        axs[0, 1].set_title("Shifted spectrum (dB)")
        axs[1, 1].imshow(
            to_numpy(hologram.intensity(hologram.spectrum_masked)), cmap=cmap
        )
        axs[1, 1].set_title("Masked spectrum (dB)")
        axs[0, 2].imshow(to_numpy(hologram.reconstructed_phase), cmap=cmap)
        axs[0, 2].set_title("Reconstructed phase (wrapped)")
        axs[1, 2].imshow(to_numpy(hologram.reconstructed_intensity), cmap=cmap)
        axs[1, 2].set_title("Reconstructed intensity (dB)")

        name = "Spatial_filtering" + (f"_{save_title}" if save_title else "")
        self._finish_figure(fig, name, show_plot)

    def plot_reconstruction_short(
        self,
        holo_class,
        show_plot=False,
        save_single=False,
        cmap="gray",
        save_title=None,
        compensation=True,
    ):
        """
        Short overview of a HologramProcessor: hologram, background hologram (if
        compensation was used), final unwrapped phase and final intensity.

        save_single: additionally save every panel as an individual image
                     (requires a save path).
        """
        self._ensure_reconstructed(holo_class)
        compensation = compensation and holo_class.background is not None

        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle("Hologram reconstruction", fontsize=16)
        axs[0, 0].imshow(to_numpy(holo_class.hologram.data), cmap="gray")
        axs[0, 0].set_title("Original hologram image")
        if compensation:
            axs[1, 0].imshow(to_numpy(holo_class.background.data), cmap="gray")
            axs[1, 0].set_title("Background hologram")
        else:
            axs[1, 0].axis("off")
        axs[0, 1].imshow(to_numpy(holo_class.phase_map), cmap=cmap)
        axs[0, 1].set_title("Reconstructed and unwrapped phase")
        axs[1, 1].imshow(to_numpy(holo_class.intensity_reconstructed), cmap="gray")
        axs[1, 1].set_title("Reconstructed intensity (dB)")

        if save_single:
            save_path = self._require_save_path()
            singles = [
                (holo_class.hologram.data, "hologram_original.png", "gray"),
                (holo_class.phase_map, "phase_map.png", cmap),
                (
                    holo_class.intensity_reconstructed,
                    "intensity_reconstructed.png",
                    "gray",
                ),
            ]
            if compensation:
                singles.append(
                    (holo_class.background.data, "hologram_background.png", "gray")
                )
            for data, file_name, colormap in singles:
                self.save_individual_plot(
                    data,
                    os.path.join(save_path, file_name),
                    cmap=colormap,
                    save=True,
                )

        name = "Hologram_reconstruction" + (f"_{save_title}" if save_title else "")
        self._finish_figure(fig, name, show_plot)

    def plot_full_reconstruction_process(
        self,
        holo_class,
        show_plot=False,
        save_single=False,
        cmap="gray",
        save_title=None,
    ):
        """
        Overview of every step of a HologramProcessor reconstruction, in
        processing order (columns):

            1. hologram / background hologram
            2. spectrum / shifted and masked spectrum
            3. intensity / wrapped phase after spatial filtering (hologram plane)
            4. intensity / wrapped phase after aberration compensation
            5. intensity / wrapped phase after propagation
            6. final intensity / unwrapped phase

        Steps that were not performed (e.g. no background) are left empty.

        save_single: additionally save every panel as an individual image.
        """
        self._ensure_reconstructed(holo_class)
        hologram = holo_class.hologram
        background = holo_class.background
        compensated = holo_class.field_compensated is not None

        def optional(condition, data):
            return data if condition else None

        # (data, title) for row 0 and row 1, column by column
        plot_configs = [
            (hologram.data, "Original hologram image"),
            (hologram.intensity(hologram.spectrum), "Spectrum of captured hologram"),
            (hologram.reconstructed_intensity, "Reconstructed intensity"),
            (
                optional(compensated, holo_class.intensity_compensated_db),
                "Intensity after compensation",
            ),
            (
                holo_class.intensity(holo_class.field_propagated),
                "Intensity after propagation",
            ),
            (holo_class.intensity_reconstructed, "Final intensity image"),
            (
                optional(background is not None, getattr(background, "data", None)),
                "Background hologram",
            ),
            (
                hologram.intensity(hologram.spectrum_masked),
                "Shifted and masked spectrum",
            ),
            (hologram.reconstructed_phase, "Reconstructed phase"),
            (
                optional(compensated, holo_class.phase_compensated),
                "Phase after compensation",
            ),
            (
                holo_class.phase(holo_class.field_propagated),
                "Phase after propagation",
            ),
            (holo_class.phase_map, "Final unwrapped phase"),
        ]

        fig, axs = plt.subplots(2, 6, figsize=(16, 9))
        fig.suptitle("Hologram reconstruction", fontsize=16)

        if save_single:
            save_path = self._require_save_path()

        for idx, (data, title) in enumerate(plot_configs):
            ax = axs[idx // 6, idx % 6]
            if data is None:
                ax.axis("off")
                continue
            data = to_numpy(data)
            ax.imshow(data, cmap=cmap)
            ax.set_title(title.replace(" ", "\n"))

            if save_single:
                base_name = title.lower().replace(" ", "_")
                suffix = f"_{save_title}" if save_title else ""
                file_name = f"{base_name}{suffix}.png"
                self.save_individual_plot(
                    data,
                    os.path.join(save_path, file_name),
                    cmap=cmap,
                    title=title,
                    dpi=600,
                    save=True,
                )

        name = "Complete_hologram_reconstruction" + (
            f"_{save_title}" if save_title else ""
        )
        self._finish_figure(fig, name, show_plot)
