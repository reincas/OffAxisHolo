import cv2
from scidatacontainer import Container

from OffAxisHolo.offaxisholo.reconstruction import Hologram, ReferenceHologram, HologramProcessor
from SciDataContainer_Handler import StructureContainer

"""
Diese Datei ersetzt den Docker unter Scidatacontainerhandler.
Diese Datei kümmert sich um die Verknüpfung von Daten und der rekonstruktion des Holograms.
"""


class DockerBase:
    def __init__(self):
        pass

    def visual_reconstruction(self, reconstructor, save_img=True, save_path_plot=None, cmap="viridis"):
        """
        Visualization of the reconstruction process
        """
        # All possible fields
        self.initial_field = None  # initial field after
        self.field_propagated = None  # field after propagation and compensation
        self.phase_compensated = None  # phase after compensation (compensation is done before propagation)
        self.intensity_compensated = None  # intensity after compensation (compensation is done before propagation)
        self.field_compensated = None  # combined field of the compensated intensity and phase
        self.field_filtered = None  # field after filtering (not yet implemented)
        self.intensity_reconstructed = None  # intensity distribution of reconstructed field (same as compensated if no filtering is done
        self.field_reconstructed = None  # final reconstructed electromagnetic field (compensated intensity + phase)
        self.phase_map = None  # unwrapped phase map of the compensated, propagated hologram
        self.height_profile = None  # final height profile of the Hologram (reconstructed field)

        if save_img and save_path_plot is None:
            raise Exception("A path for saving the images is needed! \nUse the method set_save_path for this purpose.")

        reconstructor.set_save_path(path=save_path_plot)
        reconstructor.plotImage(reconstructor.hologram.data, "Recorded Hologram", save=save_img, cmap=cmap)
        # Plotting of spectrum - normal, shifted, masked
        reconstructor.plotImage(reconstructor.hologram.intensity(reconstructor.hologram.spectrum_not_shifted),
                                "Angular spectrum not shifted",
                                save=save_img, cmap=cmap)
        reconstructor.plotImage(reconstructor.hologram.intensity(reconstructor.hologram.spectrum_shifted),
                                "Angular spectrum shifted",
                                save=save_img, cmap=cmap)
        reconstructor.plotImage(reconstructor.hologram.intensity(reconstructor.hologram.spectrum_masked),
                                "Angular spectrum shifted and masked",
                                save=save_img, cmap=cmap)

        # plotting of reconstruction process
        # Initial field after fft - shift 1st order - ifft
        reconstructor.plotImage(img=reconstructor.phase(reconstructor.initial_field),
                                title="Phase after background compensation",
                                save=save_img,
                                cmap=cmap)
        reconstructor.plotImage(img=reconstructor.intensity(reconstructor.initial_field),
                                title="Intensity after background compensation",
                                save=save_img,
                                cmap=cmap)
        # Propagated field (after compensation)
        if reconstructor.field_propagated is not None:
            reconstructor.plotImage(img=reconstructor.phase(reconstructor.field_propagated),
                                    title="Phase after propagation",
                                    save=save_img,
                                    cmap=cmap)
            reconstructor.plotImage(img=reconstructor.intensity(reconstructor.field_propagated),
                                    title="Intensity after propagation",
                                    save=save_img,
                                    cmap=cmap)
        # Aberration compensation with background image
        if reconstructor.field_compensated is not None:
            reconstructor.plotImage(img=reconstructor.phase_compensated,
                                    title="Phase after background compensation",
                                    save=save_img,
                                    cmap=cmap)
            reconstructor.plotImage(img=reconstructor.intensity_compensated,
                                    title="Intensity after background compensation",
                                    save=save_img,
                                    cmap=cmap)
        # Filtered field
        if reconstructor.field_filtered is not None:
            reconstructor.plotImage(img=reconstructor.phase(reconstructor.field_filtered),
                                    title="Phase after filtering",
                                    save=save_img,
                                    cmap=cmap)
            reconstructor.plotImage(img=reconstructor.intensity(reconstructor.field_filtered),
                                    title="Intensity after filtering",
                                    save=save_img,
                                    cmap=cmap)
        # Fully reconstructed image
        if reconstructor.field_reconstructed is not None:
            reconstructor.plotImage(img=reconstructor.phase(reconstructor.field_reconstructed),
                                    title="Phase after complete reconstruction",
                                    save=save_img,
                                    cmap=cmap)
            reconstructor.plotImage(img=reconstructor.intensity(reconstructor.field_reconstructed),
                                    title="Intensity after complete reconstruction",
                                    save=save_img,
                                    cmap=cmap)
        # Unwrapped phase map
        if reconstructor.phase_map is not None:
            reconstructor.plotImage(img=reconstructor.phase(reconstructor.phase_map),
                                    title="Unwrapped Phasemap",
                                    save=save_img,
                                    cmap=cmap)
        # Height profile of phase map
        if reconstructor.height_profile is not None:
            reconstructor.plot_height(height_profile=reconstructor.height_profile,
                                      title=f"Height profile with refractive index of {reconstructor.n_resin}",
                                      save=save_img)
            # cmap=cmap will not be changed because of better visibility of coolwarm.
        # Images of the reconstructed background image
        if reconstructor.background.reconstructed_field is not None:
            reconstructor.plotImage(img=reconstructor.phase(reconstructor.background.reconstructed_field),
                                    title="Phase map of reconstructed background hologram",
                                    save=save_img,
                                    cmap=cmap)
            reconstructor.plotImage(img=reconstructor.intensity(reconstructor.background.reconstructed_field),
                                    title="Intensity image of reconstructed background hologram",
                                    save=save_img,
                                    cmap=cmap)


class DockerSciDataContainer(DockerBase):
    def __init__(self):
        super().__init__()

    def plot_reconstruction(self, container_path,
                            save_img=True, save_path_plot=None,
                            save_data=False, save_path_data=None, data_name=None,
                            compensate=True,
                            propagate=True, prop_dist=None,
                            cmap="viridis"):
        """
        All-in-one method for investigating a taken hologram.
        If save_img = True , then the plotted images will be safed to self.img_save_path
                                - set the path with set_save_path(path)

        Plotting of all steps:
        recording of hologram
        holo = hologram.data
        field to spectrum
        spectrum_normal = hologram.spectrum_not_shifted
        shift spectrum
        spectrum_shifted = hologram.spectrum_shifted
        mask spectrum (to get the -1 diffraction order)
        spectrum_shifted_masked = hologram.spectrum_masked
        inverse fouriertransform
        reconstructed_int = hologram.reconstructed_intensity
        reconstructed_phase = hologram.reconstructed_phase
        propagation
        propagated_int = hologram.propagated_intensity
        propagated_phase = hologram.propagated_phase
        phase unwrapping
        unwrapped_phase = hologram.phase_unwrapped
        aberration compensation
        compensated_phase = self.phase_compensated
        height_profile = self.phase_to_height(compensated_phase)
        """
        # ToDo Title der Auswertungen ändern.
        # ToDo save_img und savepath überarbeiten

        data_container = StructureContainer(file=container_path)

        tmp_dict = {}
        params = data_container.dhm_params
        background_hologram = data_container.background_hologram
        finished_structure = data_container.complete_hologram

        structure = Hologram(data=finished_structure, dhm_parameter=params)
        pos = structure.first_diffraction_order_pos

        if compensate:
            background = ReferenceHologram(data=background_hologram,
                                           first_diffraction_order_pos=pos,
                                           dhm_parameter=params)
            # Reconstruction of complete structure_dhm with aberration compensation
            reconstructor = HologramProcessor(hologram=structure, reference=background)
        else:
            # Reconstruction of complete structure_dhm without background aberration compensation
            reconstructor = HologramProcessor(hologram=structure)

        # reconstructor.run(propagate=propagate, prop_dist=prop_dist, compensate=compensate)
        reconstructor.run(propagate=propagate, prop_dist=prop_dist, compensate=compensate)

        # plotting of reconstruction process
        self.visual_reconstruction(reconstructor=reconstructor,
                                   save_img=save_img, save_path_plot=save_path_plot,
                                   cmap=cmap)
        if save_data:
            assert save_path_data is not None, f"No directory {save_path_data} found to save the data."
            reconstructor.save(path=save_path_plot, name=data_name)

    def reconstruct(self, structure_path, mode: str = "raw",
                    save_data=False, save_path=None, save_name=None,
                    propagate=True, prop_dist=None,
                    return_reconstructor=False):
        if structure_path.endswith(".zdc"):
            data_container = StructureContainer(file=structure_path)
        else:
            path = structure_path + ".zdc"
            data_container = Container(file=path)

        if mode.lower() == "raw":
            compensate = False
        elif mode.lower() == "compensate":
            compensate = True

        params = data_container.dhm_params
        background_hologram = data_container.background_hologram
        finished_structure = data_container.complete_hologram

        structure = Hologram(data=finished_structure, dhm_parameter=params)
        pos = structure.first_diffraction_order_pos
        background = ReferenceHologram(data=background_hologram,
                                       first_diffraction_order_pos=pos,
                                       dhm_parameter=params)

        if mode == "raw":
            structure.run()
            field = structure.reconstructed_field
        elif mode == "compensate":
            reconstructor = HologramProcessor(hologram=structure, reference=background)
            reconstructor.run(propagate=propagate, prop_dist=prop_dist, compensate=compensate, mode="print")
            # ToDo : zweimal mode als parameter welches etwas unterschiedliches bedeutet
            field = reconstructor.field_reconstructed

        if save_data:
            if save_path is None:
                raise IOError("No save path provided")
            structure.save(path=save_path, data=field, name=save_name)

        if return_reconstructor:
            if compensate:
                return reconstructor
            else:
                return structure
        else:
            return field


class DockerImageFile(DockerBase):
    def __init__(self):
        super().__init__()
        self.dhm_params_63x = {"Objective": "Zeiss 63x",
                               "pixel pitch": 0.0869e-6,
                               "wavelength": 0.6749e-6,
                               "refractive index": 1.5,
                               "propagation distance": 0,
                               "DC radius": 304}
        self.dhm_params_20x = {"Objective": "Zeiss 20x",
                               "pixel pitch": 0.27596e-6,
                               "wavelength": 0.6749e-6,
                               "refractive index": 1.5,
                               "propagation distance": 0,
                               "DC radius": 304}

    def plot_reconstruction(self):
        # ToDo implement here: evaluate from Reconstructor
        pass

    # ToDo: make a reconstructor maker
    def reconstruct(self, img_path, background_img_path=None, objective="Zeiss 63x", img_type="tif",
                    save_data=False, save_path=None, save_name=None,
                    compensate=True,
                    propagate=True, prop_dist=None,
                    return_reconstructor=False):
        """
        :param img_path:    path of the image
        :param background_img_path:     path of the background image if available
        :param objective:   objective of the DHM - determines the parameter of reconstruction - ToDo: change to dict or dummy dhm in the future
        :param img_type:    type of image - implemented are .png and .tif
        :param save_data:   boolean to save the reconstructed field to save_path
        :param save_path:   path for the el. field of the reconstructed image
        :param save_name:   name for the numpy array for the reconstructed image
        """
        # ToDo: implement png
        if objective == "Zeiss 63x":
            dhm_params = self.dhm_params_63x
        elif objective == "Zeiss 20x":
            dhm_params = self.dhm_params_20x
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

            background = ReferenceHologram(data=back_img,
                                           first_diffraction_order_pos=structure.first_diffraction_order_pos,
                                           dhm_parameter=dhm_params)
            reconstruction = HologramProcessor(hologram=structure, reference=background)
            field = reconstruction.run(propagate=propagate, prop_dist=prop_dist, compensate=compensate,
                                       mode="developed")
        else:
            structure.run()
            field = structure.reconstructed_field

        if save_data:
            if save_path is None:
                raise IOError("No save path provided")
            structure.save(path=save_path, data=field, name=save_name)
        if return_reconstructor:
            if background_img_path is not None:
                return reconstruction
            else:
                return structure
        else:
            return field


if __name__ == "__main__":
    test_path = r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\rawdata\DHM_Print\structures\DOE1_ABZ_Zeiss 63x.zdc"
    test_eval = r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\visualization\Test_Docker"

    docker = DockerSciDataContainer()
    docker.plot_reconstruction(container_path=test_path, save_path_plot=test_eval)
