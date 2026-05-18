from typing import Literal
import numpy as np
from scidatacontainer import Container
import os
import json

if __name__ == "__main__":
    from OffAxisHolo.reconstruction import HologramProcessor, Hologram, ReferenceHologram
    from OffAxisHolo import get_logger, DataLoader
else:
    from .reconstruction import HologramProcessor, Hologram, ReferenceHologram
    from .loader import DataLoader
    from .__init__ import get_logger
from tkinter import messagebox, filedialog

class DockerBase:
    holo: Hologram
    background: ReferenceHologram
    processor: HologramProcessor

    def __init__(self, data_path, save_path, *,
                 data_type: Literal["zdc", "png", "tif"] = None,
                 dhm_dictionary=None, dhm_preset="Zeiss 63x",
                 material_dictionary=None, material_preset="SZ2080",
                 compensation_method: Literal["Background", "ZernikePolynomial", "None"] = "Background",
                 background_data_available: bool = None,
                 background_data_preset: Literal[""] = None,
                 background_data_path=None,
                 using_layer_data=False,
                 logger=None
                 ):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
        else:
            self.save_path = None
        self.data_path = data_path

        self.structure_zdc_container = False
        self.using_layer_data = using_layer_data  # no layer data available if not a structureContainer
        self.background_data_preset = background_data_preset  # initialization

        if background_data_available is None and background_data_path is None and compensation_method=="None":
            self.background_data_available = False
            self.background_data_path = None

        if self.save_path is not None:
            if logger is None:
                self.logger = get_logger(logfile=f"{save_path}/console.log")
            else:
                self.logger = logger

        if data_type is None:
            file, file_extension = os.path.splitext(self.data_path)
            self.data_type = file_extension[1:]
        else:
            self.data_type = data_type

        if self.data_type == "zdc":
            self.using_layer_data = using_layer_data
            self.check_structure_zdc()

        if compensation_method == "Background" and not self.structure_zdc_container:
            # Compensation using background
            #   1. Data path is given
            #   2. No Data path, but a preset was selected
            #   3. No Data path is given but path is selected using filedialog
            #   4. No Data path, no preset and no selected data -> Question if compensation is desired
            #                                                   -> no compensation

            self.compensation = True  # compensation mode is background so a compensation should be done
            if background_data_path is not None:
                self.background_data_path = background_data_path
                self.background_data_available = True
                # return #todo return is wrong

            if background_data_available is None and background_data_path is None:
                if background_data_preset is not None:
                    # self.background_data_preset = data in form of np.ndarray
                    # todo: Implement default background images for DHM
                    # todo: implement ask dialog for selecting preset
                    print("Not yet implemented")
                    return

                response = messagebox.askyesno(title="Background data", message="Is Background data available?")
                if response:
                    # background data is available - get background path
                    no_compensation = False
                    while not no_compensation:  # loop for falsely pressed cancel
                        path = filedialog.askopenfile(mode='r', filetypes=[('Image Files', ['*.tif', '*.png']),
                                                                           ('DataContainer', '*.zdc')])
                        if path is not None:
                            self.background_data_available = True
                            self.background_data_path = path.name
                        else:
                            no_compensation = messagebox.askyesno(title="Aberration compensation",
                                                                  message="Reconstruct Hologram without aberration \
                                                                    compensation?")
                            if no_compensation:
                                self.logger.info("Use 'compensation_method'=False if no compensation is required.")
                                self.compensation = False
                                compensation_method = "None"
                                return
                else:
                    # ask for other compensation method
                    zernike_compensation = messagebox.askyesno(title="Compensation Method",
                                                               message="Do you want to use Zernike Polynomials as \
                                                                        compensation method?")
                    if zernike_compensation:
                        compensation_method = "ZernikePolynomial"

        elif compensation_method == "Background" and self.structure_zdc_container:
            self.background_data_available = True
            self.compensation = True

        elif compensation_method == "ZernikePolynomial":
            self.compensation = True
            raise NotImplementedError("Zernike Polynomial is not yet implemented.")

        elif compensation_method == "None":
            self.compensation = False

        else:
            raise NotImplementedError(f"No compensation method {compensation_method} available.")

        self.compensation_method = compensation_method

        if dhm_dictionary is None:
            self.dhm_dictionary = {}
            if dhm_preset == "Zeiss 63x":
                with open(os.path.join(os.path.dirname(__file__), 'DHM_preset/Zeiss_63x.json'), 'r') as file:
                    self.dhm_dictionary = json.load(file)
            elif dhm_preset == "Zeiss 20x":
                with open(os.path.join(os.path.dirname(__file__), 'DHM_preset/Zeiss_20x.json'), 'r') as file:
                    self.dhm_dictionary = json.load(file)
            else:
                self.logger.warning(f"No preset {dhm_preset}")
                raise NotImplementedError(f"No preset {dhm_preset} implemented.")
            self.logger.info(f"DHM Preset {dhm_preset} loaded.")
        else:
            self.dhm_dictionary = dhm_dictionary

        if material_dictionary is None:
            self.material_dictionary = {}
            if material_preset == "SZ2080":
                with open(os.path.join(os.path.dirname(__file__), 'Background_preset/SZ2080.json'), 'r') as file:
                    self.material_dictionary = json.load(file)
            else:
                self.logger.warning(f"No preset {material_preset}")
                raise NotImplementedError(f"No preset {material_preset} implemented.")
            self.logger.info(f"Material Preset {material_preset} loaded.")
        else:
            self.material_dictionary = material_dictionary

    @property
    def pixel_pitch(self):
        try:
            return self.processor.pixel_pitch
        except AttributeError:
            raise NotImplementedError("Pixel pitch not accessible.")

    def set_visualization_options(self, visualization_options, *kwargs):
        # visualization options should be a dictionary or i will use the kwargs - not sure yet.
        # This is intended to ensure that visualizations of the respective process steps or reconstruction steps are
        # automatically created and saved.
        # todo create an overview for the different options!
        pass

    def do_visualizations(self, processor):
        # todo implement it!
        # should be the saving of the plotting in dependence on what should be done!
        pass

    def set_saving_options(self, saving_options, *kwargs):
        # same as visualization
        # has to be done
        # should serve as an access point for which data should be saved as a np.array
        # todo: create this!
        pass

    def do_savings(self, processor):
        # todo implement it!
        # should be the saving of the numpy arrays in dependence on what should be saved!
        pass

    def plot_4_publications(self, path=None, cmap='gray'):
        if path is None:
            if self.save_path is None:
                raise FileNotFoundError("Save path has to be given!")
            save_path = self.save_path,
        else:
            save_path = path

        self.processor.plotter.set_save_path(path=save_path)
        self.processor.plot_reconstruction(mode="short", save_single=True, cmap=cmap, compensation=self.compensation)

    def plot_complete_reconstruction(self, path=None, cmap="gray"):
        if path is None:
            if self.save_path is None:
                raise FileNotFoundError("Save path has to be given!")
            save_path = self.save_path
        else:
            save_path = path

        self.processor.plotter.set_save_path(path=save_path)
        self.processor.plot_reconstruction(mode="full", save_single=True, cmap=cmap, compensation=self.compensation)

    def save_data(self, path=None, mode="short"):
        if path is None:
            if self.save_path is None:
                raise FileNotFoundError("Save path has to be given!")
            save_path = self.save_path,
        else:
            save_path = path

        if mode == "short":
            # saving only minimum necessary information
            self.save_necessary_data(save_path=save_path)
        elif mode == "full" or mode == "all":
            self.save_all_data(path=save_path)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented. Data not saved.")

    def save_necessary_data(self, save_path):
        # save holo data
        np.save(os.path.join(save_path, "hologram_original_data"), self.holo.data)  # original data
        if self.background_data_available:
            np.save(os.path.join(save_path, "hologram_background"), self.background.data)  # background data
        np.save(os.path.join(save_path, "processor_intensity"), self.processor.intensity_reconstructed)
        np.save(os.path.join(save_path, "processor_phase_unwrapped"), self.processor.phase_map)

        information_saved_fields = {
            "hologram_original_data": "Captured hologram of the printed object.",
            "hologram_background": "Captured background hologram. Used for aberration compensation.",
            "processor_intensity": "Final intensity. If filtering was done, this data is after filtering.",
            "processor_phase_unwrapped": "Final unwrapped phase",
        }
        with open(os.path.join(save_path, 'reconstruction_dictionary.json'), 'w', encoding='utf8') as json_file:
            json.dump(
                self.processor.reconstruction_dict | {"Information_saved_arrays": information_saved_fields},
                json_file,
                indent=4
            )

    def save_all_data(self, path=None):
        if path is None:
            if self.save_path is None:
                raise FileNotFoundError("Save path has to be given!")
            save_path = self.save_path,
        else:
            save_path = path

        # save holo data
        np.save(os.path.join(save_path, "hologram_original_data"), self.holo.data)  # original data
        np.save(os.path.join(save_path, "hologram_Reconstructed_field"),
                self.holo.reconstructed_field)  # reconstructed field data
        if self.background_data_available:
            np.save(os.path.join(save_path, "hologram_background"), self.background.data)  # background data

        # save processed data
        np.save(os.path.join(save_path, "processor_field_propagated"), self.processor.field_propagated)
        np.save(os.path.join(save_path, "processor_field_compensated"), self.processor.field_compensated)
        np.save(os.path.join(save_path, "processor_reconstructed_field"), self.processor.field_reconstructed)
        np.save(os.path.join(save_path, "processor_intensity"), self.processor.intensity_reconstructed)
        np.save(os.path.join(save_path, "processor_phase_wrapped"), self.processor.phase_compensated)
        np.save(os.path.join(save_path, "processor_phase_unwrapped"), self.processor.phase_map)

        information_saved_fields = {
            "hologram_original_data": "Captured hologram of the printed object.",
            "hologram_Reconstructed_field": "Reconstructed field of the object. Only spatial filtering.",
            "hologram_background": "Captured background hologram. Used for aberration compensation.",
            "processor_field_propagated": "Field after propagation. Before aberration compensation.",
            "processor_field_compensated": "Recombined field of the compensated intensity and phase",
            "processor_reconstructed_field": "final reconstructed electromagnetic field",
            "processor_intensity": "Final intensity. If filtering was done, this data is after filtering.",
            "processor_phase_wrapped": "Final wrapped phase",
            "processor_phase_unwrapped": "Final unwrapped phase",
        }
        with open(os.path.join(save_path, 'reconstruction_dictionary.json'), 'w', encoding='utf8') as json_file:
            json.dump(
                self.processor.reconstruction_dict | {"Information_saved_arrays": information_saved_fields},
                json_file,
                indent=4
            )

    def run_reconstruction(self, **kwargs):
        # propagation_distance: kwargs have priority, then default
        propagation_distance = (
            kwargs.get("propagation_distance")
            if kwargs.get("propagation_distance") is not None
            else kwargs.get("prop_dist")
            if kwargs.get("prop_dist") is not None
            else kwargs.get("prop_distance")
            if kwargs.get("prop_distance") is not None
            else self.dhm_dictionary['propagationDistance']
        )

        # propagation_method: kwargs have priority, then default
        propagation_method = (
            kwargs.get("propagation_method")
            if kwargs.get("propagation_method") is not None
            else kwargs.get("prop_method")
            if kwargs.get("prop_method") is not None
            else "angularSpectrum"  # default method
        )

        # phase_unwrapping_method: kwargs have priority, then default
        phase_unwrapping_method = (
            kwargs.get("phase_unwrapping_method")
            if kwargs.get("phase_unwrapping_method") is not None
            else kwargs.get("unwrap_method")
            if kwargs.get("unwrap_method") is not None
            else "Fast 2D"
        )

        # mode_structure: kwargs have priority, then default
        mode_structure = (
            kwargs.get("mode_structure")
            if kwargs.get("mode_structure") is not None
            else "print"
        )

        # compensation: kwargs have priority, then default
        compensation = (
            kwargs.get("compensation")
            if kwargs.get("compensation") is not None
            else kwargs.get("do_compensation")
            if kwargs.get("do_compensation") is not None
            else self.compensation
        )

        # compensation_method: kwargs have priority, then default (but None if ‘None’)
        compensation_method = (
            kwargs.get("compensation_method")
            if kwargs.get("compensation_method") is not None
            else kwargs.get("compensationMethod")
            if kwargs.get("compensationMethod") is not None
            else (None if self.compensation_method == "None" else self.compensation_method)
        )

        # refractive_index: kwargs have priority, default is None
        refractive_index = (
            kwargs.get("refractive_index")
            if kwargs.get("refractive_index") is not None
            else kwargs.get("refractiveIndex")
            if kwargs.get("refractiveIndex") is not None
            else kwargs.get("refractiveindex")
            if kwargs.get("refractiveindex") is not None
            else kwargs.get("n_resin")
        )

        # filter_list: kwargs have priority, default is None
        filter_list = (
            kwargs.get("list_filter")
            if kwargs.get("list_filter") is not None
            else kwargs.get("list_filters")
            # if kwargs.get("list_filters") is not None
            # else kwargs.get("filter_list")
        )

        # filtering: based on filter_list or kwargs
        filtering = (
            kwargs.get("filtering")
            if kwargs.get("filtering") is not None
            else kwargs.get("do_filtering")
            if kwargs.get("do_filtering") is not None
            else (filter_list is not None and filter_list != [])
        )

        # propagate: based on propagation_distance or kwargs
        propagate = (
            kwargs.get("propagation")
            if kwargs.get("propagation") is not None
            else (propagation_distance is not None and propagation_distance != 0.0)
        )

        # Acquire Data
        data, background_data = self.get_data()

        if self.using_layer_data:
            raise NotImplementedError("Reconstruction of layered data is not yet implemented.")


        self.holo = Hologram(data=data,
                             dhm_parameter=self.dhm_dictionary,
                             logger=self.logger)

        if compensation == False or compensation_method is None:
            # do reconstruction without background

            self.processor = HologramProcessor(hologram=self.holo, reference=None,
                                               dhm_parameter=self.dhm_dictionary,
                                               material_parameter=self.material_dictionary)

            self.processor.run(prop_dist=propagation_distance, propagation_method=propagation_method,
                               propagate=propagate,
                               compensation_mode=compensation_method, compensate=compensation,
                               refractive_index=refractive_index,
                               phase_unwrapping_method=phase_unwrapping_method,
                               mode=mode_structure,
                               filtering=filtering, filter_applied=filter_list
                               )
        else:
            self.background = ReferenceHologram(data=background_data,
                                                first_diffraction_order_pos=self.holo.first_diffraction_order_pos,
                                                dhm_parameter=self.dhm_dictionary,
                                                logger=self.logger)

            self.processor = HologramProcessor(hologram=self.holo, reference=self.background,
                                               dhm_parameter=self.dhm_dictionary,
                                               material_parameter=self.material_dictionary)

            self.processor.run(prop_dist=propagation_distance, propagation_method=propagation_method, propagate=propagate,
                               compensation_mode=compensation_method, compensate=compensation,
                               refractive_index=refractive_index,
                               phase_unwrapping_method=phase_unwrapping_method,
                               mode=mode_structure,
                               filtering=filtering, filter_applied=filter_list
                               )

        # self.do_savings(self.processor)
        # self.do_visualizations(self.processor)

    def get_data(self):
        # Preset used for background data + loaded data
        if self.background_data_preset is not None:
            loader = DataLoader(file_path=self.data_path,
                                file_type=self.data_type,
                                logger=self.logger,
                                loading_background=False,
                                structure_container=self.structure_zdc_container,
                                loading_layer_data=self.using_layer_data)
            return loader.get_data(), self.background_data_preset

        # Background image in Container + data of hologram or layered data
        if self.structure_zdc_container:
            loader = DataLoader(file_path=self.data_path,
                                file_type=self.data_type,
                                logger=self.logger,
                                loading_background=self.background_data_available,
                                structure_container=self.structure_zdc_container,
                                loading_layer_data=self.using_layer_data)
            return loader.get_data()  # equal to [data, background_data]

        # All other cases are either a normal SciDataContainer or image data
        if self.compensation_method == "Background" and self.background_data_available:
            background = DataLoader(file_path=self.background_data_path,
                                    logger=self.logger,
                                    loading_background=False)
            loader = DataLoader(file_path=self.data_path,
                                file_type=self.data_type,
                                logger=self.logger,
                                loading_background=False)
            return loader.get_data(), background.get_data()

        elif self.compensation_method == "None":
            background = []
            loader = DataLoader(file_path=self.data_path,
                                file_type=self.data_type,
                                logger=self.logger,
                                loading_background=False)
            return [loader.get_data(), background]

    def check_structure_zdc(self):
        dc = Container(file=self.data_path)
        if dc.content['containerType']['name'] == "StructureContainer":
            self.structure_zdc_container = True
            self.background_data_available = True


if __name__ == "__main__":
    test_path = r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\rawdata\DHM_Print\structures\DOE1_ABZ_Zeiss 63x.zdc"
    test_eval = r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\visualization\Test_Docker"

    test_zdc = Container(file=test_path)

    print("stop")
    # docker = DockerSciDataContainer()
    # docker.plot_reconstruction(container_path=test_path, save_path_plot=test_eval)
