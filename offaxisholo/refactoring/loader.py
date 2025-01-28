from typing import Optional
import os
from scidatacontainer import Container
from SciDataContainer_Handler import StructureContainer
import cv2


class DataLoader:
    def __init__(self, file_path, file_type: Optional = None, logger=None, **kwargs):
        """
        file_path: absolute path to file
        file_type: extension of file
        logger: logging object
        ----
        Keyword arguments:
            structure_container: Specific SciDataContainer class for 2PP printing.
            loading_background: Additionally loads the background image. Only available with file type "zdc"
            loading_layer_data: Only available with file type "zdc". Loads data for each layer.
            convert_to_grayscale: Only for image types(png, tif). Converts image to grayscale image.
        ----
        get_data returns the loaded data
        """
        self.path = file_path
        if logger is not None:
            self.logger = logger

        if file_type is None:
            file_name, file_extension = os.path.splitext(file_path)
            file_type = file_extension[1:]
        else:
            # only validate file extension if necessary.
            self._validate_file_type(file_type)

        self.file_type = file_type

        # Set default values for optional parameters
        self.loading_background = kwargs.get('loading_background', None)
        self.loading_layer_data = kwargs.get('loading_layer_data', None)
        self.structure_container = kwargs.get('structure_container', False)
        self.convert_to_grayscale = kwargs.get('convert_to_grayscale', None)

        self._validate_kwargs()

        # Set any additional kwargs
        for key, val in kwargs.items():
            if not hasattr(self, key):  # Only set if not already set
                setattr(self, key, val)

        if file_type == "zdc":
            if self.structure_container:  # specific type of container
                data_container = StructureContainer(file=self.path)
                self.dhm_params = StructureContainer.dhm_params
                if self.loading_background:
                    if self.logger:
                        self.logger.INFO("Loading background image from Container...")
                    self.background_data = data_container.background_hologram
                if self.loading_layer_data:
                    # ToDo implement it in StructureContainer as a property
                    if self.logger:
                        self.logger.INFO("Loading layer data from Container...")
                    data = [data_container[f"meas/dhm/raw/layer_{i}.png"].data for i in
                            range(data_container.number_of_layer)]
                    # todo check if final layer == complete structure otherwise append the final structure!
                    # data = data_container.data_layered  # to be implemented
                else:
                    # loading of completed print
                    if self.logger:
                        self.logger.INFO("Loading data from Container...")
                    data = data_container.complete_hologram
            else:
                if self.loading_background or self.loading_layer_data:
                    raise Warning(
                        f"Background images and layered data can only be loaded with specific structure container."
                        f"Set the kwarg 'structure_container' = True to access the features.")

                data = self.load_scidatacontainer()
        elif file_type == "png" or file_type == "tif" or file_type == "tiff":
            if self.logger:
                self.logger.INFO("Loading image ...")
            data = self.load_image()
        else:
            raise ValueError(f"Unsupported file type {file_type}")
        if self.logger:
            self.logger.INFO("Data loaded.")
        self.data_loaded = data

    def _validate_file_type(self, file_type):
        file_name, file_extension = os.path.splitext(self.path)
        try:
            match = file_type == file_extension
            if not match:
                raise AttributeError(f"File type {file_type} does not ending of file in path: {self.path}.")
        except Exception as e:
            print(f"An Error occurred while validating file type: {e}")

    def _validate_kwargs(self):
        assert isinstance(self.structure_container, bool), "Keyword 'structure container' has to be a boolean"
        assert isinstance(self.loading_background, bool), "Keyword 'loading_background' has to be a boolean"
        assert isinstance(self.loading_layer_data, bool), "Keyword 'loading_layer_data' has to be a boolean"
        assert isinstance(self.convert_to_grayscale, bool), "Keyword 'convert_to_grayscale' has to be a boolean"

    def load_scidatacontainer(self):
        dc = Container(file=self.path)
        data = dc['meas/image.png']
        return data.data

    def load_scidatacontainer_layers(self):  # todo: to be removed
        print("Layer data loader not yet implemented.")
        layer_data = []
        # get total number of layer
        num_layer = 0  # ToDo get it from zdc json file with layer number
        for i in range(num_layer):
            # layer_data = [data for data_name in scidatacontainer_zdc]  # todo implement this
            pass
        return layer_data

    def load_image(self):
        data = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if getattr(self, 'convert_to_grayscale', None):
            # Convert to grayscale
            if self.logger:
                self.logger.INFO("Converting image to grayscale...")
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return data

    def get_data(self):
        try:
            if self.loading_background and self.structure_container:
                return self.data_loaded, self.background_data
            else:
                return self.data_loaded
        except Exception as e:
            raise ValueError(f"Error while loading data from {self.path}")

    def get_dhm_params(self):
        if not self.structure_container:
            raise AttributeError(f"DHM Parameter only available with specific structure container.")
        try:
            return self.dhm_params
        except Exception as e:
            raise ValueError(f"Error {e} while loading DHM parameters from {self.path}")
