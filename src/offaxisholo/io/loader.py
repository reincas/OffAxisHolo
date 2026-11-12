import os
from typing import Optional

import cv2
import numpy as np
from scidatacontainer import Container

from .structure import StructureContainer


class DataLoader:
    def __init__(
        self, file_path, file_type: Optional[str] = None, logger=None, **kwargs
    ):
        """
        file_path: absolute path to file
        file_type: extension of file
        logger: logging object
        ----
        Keyword arguments:
            structure_container:
                    Specific SciDataContainer class for 2PP printing.
            loading_background:
                    Additionally loads the background image. Only available with
                    file type "zdc"
            loading_layer_data:
                    Only available with file type "zdc". Loads data for each layer.
            convert_to_grayscale:
                    Only for image types (png, tif). Converts a colour image to a
                    single channel image. The bit depth is preserved (e.g. 16 bit).
            allow_8bit:
                    Only for image types (png, tif). If True, images with a higher
                    bit depth are linearly scaled to 8 bit (uint8) to save memory.
                    This reduces the dynamic range. Default: False.
        ----
        get_data returns the loaded data
        """
        self.path = file_path
        self.logger = logger

        if file_type is None:
            file_name, file_extension = os.path.splitext(file_path)
            file_type = file_extension[1:]
        else:
            # only validate file extension if necessary.
            self._validate_file_type(file_type)

        self.file_type = file_type

        # Set default values for optional parameters
        self.loading_background = kwargs.get("loading_background", False)
        self.loading_layer_data = kwargs.get("loading_layer_data", False)
        self.structure_container = kwargs.get("structure_container", False)
        self.convert_to_grayscale = kwargs.get("convert_to_grayscale", True)
        self.allow_8bit = kwargs.get("allow_8bit", False)

        self._validate_kwargs()

        # Set any additional kwargs
        for key, val in kwargs.items():
            if not hasattr(self, key):  # Only set if not already set
                setattr(self, key, val)

        if file_type == "zdc":
            if self.structure_container:  # specific type of container
                data_container = StructureContainer(file=self.path)
                self.dhm_params = data_container.dhm_params
                if self.loading_background:
                    if self.logger:
                        self.logger.info("Loading background image from Container...")
                    self.background_data = data_container.background_hologram
                if self.loading_layer_data:
                    # ToDo implement it in StructureContainer as a property
                    if self.logger:
                        self.logger.info("Loading layer data from Container...")
                    data = [
                        data_container[f"meas/dhm/raw/layer_{i}.png"].data
                        for i in range(data_container.number_of_layer)
                    ]
                    # todo check if final layer == complete structure otherwise append
                    # the final structure!
                    # data = data_container.data_layered  # to be implemented
                else:
                    # loading of completed print
                    if self.logger:
                        self.logger.info("Loading hologram data from Container...")
                    data = data_container.complete_hologram
            else:
                if (self.loading_background or self.loading_layer_data) and self.logger:
                    self.logger.warning(
                        "Background images and layered data can only be loaded "
                        "with specific structure container. Set the kwarg "
                        "'structure_container' = True to access the features."
                    )
                    # warnings.warn(
                    #     f"Background images and layered data can only be loaded with
                    #     f"specific structure container."
                    #     f"Set the kwarg 'structure_container' = True to access the"
                    #     f" features.")

                data = self.load_scidatacontainer()

        elif file_type == "png" or file_type == "tif" or file_type == "tiff":
            if self.logger:
                self.logger.info("Loading image ...")
            data = self.load_image()
        else:
            raise ValueError(f"Unsupported file type {file_type}")
        if self.logger:
            self.logger.info("Data loaded.")
        self.data_loaded = data

    def _validate_file_type(self, file_type):
        file_name, file_extension = os.path.splitext(self.path)
        if file_type != file_extension[1:]:
            raise ValueError(
                f"File type {file_type} does not match ending of file in path: "
                f"{self.path}."
            )

    def _validate_kwargs(self):
        assert isinstance(
            self.structure_container, bool
        ), "Keyword 'structure container' has to be a boolean"
        assert isinstance(
            self.loading_background, bool
        ), "Keyword 'loading_background' has to be a boolean"
        assert isinstance(
            self.loading_layer_data, bool
        ), "Keyword 'loading_layer_data' has to be a boolean"
        assert isinstance(
            self.convert_to_grayscale, bool
        ), "Keyword 'convert_to_grayscale' has to be a boolean"
        assert isinstance(
            self.allow_8bit, bool
        ), "Keyword 'allow_8bit' has to be a boolean"

    def load_scidatacontainer(self):
        dc = Container(file=self.path)
        data = dc._items["meas/image.png"]
        return data.data

    def load_scidatacontainer_layers(self):  # todo: to be removed
        print("Layer data loader not yet implemented.")
        layer_data = []
        # get total number of layer
        num_layer = 0  # ToDo get it from zdc json file with layer number
        for _i in range(num_layer):
            # layer_data = [data for data_name in scidatacontainer_zdc]
            # todo implement this
            pass
        return layer_data

    def load_image(self):
        """
        Load a png or tif image.

        The image is read with its original bit depth (cv2.IMREAD_UNCHANGED), so
        16 bit camera images keep their full dynamic range. The integer data type
        is kept (no conversion to float) to keep the memory consumption low.

        convert_to_grayscale: colour images (3 or 4 channels) are converted to a
            single channel. Single channel images are left unchanged.
        allow_8bit: images with more than 8 bit are scaled linearly to uint8
            (value * 255 / max(value)). The zero point is kept, so the relative
            intensities remain correct, but the quantisation becomes coarser.
        """
        data = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if data is None:  # cv2.imread does not raise on missing/unreadable files
            raise FileNotFoundError(f"Image {self.path} could not be read.")

        if self.convert_to_grayscale and data.ndim == 3:
            if self.logger:
                self.logger.info("Converting image to grayscale...")
            if data.shape[2] == 4:
                data = cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
            elif data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            else:
                data = data[:, :, 0]

        if self.allow_8bit and data.dtype != np.uint8:
            if self.logger:
                self.logger.info(f"Scaling {data.dtype} image to 8 bit...")
            maximum = float(np.max(data))
            scale = 255.0 / maximum if maximum > 0 else 0.0
            data = np.round(data.astype(np.float32) * scale).astype(np.uint8)

        return data

    def get_data(self):
        try:
            if self.loading_background and self.structure_container:
                return [self.data_loaded, self.background_data]
            else:
                return self.data_loaded
        except Exception as err:
            raise ValueError(f"Error while loading data from {self.path}") from err

    def get_dhm_params(self):
        if not self.structure_container:
            raise AttributeError(
                "DHM Parameter only available with specific structure container."
            )
        try:
            return self.dhm_params
        except Exception as e:
            raise ValueError(
                f"Error {e} while loading DHM parameters from {{self.path}}"
            ) from e
