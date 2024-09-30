"""
GENERAL TODOS WHICH HAVE TO BE DONE IN THE FUTURE.

ToDo 1:     license thing on top of this file
ToDo 2:     update regularly this usage if new components are added to the aerotech file/ saving - system
ToDo 3:     Make a "StructureContainer" of the normal Container in the StructureCollector in save_container
ToDo 4:     saving and using the locations of the camera images -  until now it is only saved in the original zdc
ToDo 5:     
"""

import fnmatch
from typing import Dict, Any

import cv2 as cv
import numpy as np
import os
import glob
from scidatacontainer import Container


class StructureContainer(Container):
    containerType = "StructureInformation"
    containerVersion = 1.1

    def __pre_init__(self):

        """ Build items dictionary if in creation mode. """

        # Not in creation mode
        if (self.kwargs["file"] is not None) or \
                (self.kwargs["uuid"] is not None):
            return

        # Initialize items dictionary
        if self.kwargs["items"] is None:
            items = {}
        else:
            items = dict(self.kwargs["items"])

        # Container type
        content = items.get("content.json", {})
        content["containerType"] = {
            "name": self.containerType,
            "version": self.containerVersion}

        # Basic meta data
        meta = items.get("meta.json", {})
        meta["title"] = "Structure Information"
        meta["general description"] = ("Collected Information of a 2PP printed structure. Used for easy handling of "
                                       "all information of the structure. DHM as well as Camera images are saved.")

        # Update items dictionary
        items["content.json"] = content
        items["meta.json"] = meta

        # Camera image
        img = self.kwargs.pop("img")
        if not isinstance(img, np.ndarray) or len(img.shape) != 2:
            raise RuntimeError("Hologram image expected!")
        items["meas/image.png"] = img

        # Camera parameters
        params = self.kwargs.pop("params")
        if not isinstance(params, dict):
            raise RuntimeError("Parameter dictionary expected!")
        items["data/camera.json"] = params

        # Objective parameters
        objective = self.kwargs.pop("objective")
        if not isinstance(objective, dict):
            raise RuntimeError("Objective dictionary expected!")
        items["data/objective.json"] = objective

        # Optional location coordinates
        loc = self.kwargs.pop("loc", None)
        if loc:
            if not isinstance(loc, dict):
                raise RuntimeError("Location dictionary expected!")
            items["data/location.json"] = loc

        # Replace container items dictionary
        self.kwargs["items"] = items

    def __post_init__(self):

        """ Initialize this container. """

        # Type check of the container
        if (self.content["containerType"]["name"] != self.containerType) or \
                (self.content["containerType"]["version"] != self.containerVersion):
            raise RuntimeError(f"Containertype must be '{self.containerType}'!")

    @property
    def img(self):

        """ Shortcut to the camera image. """

        return self["meas/image.png"]

    @property
    def params(self):

        """ Shortcut to the parameter data dictionary. """

        return self["data/camera.json"]

    @property
    def location(self):

        """ Return xyz position of the image or None. """

        if "data/location.json" not in self:
            return None
        return self["data/location.json"]


class StructureCollector:
    def __init__(self, root_path, exp_description=None):
        # super(StructureContainer, self).__init__(root_path)

        # xxx\20240820_print\structures\corner_bl
        self.root_path = root_path
        self.dhm_path = os.path.join(self.root_path, 'dhm')
        self.camera_path = os.path.join(self.root_path, 'camera')
        self.program_path = os.path.join(self.root_path, 'programs')

        # initialize content lists
        self.program_content = []  # txt files in the program folder
        self.dhm_content = []  # zdc files in the dhm folder
        self.camera_content = []  # zdc files in the camera folder
        self.root_content = []  # all files located in the root folder
        self.root_zdc_content = []  # list of all zdc files in root directory
        self.root_dhm_content = []  # list of the dhm-containers in root directory
        self.root_camera_content = []  # list of the camera-containers in root directory

        # initialize dictionaries
        self.struc_dict = {}  # complete dictionary for the whole dictionary
        self.exp_description = None

        if exp_description is not None:
            if isinstance(exp_description, str):
                self.exp_description = exp_description
            elif isinstance(exp_description, dict):
                # If this is changed - it also has to be changed in self.exchange_general_information
                raise NotImplementedError("Not yet implemented if more than one information is given.")

    def collect_dict(self):
        tmp = self.get_general_info()
        self.struc_dict.update(tmp)
        tmp = self.get_dhm_dict()
        self.struc_dict.update(tmp)
        tmp = self.get_camera_dict()
        self.struc_dict.update(tmp)
        tmp = self.get_program_dict()
        self.struc_dict.update(tmp)

    def save_container(self, save_path=None):
        if save_path is None:
            save_path = self.root_path + ".zdc"
        dc = Container(items=self.struc_dict)
        dc.write(save_path)

    def get_container(self):
        if self.struc_dict == {}:
            self.collect_dict()

        self.save_container()
        return self.struc_dict

    def get_general_info(self):
        """should be used for merging all the different general data/ infos of the different container into one"""
        further_elems = []
        tmp_dict = {}
        if not self.root_content:
            self.get_root_content()
        tmp_dict = self.get_background_dict(tmp_dict)
        for element in self.root_content:
            if element not in self.root_zdc_content:
                tmp_info = {}
                further_elems.append(element)
                if element[-4:] == ".txt" and element[:7] == "program":
                    text = open(os.path.join(self.root_path, element), "r").read()
                    tmp_info = {f"info/complete_program.txt": text}
                elif element[-4:] == ".png" and element[:4] == "plot":
                    img = cv.imread(os.path.join(self.root_path, element))
                    tmp_info = {f"info/plotting_of_print.png": img}
                else:
                    pass
                    # ToDo(HR): Future implementation of different types if informations?
                if not tmp_info == {}:
                    tmp_dict.update(tmp_info)

        # getting information necessary for SciDataContainer and which is not changing throughout the structure
        tmp = self.get_same_content_dict()
        tmp_dict.update(tmp)
        return tmp_dict

    def get_same_content_dict(self):
        """
        ToDo: How to determine if anything did change throughout the sample?
        """
        tmp_dict = {}
        reference_file = self.root_zdc_content[0]
        dc = Container(file=os.path.join(self.root_path, reference_file))
        dc_content = dc.content
        dc_meta = dc.meta
        tmp_dict.update({"content.json": dc_content, "meta.json": dc_meta})
        tmp_dict = self.exchange_general_information(tmp_dict)
        # ToDo(HR): Future improvement to the experiment data gathering has to be implemented in here
        return tmp_dict

    def exchange_general_information(self, tmp_dict):
        tmp_dict["content.json"]["containerType"] = {"name": "StructureContainer", "version": 1.0}
        tmp_dict["meta.json"]["title"] = "Complete Structure Information"
        if self.exp_description is not None:
            tmp_dict["meta.json"]["description"] = self.exp_description
        else:
            tmp_dict["meta.json"]["description"] = "collection of all structure information."
        return tmp_dict

    def set_general_info(self, title, description, containerType=None):
        assert self.struc_dict != {}
        if containerType is not None:
            self.struc_dict["content.json"]["containerType"] = {"name": containerType, "version": 1.0}
        self.struc_dict["meta.json"]["title"] = title
        self.struc_dict["meta.json"]["description"] = description

    def get_background_dict(self, tmp_dict):
        """
        Deals with the background and afterwards images of the Camera and the DHM.
        Also saves the information, which is the same for each structure.
        ToDo(HR): How can we save all the locations, which are saved together with the image of the camera.
        """
        if not self.root_dhm_content:
            self.get_root_content()
        if not self.root_camera_content:
            self.get_root_content()

        # DHM Images
        file_after = os.path.join(self.root_path, self.root_dhm_content[0])  # after dhm picture
        file_before = os.path.join(self.root_path, self.root_dhm_content[1])  # before dhm picture
        dc_after = Container(file=file_after)
        dc_before = Container(file=file_before)
        dc_after_img = dc_after._items["meas/image.png"].data
        dc_before_img = dc_before._items["meas/image.png"].data
        dc_data_holo = dc_before._items["data/hologram.json"].data
        dc_data_objective = dc_before._items["data/objective.json"].data
        dc_dict_holo = {f"meas/dhm/finished.png": dc_after_img, f"meas/dhm/background.png": dc_before_img,
                        "data/hologram.json": dc_data_holo, "data/objective.json": dc_data_objective}

        # Camera Images
        file_after = os.path.join(self.root_path, self.root_camera_content[0])  # after camera picture
        file_before = os.path.join(self.root_path, self.root_camera_content[1])  # before camera picture
        dc_after = Container(file=file_after)
        dc_before = Container(file=file_before)
        dc_after_img = dc_after._items["meas/image.png"].data
        dc_before_img = dc_before._items["meas/image.png"].data
        dc_data_camera = dc_before._items["data/camera.json"].data
        dc_dict_camera = {f"meas/camera/finished.png": dc_after_img, f"meas/camera/background.png": dc_before_img,
                          f"data/camera.json": dc_data_camera}

        # update dict with same info
        tmp_dict.update(dc_dict_holo)
        tmp_dict.update(dc_dict_camera)
        return tmp_dict

    def get_root_content(self):
        content = os.listdir(self.root_path)
        zdc_content = fnmatch.filter(os.listdir(path), '*.zdc')  # make sure only zdc files are selected
        self.root_content.extend(content)
        self.root_zdc_content.extend(zdc_content)
        dhm_root_content = fnmatch.filter(zdc_content, '[dhm]*')  # make sure only zdc files are selected
        camera_root_content = fnmatch.filter(zdc_content, '[camera]*')  # make sure only zdc files are selected
        self.root_dhm_content.extend(dhm_root_content)
        self.root_camera_content.extend(camera_root_content)
        return content

    def get_dhm_content(self):
        content = fnmatch.filter(os.listdir(self.dhm_path), '*.zdc')  # make sure only zdc files are selected
        self.dhm_content.extend(content)
        return content

    def get_camera_content(self):
        content = fnmatch.filter(os.listdir(self.camera_path), '*.zdc')  # make sure only zdc files are selected
        self.camera_content.extend(content)
        return content

    def get_program_content(self):
        content = fnmatch.filter(os.listdir(self.program_path), '*.txt')
        self.program_content.extend(content)
        return content

    def get_program_dict(self):
        tmp_dict: dict[Any, Any] = {}
        if not self.program_content:
            self.get_program_content()
        for i in range(len(self.program_content)):
            program = open(os.path.join(self.program_path, self.program_content[i]), "r").read()
            layer = {f"program_files/layer_{self.program_content[i][-7:-4]}.txt": program}
            tmp_dict.update(layer)
        return tmp_dict

    def get_dhm_dict(self):
        tmp_dict: dict[Any, Any] = {}
        if not self.dhm_content:
            self.get_dhm_content()
        for i in range(len(self.dhm_content)):
            file = os.path.join(self.dhm_path, self.dhm_content[i])
            dc = Container(file=file)
            dc_img = dc._items["meas/image.png"].data
            dictionary = {f"meas/dhm/layer_{i}.png": dc_img}
            tmp_dict.update(dictionary)
        return tmp_dict

    def get_camera_dict(self):
        tmp_dict: dict[Any, Any] = {}
        if not self.camera_content:
            self.get_camera_content()
        for i in range(len(self.camera_content)):
            file = os.path.join(self.camera_path, self.camera_content[i])
            dc = Container(file=file)
            dc_img = dc._items["meas/image.png"].data
            dictionary = {f"meas/camera/layer_{i}.png": dc_img}
            tmp_dict.update(dictionary)
        return tmp_dict


if __name__ == "__main__":
    path = "C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/20240820_dhm_testprint_Zeiss 63x/structures/corner_bl"
    description = "Structure corner_bl, which is the bottom left corner of the structure."
    test = StructureCollector(path, exp_description=description)
    # a=test.get_dhm_content()
    # b=test.get_root_content()
    # test.get_program_content()
    # print(b)
    # b = test.get_root_content()
    # a = test.get_dhm_dict()
    c = test.get_container()


