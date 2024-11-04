"""
GENERAL TODOS WHICH HAVE TO BE DONE IN THE FUTURE.

ToDo 1:     license thing on top of this file
ToDo 2:     update regularly this usage if new components are added to the aerotech file/ saving - system
DONEToDo 3:     Make a "StructureContainer" of the normal Container in the StructureCollector in save_container
ToDo 4:     saving and using the locations of the camera images -  until now it is only saved in the original zdc
ToDo 5:

EXAMPLE USAGE:
--- only two inputs necessary ---

path = ".output/dhm_paper/print/structures/lens_galvo1"
description = "Structure lens 1, which is the bottom left corner of the structure."

--- only two line to get the container object and have a saved .zdc file ---

collector = StructureCollector(path, exp_description=description)
structureContainer = collector.collect_container()

"""
from .structure import StructureContainer, ExperimentContainer
from .util import update_container, set_container_description

from scidatacontainer import Container, load_config
from typing import Any
import cv2 as cv
import fnmatch
import json
import os


class StructureCollector:
    def __init__(self, root_path, exp_description=None):
        # xxx\20240820_print\structures\corner_bl
        self.root_path = root_path
        self._pre_init__()
        # container variables
        self.container = None  # container object of the structure data
        self.saving_directory_container = None  # saving directory if the container has been saved
        self.exp_description = None

        if exp_description is not None:
            if isinstance(exp_description, str):
                self.exp_description = exp_description
            elif isinstance(exp_description, dict):
                # If this is changed - it also has to be changed in self.exchange_general_information
                raise NotImplementedError("Not yet implemented if more than one information is given.")

        self._post_init()  # setting reference

    @property
    def number_of_layers(self):
        # should always be the same amount of camera images and holograms
        assert len(self.camera_content) == len(self.dhm_content)
        return len(self.dhm_content)

    @property
    def layer_height(self):
        return None

    @property
    def hatch_size(self):
        return None

    @property
    def max_width(self):
        return None

    @property
    def max_length(self):
        return None

    def _pre_init__(self):
        """
        Method for initiating all variables for a cleaner view
        :return:
        """
        self.dhm_path = os.path.join(self.root_path, 'dhm')
        self.camera_path = os.path.join(self.root_path, 'camera')
        self.program_path = os.path.join(self.root_path, 'programs')

        # initialize content lists
        self.program_content = []  # txt files in the program folder
        self.dhm_content = []  # zdc files in the dhm_parameter folder
        self.camera_content = []  # zdc files in the camera folder
        self.root_content = []  # all files located in the root folder
        self.root_zdc_content = []  # list of all zdc files in root directory
        self.root_dhm_content = []  # list of the dhm_parameter-containers in root directory
        self.root_camera_content = []  # list of the camera-containers in root directory

        # initialize dictionaries
        self.struc_dict = {}  # dictionary of the structure data

    def _post_init(self):
        """
        Only for calling methods in connection to referencing
        :return:
        """
        self.reference_container = None
        self.uuid = None
        # Change ContainerType and get meta and content.json
        tmp = self.get_general_info()
        # reference file: get general info - get background dict -> reference file and uuid
        self.struc_dict.update(tmp)

    def collect(self):
        """
        Collects all .zdc files of an experiment into one single container.
        :return:
        container object of the structure
        """
        self.collect_dict()
        self.save_container()
        return self.container

    def collect_dict(self):
        """
        Methode for collecting all measure-dictionaries of the .zdc files
        """
        # Get all DHM images
        tmp = self.get_dhm_dict()
        self.struc_dict.update(tmp)
        # Get all Camera images
        tmp = self.get_camera_dict()
        self.struc_dict.update(tmp)
        # Collect all programs
        tmp = self.get_program_dict()
        self.struc_dict.update(tmp)

        # updating properties
        properties = {
            "number of layer": self.number_of_layers,
            "hatch size": self.hatch_size,
            "layer height": self.layer_height,
            "maximum length": self.max_length,
            "maximum width": self.max_width
        }
        self.struc_dict.update({"info/properties.json": properties})

    def save_container(self, save_path=None):
        """
        Saves the Container by default in the root directory of the folder of experiment (structure-folder).
        :param save_path: saving directory of the container object
        :return:
        """
        if save_path is None:
            save_path = self.root_path + ".zdc"
        if save_path[-4:] != ".zdc":
            save_path = save_path + ".zdc"
        if self.struc_dict == {}:
            raise NotImplementedError("No Data collected!")
        else:
            structure_container = StructureContainer(items=self.struc_dict)
            self.container = structure_container
            self.saving_directory_container = save_path
            structure_container.write(save_path)

    def get_container(self, save=False):
        if self.struc_dict == {}:
            self.collect_dict()
            if save:
                self.save_container()
        return self.struc_dict

    def get_general_info(self):
        """
        Should be used for merging all the different general data/ infos of the different container into one.
        meta.json and content.json in the changed format (StructureContainer)
        """
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

        return tmp_dict

    def exchange_general_information(self, tmp_dict):
        try:
            dc = StructureContainer(items=tmp_dict)
            if self.exp_description is not None:
                tmp_dict["meta.json"]["description"] = self.exp_description
            tmp_dict = dc.items()
        except Exception:
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

        self.reference_container = Container(file=os.path.join(self.root_path, self.root_camera_content[0]))
        self.uuid = self.reference_container.uuid
        tmp = self.get_reference_information(reference_file=self.reference_container)
        tmp_dict.update(tmp)

        # DHM Images
        file_after = os.path.join(self.root_path, self.root_dhm_content[0])  # after dhm_parameter picture
        file_before = os.path.join(self.root_path, self.root_dhm_content[1])  # before dhm_parameter picture
        dc_after = Container(file=file_after)
        dc_before = Container(file=file_before)
        dc_after_img = dc_after._items["meas/image.png"].data
        dc_before_img = dc_before._items["meas/image.png"].data
        dc_dict_holo = {f"meas/dhm/finished.png": dc_after_img, f"meas/dhm/background.png": dc_before_img}

        # Camera Images
        file_after = os.path.join(self.root_path, self.root_camera_content[0])  # after camera picture
        file_before = os.path.join(self.root_path, self.root_camera_content[1])  # before camera picture
        dc_after = Container(file=file_after)
        dc_before = Container(file=file_before)
        dc_after_img = dc_after._items["meas/image.png"].data
        dc_before_img = dc_before._items["meas/image.png"].data
        dc_dict_camera = {f"meas/camera/finished.png": dc_after_img, f"meas/camera/background.png": dc_before_img}

        # update dict with same info
        tmp_dict.update(dc_dict_holo)
        tmp_dict.update(dc_dict_camera)
        return tmp_dict

    def get_reference_information(self, reference_file):
        """
        Gets the UUID of the reference-file and sets it as the UUID of this container.
        Also gets and changes the content and meta.json files.

        # ToDo(HR): Future improvement to the GENERAL experiment data gathering has to be implemented in here
        """
        # getting information necessary for SciDataContainer and which is not changing throughout the structure
        tmp_dict = {}
        if isinstance(reference_file, Container):
            dc = reference_file
        else:
            dc = Container(file=os.path.join(self.root_path, reference_file))

        holo_dc = Container(file=os.path.join(self.root_path, self.root_dhm_content[0]))
        data_holo = holo_dc._items["data/hologram.json"].data

        dc_content = dc.content
        dc_meta = dc.meta
        tmp_dict.update({"content.json": dc_content, "meta.json": dc_meta})
        tmp_dict = self.exchange_general_information(tmp_dict)

        # Data, which should come from the reference file
        data_objective = dc._items["data/objective.json"].data
        data_camera = dc._items["data/camera.json"].data
        data_location = dc._items["data/location.json"].data
        dict_data = {"data/hologram.json": data_holo, "data/objective.json": data_objective,
                     f"data/camera.json": data_camera, f"data/location.json": data_location}

        tmp_dict.update(dict_data)
        return tmp_dict

    def get_root_content(self):
        content = os.listdir(self.root_path)
        zdc_content = fnmatch.filter(os.listdir(self.root_path), '*.zdc')  # make sure only zdc files are selected
        self.root_content.extend(content)
        self.root_zdc_content.extend(zdc_content)
        dhm_root_content = fnmatch.filter(zdc_content, '[dhm_parameter]*')  # make sure only zdc files are selected
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
            layer = {f"data/program_files/layer_{self.program_content[i][-7:-4]}.txt": program}
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
            dictionary = {f"meas/dhm/raw/layer_{i}.png": dc_img}
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


class ExperimentCollector:
    """
    Class for an Experiment. Automatically creates an .zdc file for each structure.
    Container is based on the layer-container of the plane-fitting algorithm!
    ToDo: DHM.json in layer.zdc is not correct!
    ToDo: description of the sinlge structures has to be automated!
    ToDo Change later to the already existing Experiment container
            ToDo: Look at creation of plane.zdc for gathering necessary information
    """
    def __init__(self, root_path):
        self.root_path = root_path
        self.container_path = self.root_path + ".zdc"
        self.struct_info_dict = {}
        self.structure_path_list = []
        self.structure_name = []
        self.structure_reference_list = {}

    def init_container(self):
        # ToDo: After creation of experiment zdc, which is always constructed, rework this!#
        config = load_config(
            author="Hannes Robben",
            email="hannes.robben@phoenixd.uni-hannover.de",
            organization="PhoenixD",
            orcid="000"
        )
        self.container = ExperimentContainer(config=config)
        self.container.write(self.root_path+".zdc")

    def collect(self):
        # Initialization of experiment object with path of the experiment
        self.init_container()  # creation of a base-container

        # Saving all general information: structure.json, experiment.png, oplscan.txt, plane-fit
        self.structure_information()  # gets the list of the names of the structures + saving the file in .zdc
        self.experiment_information()  # gets the .png plotting file + console + opl_scan file and saves it
        self.planefit_information()  # handles all the information of the plane-fitting process

        # do the zdc stuff with the structures
        self.create_structure_container()  # create the individual structure container
        self.create_reference_file()  # creation of a reference file containing all UUIDs of the different structures

    def structure_information(self):
        pfad = self.root_path + "\\structures.json"
        with open(pfad) as file:
            dictionary = json.load(file)
        self.struct_info_dict = dictionary
        for structure in dictionary:
            self.structure_name.append(structure["name"])
            self.structure_path_list.append(os.path.join(self.root_path, "structures", structure["name"]))
        # uploading the structure.json file to the experiment container
        update_container(self.container_path, {"meas/structures.json": dictionary})

    def create_reference_file(self):
        """
        Creates a dictionary of UUIDs for the different structures and saves it to the container
        :return:
        """
        i = 0
        for structure_path in self.structure_path_list:
            try:
                path_with_ending = structure_path + ".zdc"
                dc = Container(file=path_with_ending)
                tmp_dict = dc.items()
            except FileNotFoundError:
                print(f"No Container exists in directory {structure_path}.")

            self.structure_reference_list.update({f"{self.structure_name[i]}": tmp_dict["content.json"]["uuid"]})
            i += 1
        # saves the dictionary of the uuids
        update_container(container_path=self.container_path,
                         update_dict={"meas/structure_reference.json": self.structure_reference_list})

    def experiment_information(self):
        """
        Plot of the Experiment, the console log file and the opl_scan is saved.
        """
        img = cv.imread(os.path.join(self.root_path, "experiment.png"))
        console = open(os.path.join(self.root_path, "console.log"), "r").read()
        opl_scan = open(os.path.join(self.root_path, "oplscan/opl.txt"), "r").read()
        tmp_info = {f"info/Plot_of_Experiment.png": img, f"log/console.log": console, f"info/opl_scan.txt": opl_scan}
        update_container(self.container_path, tmp_info)

    def planefit_information(self):
        """
        Handles all necessary information of the plane-fiiting procedure and saves it.
        """
        # ToDo Find a good way to handle all the information in the layer container
        layer_container_path = os.path.join(self.root_path, "planefit/plane.zdc")
        plane_dictionary = {}

        dc = Container(file=layer_container_path)
        tmp_dict = dc.items()
        for key, val in tmp_dict.items():
            if key == "meas/result.json":
                plane_dictionary.update({"meas/planefit/result.json": val})
            elif key == "meas/steps.json":
                plane_dictionary.update({"meas/planefit/steps.json": val})
            elif key == "reference.json":
                plane_dictionary.update({"meas/planefit/layer_reference_ids.json": val})
            elif key == "content.json":
                plane_dictionary.update({"meas/planefit/original_uuid.txt": val["uuid"]})

        # Saving the dictionary to the Container File
        update_container(self.container_path, plane_dictionary)

    def create_structure_container(self):
        for structure_path in self.structure_path_list:
            structure = StructureCollector(structure_path, exp_description="")
            structureContainer = structure.collect()
            save_path = structure.saving_directory_container
            # ToDo change the description in a informative way: where is the structure - maybe grid number if possible, what was printed and what is the name
            description = ""
            set_container_description(save_path, description=description)







if __name__ == "__main__":
    # path = "C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/20240820_dhm_testprint_Zeiss 63x/structures/corner_bl"
    path = "C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/print4paper_20240912_dhm_Zeiss 63x/structures/lens_galvo1"
    description = "Structure lens1, which is the bottom left corner of the structure."

    import datetime
    now = datetime.datetime.now()

    collector = StructureCollector(path, exp_description=description)
    structureContainer = collector.collect()
    structureContainer.number_of_layer()

    oh = datetime.datetime.now()
    time = oh - now

    print(f"Collection of the structured data took {time}.")


