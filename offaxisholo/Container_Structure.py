import fnmatch

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

        # ToDo: Anpassen der hier folgenden Sachen-> Wichtig dabei , dass jedes Bild und Hologram aufgenommen wurde
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


class ExperimentCollector:
    def __init__(self, root_path):
        # super(StructureContainer, self).__init__(root_path)

        # xxx\20240820_print\structures\corner_bl
        self.root_path = root_path
        self.dhm_path = os.path.join(self.root_path, 'dhm')
        self.camera_path = os.path.join(self.root_path, 'camera')
        self.program_path = os.path.join(self.root_path, 'programs')

        # initialize content lists
        self.dhm_content = []  # zdc files in the dhm folder
        self.camera_content = []  # zdc files in the camera folder
        self.root_content = []  # all files located in the root folder
        self.root_zdc_content = []  # list of all zdc files in root directory
        self.root_dhm_content = []  # list of the dhm-containers in root directory
        self.root_camera_content = []  # list of the camera-containers in root directory

        # initialize dictionaries
        self.program_dict = {}

    def collect_dict(self):
        dhm_dict = self.get_dhm_dict()
        # ToDO should be used to create the item dict for Container
        items = {}

        return items

    def container(self, container_name):
        pass

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
        for i in range(len(content)):
            program = open(os.path.join(self.program_path, content[i]))
            layer = {f"program_files/layer_{content[i][-7:-4]}.txt": program}
            self.program_dict.update(layer)
        # ToDo : Überprüfen ob es als txt abgespeichert wird .. bisher ist es nur ein wrapper. wenn man program.read() nimmt, dann hat man den ganzen inhalt aber als string
        # ToDo: find a way to save it as txt in new folder in data

        # --> vielleicht in eine liste und jeden eintrag der liste mit update() ins items verzeichnis heben
        return self.program_dict

    def get_dhm_dict(self):
        tmp_dict = {}
        if not self.dhm_content:  # sanity check
            self.get_dhm_content()
        for i in range(len(self.dhm_content)):
            file = os.path.join(self.dhm_path, self.dhm_content[i])
            dc = Container(path=file)
            dc_img = dc._items["meas/image.png"]
            dictionary = {f"meas/dhm/self.dhm_content[i].png": dc_img}
            tmp_dict.update(dictionary)
        return tmp_dict

def get_datafiles(root, subdir=False, ending='.dat') -> list:
    if subdir:
        files='**\*'+ending
        path = os.path.join(root, files)
    else:
        files = '*' + ending
        path = os.path.join(root, files)
    try:
        return glob.glob(path)
    except Exception as e:
        raise FileNotFoundError(f"No directory {root}. Exception {e}")


def get_hologram(path, filename=None) -> "holo_get":
    if filename is None:
        if path[-4:] == ".zdc":
            dc = Container(file=path)
        else:
            path = os.path.join(path, ".zdc")
            try:
                dc = Container(file=path)
            except FileNotFoundError:
                raise FileNotFoundError(f"No file at {path} found.")
    else:
        path = os.path.join(path, filename)
        try:
            dc = Container(file=path)
        except Exception as e:
            print(f"{type(e)}: {e}")
    return dc._items['meas/image.png']


if __name__ == "__main__":
    path = "C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/dhm_paper/20240820_dhm_testprint_Zeiss 63x/structures/corner_bl"

    test = ExperimentCollector(path)
    # a=test.get_dhm_content()
    # b=test.get_root_content()
    # test.get_program_content()
    # print(b)

    a=test.get_dhm_dict()
    print(a)

"""

"""