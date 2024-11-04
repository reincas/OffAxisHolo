import numpy as np
from scidatacontainer import Container

"""
Container for the structure and the experiment.
ToDo: Define the necessary parameters of a Container!
-> Absprache mit reinhard
"""

# ToDo: porperties für Structure container weiter bearbeiten!

class StructureContainer(Container):
    containerType = "StructureContainer"
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
        # img = self.kwargs.pop("img")
        # if not isinstance(img, np.ndarray) or len(img.shape) != 2:
        #     raise RuntimeError("Hologram image expected!")
        # items["meas/image.png"] = img

        # Camera parameters
        # params = self.kwargs.pop("params")
        # if not isinstance(params, dict):
        #     raise RuntimeError("Parameter dictionary expected!")
        # items["data/camera.json"] = params
        #
        # # Objective parameters
        # objective = self.kwargs.pop("objective")
        # if not isinstance(objective, dict):
        #     raise RuntimeError("Objective dictionary expected!")
        # items["data/objective.json"] = objective

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
    def objective(self):
        """ Shortcut to the parameter data dictionary. """
        return self["data/objective.json"]

    @property
    def dhm_params(self):
        """ Shortcut to the dhm_parameter-parameter data dictionary. """
        obj_name = self["data/objective.json"]["key"]
        dc_radius = self["data/objective.json"]["dcRadius"]
        magnification = self["data/objective.json"]["magnification"]
        na_objective = self["data/objective.json"]["numericalAperture"]
        wavelength_m = self["data/hologram.json"]["device"]["laser"]["wavelengthUm"] * 10e-6
        if obj_name == "Zeiss 63x":
            pixel_pitch_m = [0.0869e-6, 0.0869e-6]  # ToDo implement it for 63 obj
        else:
            pixel_pitch_m = [self["data/hologram.json"]["objective"]["xPixelSizeUm"] * 10e-6,
                             self["data/hologram.json"]["objective"]["yPixelSizeUm"] * 10e-6]
        if "propagationDistance" in self["data/hologram.json"]["device"]["dhm"]:
            propagation_distance = self["data/hologram.json"]["device"]["dhm"]["propagationDistance"] # ToDo implement it!
        else:
            propagation_distance = 0.0
        n_resin = 1.5
        if "info/substrate.json" in self:  # ToDo create the substrate json file
            if "refractive index material" in self["info/substrate.json"]:
                n_resin = self["info/substrate.json"]["refractive index material"]

        dhm_params = {"objective name": obj_name,
                      "DC radius": dc_radius,
                      "magnification": magnification,
                      "propagation distance": propagation_distance,
                      "wavelength": wavelength_m,
                      "pixel pitch": pixel_pitch_m,
                      "NA objective": na_objective,
                      "refractive index": n_resin}

        return dhm_params

    @property
    def background_hologram(self):
        """ Shortcut to the background hologram. """
        return self["meas/dhm_parameter/background.png"]

    @property
    def complete_hologram(self):
        """ Shortcut to the last taken hologram. """
        return self["meas/dhm_parameter/finished.png"]

    @property
    def number_of_layer(self):
        return self["info/properties.json"]["number of layer"]

    @property
    def camera_params(self):

        """ Shortcut to the amera-parameter data dictionary. """

        return self["data/camera.json"]

    @property
    def location(self):

        """ Return xyz position of the image or None. """

        if "data/location.json" not in self:
            return None
        return self["data/location.json"]


class ExperimentContainer(Container):
    containerType = "ExperimentContainer"
    containerVersion = 1.0

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
        meta["title"] = "Experiment Information"
        meta["general description"] = ("Complete Experiment of a Femtika print.")

        # Update items dictionary
        items["content.json"] = content
        items["meta.json"] = meta

        # Replace container items dictionary
        self.kwargs["items"] = items

    def __post_init__(self):

        """ Initialize this container. """

        # Type check of the container
        if (self.content["containerType"]["name"] != self.containerType) or \
                (self.content["containerType"]["version"] != self.containerVersion):
            raise RuntimeError(f"Containertype must be '{self.containerType}'!")

    @property
    def objective(self):
        """ Shortcut to the parameter data dictionary. """
        return self["data/objective.json"]

    @property
    def dhm_params(self):
        """ Shortcut to the parameter data dictionary. """
        return self["data/hologram.json"]

    @property
    def camera_params(self):
        """ Shortcut to the parameter data dictionary. """
        return self["data/camera.json"]

    @property
    def location(self):
        """ Return xyz position of the image or None. """
        if "data/location.json" not in self:
            return None
        return self["data/location.json"]
