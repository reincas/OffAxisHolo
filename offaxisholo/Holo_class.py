import fnmatch
import os
from skimage.restoration import unwrap_phase
import numpy as np
import scidatacontainer
from matplotlib import pyplot as plt
from scidatacontainer import Container

from offaxisholo.reconstruct import locateOrder, rollImage, circularMask, holo2Field, angularSpectrum, getField
from offaxisholo.utils import get_hologram


class HoloStructure(Container):

    def __init__(self, logger=None):
        super().__init__(logger)
        self.logger = logger

        # Initialisation
        self.layer_container = {}

        self.layer = None


class DHM_DUMMY:
    def __init__(self):
        self.pixel_pitch = []
        self.wavelength = 0
        self.prop_dist = 2000


class Reconstruction:
    def __init__(self, hologram, objective, DHM=None):
        if isinstance(hologram, scidatacontainer.fileimage.PngFile):
            self.hologram = hologram.data
        else:
            self.hologram = hologram

        if len(self.hologram.shape) != 2:
            raise Exception(f"An Error occurred. 2D Hologram image required! Check data.")
        if self.hologram.shape[0] != self.hologram.shape[1]:
            raise Exception("Quadratic hologram image required!")

        # initialisation of spectrum and field saving variables
        self.spectrum_not_shifted = None
        self.spectrum_shifted = None
        self.spectrum_masked = None
        self.field = None

        self.first_diffraction_order_pos = []

        self.img_save_path = None
        self.var_4_saving = 0

        # DHM Informations
        # ToDo: Import DHM Class and extract information
        if DHM is None:
            self.DHM = DHM_DUMMY()
            if objective == "Zeiss 20x":
                self.DHM.pixel_pitch = [0.276, 0.276]  # 20x objective
            elif objective == "Zeiss 63x":
                self.DHM.pixel_pitch = [0.0869, 0.0869]  # 63x objective
            else:
                raise NotImplementedError(f"Objective {objective} not implemented!")
            self.DHM.prop_dist = -2000.0  # Distance of sensor to back focal plane of tube lens
            self.DHM.wavelength = 0.000675  # wavelength of the laser
        else:
            raise Warning('Please check if DHM has the necessary attributes!')

    def _locate_order(self) -> tuple[int, int]:
        # ToDo: Check how good the locateOrder works with 63x objective
        spectrum, fx, fy, weight = locateOrder(holo=self.hologram)
        self.spectrum_not_shifted = spectrum

        self.first_diffraction_order_pos = [fx, fy]
        self._calc_radius_mask(fx, fy)
        return fx, fy

    def _holo2field(self, holo=None, r=None):
        if not self.first_diffraction_order_pos:
            fx, fy = self._locate_order()
        else:
            fx = self.first_diffraction_order_pos[0]
            fy = self.first_diffraction_order_pos[1]
        if r is None:
            r = self._calc_radius_mask(fx, fy)
        if holo is None:
            holo = self.hologram

        self.field = holo2Field(holo, fx, fy, r)

    def _propagation(self, holo=None, field=None, z=None, r=None):
        if z == None:
            z = self.DHM.prop_dist
        if field is None:
            if self.field is None:
                self._holo2field(holo=holo, r=r)
            field = self.field

        dx = self.DHM.pixel_pitch[0]
        dy = self.DHM.pixel_pitch[1]
        wl = self.DHM.wavelength
        field = angularSpectrum(field, z=z, wavelength=wl, dx=dx,
                                dy=dy)  # ToDo: Kontrollieren wo der unterschied zwischen pixelpitch und dem pc ist
        self.reconstructed_field = field

    def reconstruct(self, holo=None, input_field=None, z=None, r=None, pos_first_order=None, return_field=False):
        """
        Reconstruction of the hologram with the opportunity to tweak inputs.
        z Propagation Distance
            Returns the intensity and phase
        if return_field == True:
            Returns the field
        """
        if pos_first_order is not None:
            if isinstance(pos_first_order, int | float):
                self.first_diffraction_order_pos = [pos_first_order, pos_first_order]
            if isinstance(pos_first_order, list) and len(pos_first_order) == 2:
                self.first_diffraction_order_pos = pos_first_order
            else:
                raise Exception(f"Check type of {pos_first_order}. Allowed are int, float and lists.")
        self._propagation(holo=holo, field=input_field, z=z, r=r)
        if return_field:
            return self.reconstructed_field

        int_reconstructed = self.intensity(self.reconstructed_field)
        phase_reconstructed = self.phase(self.reconstructed_field)
        return int_reconstructed, phase_reconstructed

    def phase_unwrapping(self, phase_wrapped):
        # ToDo: Implement phase unwrapping
        phase_unwrapped = unwrap_phase(phase_wrapped)
        return phase_unwrapped

    def run(self):
        int, phase = self.reconstruct()
        phase_unwrapped = self.phase_unwrapping(phase)
        # account for aberration
        # give the intensity and unwrapped phase back

    def plotImage(self, img, title=None, save=False):
        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(self.img_save_path, name)
                plt.imsave(save_path, img, cmap='viridis')
            else:
                save_path = os.path.join(self.img_save_path, f"picture{self.var_4_saving}.png")
                plt.imsave(save_path, img, cmap='viridis')
                self.var_4_saving += 1
        else:
            if title == None:
                plt.imshow(img, cmap='viridis')
            else:
                plt.imshow(img, cmap='viridis')
                plt.title(title)
            plt.show()  # show image
        return

    def _calc_radius_mask(self, fx, fy):
        return np.sqrt(fx * fx + fy * fy) - 5

    def intensity(self, inp, log=True):
        out = np.abs(inp)
        if not log:
            out = out * out
        else:
            out = 20 * np.log(out)
            out[out == np.inf] = 0
            out[out == -np.inf] = 0
        return out

    def phase(self, inp):
        out = np.angle(inp)
        return out

    # function without usage
    def shift_spectrum(self, spectrum, fx, fy):
        spectrum_shifted = rollImage(spectrum, fx, fy)
        return spectrum_shifted

    def masked_spectrum(self, spectrum, r=None):
        if not self.first_diffraction_order_pos:
            fx, fy = self._locate_order()
        else:
            fx = self.first_diffraction_order_pos[0]
            fy = self.first_diffraction_order_pos[1]

        if r is None:
            r = np.sqrt(fx * fx + fy * fy) - 5  # ToDo: Fragen warum hier -5 sind
        spectrum_masked = circularMask(spectrum, r)
        return spectrum_masked

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically determine a subfolder /img/ for the images - maybe done in the future for the complete structure class

    def evaluation(self, save_img=False):
        """
        All-in-one method for investigating a taken hologram.
        If save_img = True , then the plotted images will be safed to self.img_save_path
                                - set the path with set_save_path(path)
        """
        # ToDo Kontrollieren wie es aussieht mit den namen der einzelnen komponenten

        if save_img and self.img_save_path is None:
            raise Exception("A path for saving the images is needed! \nUse the method set_save_path for this purpose.")

        # Spectrum of camera image (magnitude of field)
        fx, fy = self._locate_order()
        r = self._calc_radius_mask(fx, fy)

        print(f"Order offset x: {fx}")
        print(f"Order offset y: {fy}")
        print(f"Radius mask x: {r}\n")  # ToDo Vergleichen mit unten der Funktion wo r auch berechnet wird.

        # self.plotImage(self.spectrum_not_shifted, "Spectrum not shifted", save=save_img)
        self.plotImage(self.intensity(self.spectrum_not_shifted), "Intensity not shifted", save=save_img)
        self.plotImage(self.phase(self.spectrum_not_shifted), "Phase not shifted", save=save_img)

        spectrum_shifted = self.shift_spectrum(self.spectrum_not_shifted, fx, fy)
        spectrum_masked = self.masked_spectrum(spectrum_shifted, r=r)

        self.plotImage(self.intensity(spectrum_shifted), "Shifted spectrum - intensity", save=save_img)
        self.plotImage(self.intensity(spectrum_masked), "Masked spectrum - intensity", save=save_img)

        field = getField(spectrum_masked)
        self._propagation(holo=self.hologram)
        field_prop = self.reconstructed_field

        self.plotImage(self.intensity(field), "intensity before prop", save=save_img)
        self.plotImage(self.phase(field), "phase before prop and unwrapping", save=save_img)

        phase = self.phase_unwrapping(self.phase(field))
        phase_prop = self.phase_unwrapping(self.phase(field_prop))
        int_rolled = self.intensity(field_prop)

        self.plotImage(int_rolled, "intensity after prop", save=save_img)
        self.plotImage(phase, "phase after unwrapping before prop", save=save_img)
        self.plotImage(phase_prop, "phase after prop and then unwrapping", save=save_img)

        # prop_dist = 2000.0  # Distance of sensor to back focal plane of tube lens
        # field_rolled = getField(spectrum_rolled)
        # field_rolled = angularSpectrum(field_rolled, z=-prop_dist, wavelength=1E-3 * wl, dx=dx,
        #                                dy=dy)  # ToDo: Kontrollieren wo der unterschied zwischen pixelpitch und dem pc ist

    def debugging(self):
        spectrum, fx, fy, weight = locateOrder(holo=self.hologram)
        spectrum_shifted = self.shift_spectrum(spectrum, fx, fy)

        self.first_diffraction_order_pos = [fx, fy]

        # Bestimmung des Radius
        r = self._calc_radius_mask(fx, fy)
        r0 = dhm.objective["dcRadius"]

        rmax = np.sqrt(fx ** 2 + fy ** 2) - r0
        rmax = min(rmax, abs(fx), w // 2 - abs(fx), abs(fy), h // 2 - abs(fy))



        spectrum_masked = self.masked_spectrum(spectrum_shifted, r=r)

        self.plotImage(self.intensity(spectrum), "Intensity not shifted", save=False)
        self.plotImage(self.intensity(spectrum_shifted), "Intensity shifted", save=False)
        self.plotImage(self.intensity(spectrum_masked), "Intensity shifted and masked", save=False)


'''
# method from "live_tilt_cv.py" - obviously adapted to current directory - not changed that much :)
def getImage(dhm):
    holo, count = dhm.getimage()
    spectrum, fx, fy, weight = reconstruct.locateOrder(holo, 16)
    dhm.log.info(f"First order coordinates: {fx:d}, {fy:d} [{100 * weight:.1f}%]")

    maxpixel = 255
    numof = np.count_nonzero(holo >= maxpixel)
    dhm.log.info(f"Overflow pixels: {numof:d}")

    img = np.log(np.abs(spectrum))
    h, w = img.shape
    vmax = 0.5 * np.max(img)
    img = np.where(img > vmax, vmax, img)

    img = image.normcolor(img)
    r0 = dhm.objective["dcRadius"]
    img = image.drawCircle(img, 0, 0, r0, image.CV_RED, 1)

    rmax = np.sqrt(fx ** 2 + fy ** 2) - r0
    rmax = min(rmax, abs(fx), w // 2 - abs(fx), abs(fy), h // 2 - abs(fy))
    dhm.log.info(f"Maximum radius: {rmax:.0f} pixels")
    if rmax > 0:
        img = image.drawCircle(img, fx, fy, rmax, image.CV_RED, 1)
        img = image.drawCircle(img, -fx, -fy, rmax, image.CV_RED, 1)
    img = image.drawCross(img, fx, fy, 30, image.CV_RED, 1)
    img = image.drawCross(img, -fx, -fy, 30, image.CV_RED, 1)
    return img
'''

if __name__ == "__main__":

    path = "C:/Users/hanne/Desktop/Test4DHMReconstruction"
    path_pic = os.path.join(path, "dhm_stair_galvo.41")
    holo = get_hologram(path=path_pic)
    DHM_obj = Reconstruction(hologram=holo, objective="Zeiss 20x")
    DHM_obj.debugging()

    # DHM_obj.set_save_path(path=save_path)
    # DHM_obj.debugging(save_img=True)
