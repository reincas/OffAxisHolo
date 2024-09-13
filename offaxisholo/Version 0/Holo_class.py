import os
from skimage.restoration import unwrap_phase
import numpy as np
import scidatacontainer
from matplotlib import pyplot as plt

from offaxisholo.reconstruct import locateOrder, rollImage, circularMask, angularSpectrum, getField, \
    getSpectrum
from offaxisholo import get_hologram

class DHM:
    def __init__(self):
        self.pixel_pitch = []
        self.wavelength = 0
        self.prop_dist = 20
        self.n_resin = 0


# ToDo: Umschreiben der gesamten Klasse sodass immer nur der Container erforderlich ist. Braucht man eine nicht-Container VErsion?

class HologramTomographic:

    def __init__(self, root_directory, logger=None):
        self.logger = logger

        # Initialisation
        self.layer_container = {
            'Position': [],  # should have all the global information available.
            'Layer Height': 0,  # specific height fpr this layer.
        }
        self.layer = None

        # ToDo: figure out how i want to use the information on where the holograms are saved.
        self.root_directory = root_directory
        self.working_directory = os.path.join(root_directory,
                                              'Reconstructed')  # should act as a saving directory for all the data and so on.

    def add_layer(self, layer):
        # adding a hologram to the list/ dict of layer.
        # layer.run have to be called
        # set background with this background here
        pass

    def set_background(self):
        # field, phase, int should be saved for giving it to the layers.
        pass

    def _get_hologram_layer_list(self):
        # returns a list of the holograms for the investigated structure
        # with fnmatch and glob
        pass

    def _getLoggerInformation(self):
        # acts like a catching point for all the information of the logger folder.
        # all the informations are dozen of times available. - make it one
        # add additional information regarding the structures
        # how many layers were printed?
        # what kind of structure
        # where
        # maximal dimension of those structures
        pass



class Reconstruction:
    def __init__(self, hologram, objective, DHM=None, background_hologram=None):
        if isinstance(hologram, scidatacontainer.fileimage.PngFile):
            self.hologram = hologram.data
        else:
            self.hologram = hologram
        if len(self.hologram.shape) != 2:
            raise Exception(f"An Error occurred. 2D Hologram image required! Check data.")
        if self.hologram.shape[0] != self.hologram.shape[1]:
            raise Exception("Quadratic hologram image required!")
        # initialisation of spectrum and field saving variables

        self.reconstructed_field = None
        self.int_reconstructed = None
        self.phase_reconstructed = None
        self.phase_unwrapped = None
        self.phase_compensated = None
        self.height_profile = None

        self.first_diffraction_order_pos = []
        self.radius_mask = None

        self.objective = objective
        # ToDo Check these variables
        self.img_save_path = None
        self.var_4_saving = 0

        # DHM Information
        # ToDo: Import DHM Class and extract information
        if DHM is None:
            self.DHM = DHM_DUMMY()
            if objective == "Zeiss 20x":
                self.DHM.pixel_pitch = [0.276, 0.276]  # 20x objective
                self.DHM.prop_dist = -20.0  # Distance of sensor to back focal plane of tube lens
                self.r0 = 304  # ORIGINAL: dhm.objective["dcRadius"]  # dc_radius: 304
            elif objective == "Zeiss 63x":
                self.DHM.pixel_pitch = [0.0869, 0.0869]  # 63x objective
                self.DHM.prop_dist = -20.0  # ToDo: Ändern - welche Einheit braucht man !
                self.r0 = 304  # noch nicht implementiert - weiß nicht wo reinhard das andere her hat # ToDo: Fragen wo das herkommt
            else:
                raise NotImplementedError(f"Objective {objective} not implemented!")
            self.DHM.wavelength = 0.000675  # wavelength of the laser
            self.DHM.n_resin = 1.5
        else:
            raise Warning('Please check if DHM has the necessary attributes!')

        self.__locate_order()
        if background_hologram is None:
            self.background_available = False
            self.background_hologram = None
            self.background_phase = None
            self.background_intensity = None
        else:
            self.background_available = True
            self.background_hologram = background_hologram
            self.calc_background(background_hologram)

    def calc_background(self, background_holo):
        self.background_available = True
        self.background_hologram = background_holo
        intensity, phase = self.reconstruct(holo=background_holo, z=self.DHM.prop_dist)
        phase_unwrapped = self.phase_unwrapping(phase)
        self.background_intensity = intensity
        self.background_phase = phase_unwrapped
        # ToDo für alle weiteren auch set methoden einführen
        # ToDo: überlegen wann man dann den background available setzt, immer nur wenn das ander (phase intensität) schon gesetzt wurde?

    def __locate_order(self):
        """
        Internal Methode.
        Background image is using the same position and mask!
        ToDo Maybe insert another method for a hologram as a function?
        """
        # ToDo: Check how good the locateOrder works with 63x objective
        spectrum, fx, fy, weight = locateOrder(holo=self.hologram)
        self.spectrum_not_shifted = spectrum
        self.first_diffraction_order_pos = [fx, fy]
        self._calc_radius_mask(fx, fy)

    def _propagation(self, field, z=None):
        if z is None:
            z = self.DHM.prop_dist

        dx = self.DHM.pixel_pitch[0]
        dy = self.DHM.pixel_pitch[1]
        wl = self.DHM.wavelength

        field = angularSpectrum(field, z=z, wavelength=wl, dx=dx, dy=dy)
        # ToDo: Kontrollieren wo der unterschied zwischen pixelpitch und dem pc(Reinahrds Variable) ist
        self.reconstructed_field = field
        return field

    def _calc_radius_mask(self, fx, fy):
        if self.r0 is None:
            raise Exception(f"Dc Radius not implemented for objective {self.objective}")
        h, w = self.hologram.shape
        rmax = np.sqrt(fx ** 2 + fy ** 2) - self.r0
        rmax = min(rmax, abs(fx), w // 2 - abs(fx), abs(fy), h // 2 - abs(fy))
        self.radius_mask = rmax
        return rmax

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

    def reconstruct(self, holo=None, z=None, pos_first_order=None, radius_of_mask=None,
                    return_field=False) -> "np.array":
        """
        Reconstruction of the hologram with the opportunity to tweak inputs. Returns intensity and phase by default.
        z Propagation Distance
            Returns the intensity and phase
        if return_field == True:
            Returns the field
        """
        if pos_first_order is not None:
            # sanity check - should never happen, nothing to reconstruct - pos_first_order is already given with internal value.
            assert holo is not None, "pos_first_order is already given with internal value."

            if isinstance(pos_first_order, int | float):
                fx = pos_first_order
                fy = pos_first_order
            if isinstance(pos_first_order, list) and len(pos_first_order) == 2:
                fx = pos_first_order[0]
                fy = pos_first_order[1]
            else:
                raise Exception(f"Check type of {pos_first_order}. Allowed are int, float and lists.")
        else:
            fx = self.first_diffraction_order_pos[0]
            fy = self.first_diffraction_order_pos[1]

        if holo is None:
            # Reconstruction for the object itself !
            self.spectrum_shifted = self.shift_spectrum(self.spectrum_not_shifted, fx, fy)
            self.spectrum_masked = self.masked_spectrum(self.spectrum_shifted, r=self.radius_mask)

            if z is None:  # No compensation
                self.reconstructed_field = getField(self.spectrum_masked)
            else:
                field = getField(self.spectrum_masked)
                self.reconstructed_field = self._propagation(field, z=z)
            field = self.reconstructed_field  # for returning the field later
        else:
            # Reconstruct for a different hologram (i.e. background hologram)
            # ToDO: Wie soll ich mit der fx,fy umgehen? Gibt es fälle bei denen ich das hier probleme gibt?

            spectrum = getSpectrum(holo.astype(np.float64))
            spectrum_shifted = self.shift_spectrum(spectrum, fx, fy)
            if radius_of_mask is not None:
                r = radius_of_mask
            else:
                r = self.radius_mask
            spectrum_masked = self.masked_spectrum(spectrum_shifted, r=r)
            field = getField(spectrum_masked)
            if z is not None:
                field = self._propagation(field, z=z)
            self.reconstructed_field = field

        if return_field:  # Returning just the field
            return field

        self.int_reconstructed = self.intensity(field)
        self.phase_reconstructed = self.phase(field)
        return self.int_reconstructed, self.phase_reconstructed

    def phase_unwrapping(self, phase_wrapped):
        """
        This phase unwrapping algorithm is based on:
        Miguel Arevallilo Herráez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path,"
        Appl. Opt. 41, 7437-7444 (2002)
        https://opg.optica.org/ao/abstract.cfm?URI=ao-41-35-7437
        """
        phase_unwrapped = unwrap_phase(phase_wrapped)
        return phase_unwrapped

    def compensate_background(self):
        assert self.background_available == True
        assert self.phase_unwrapped is not None
        self.phase_compensated = self.phase_unwrapped - self.background_phase
        return self.phase_compensated

    def run(self, z=None, compensation=False):
        """
        Returns unwrapped oder compensated phase
        """
        if z is None:
            int, phase = self.reconstruct(z=self.DHM.prop_dist)
        else:
            int, phase = self.reconstruct(z=z)
        phase = self.phase_unwrapping(phase)
        self.phase_unwrapped = phase
        # account for aberration
        if compensation:
            # check if background image is available
            assert self.background_available == True
            phase = self.compensate_background()

            height = self.phase_to_height(phase)
            self.height_profile = height

        return phase

    def phase_to_height(self, phase):
        """
        Height reconstruction based on the theoretical investigation by Nguyen et al.
        Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and Rola Aylo, "Accurate quantitative phase digital holographic microscopy with single- and multiple-wavelength telecentric and nontelecentric configurations,"
        Appl. Opt. 55, 5666-5683 (2016)
        http://dx.doi.org/10.1364/AO.55.005666
        """
        n_air = 1
        delta_n = n_air - self.DHM.n_resin  # change in refractive index - only approximate values
        height_profile = self.DHM.wavelength * phase / (2 * np.pi * delta_n)
        return height_profile

    def plotImage(self, img, title=None, save=False, cmap='viridis'):
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
                plt.imshow(img, cmap=cmap)
            else:
                plt.imshow(img, cmap=cmap)
                plt.title(title)
            plt.show()  # show image
        return

    def plot_height(self, height_profile, title=None, save=False, legend_bar=True, cmap='coolwarm'):
        """
        Plotting of the reconstructed height profile. Make sure the dimensions of the height profile matches the
        dimensions of the hologram.

        height_profile:  Height profile of the image
        title:           Title of the image. If save then this will also be the name of the saved image.
        save:            Boolean if it should be saved. If False then it will be shown.
        legend_bar:      Boolean if the color bar should be shown.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Using linspace to generate exactly 1024 points in each direction
        X = np.linspace(0, (self.hologram.shape[1] - 1) * self.DHM.pixel_pitch[1], self.hologram.shape[1])
        Y = np.linspace(0, (self.hologram.shape[0] - 1) * self.DHM.pixel_pitch[0], self.hologram.shape[0])
        # Creating the meshgrid
        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, height_profile, cmap=cmap,
                               linewidth=0, antialiased=False)
        if title is not None:
            plt.title(title)

        if legend_bar:
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            if title is not None:
                name = title.replace(" ", "_") + '.png'
                save_path = os.path.join(self.img_save_path, name)
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
            else:
                save_path = os.path.join(self.img_save_path, f"3D_plot_{self.var_4_saving}.png")
                plt.savefig(fname=save_path, dpi=400)  # transparent=True)
                self.var_4_saving += 1
        else:
            plt.show()

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

    def get_phase(self, compensation=False):
        if compensation:
            if self.background_available == False:
                raise Warning("No background for compensation!")
            return self.phase_compensated
        else:
            return self.phase_unwrapped

    def set_save_path(self, path):
        self.img_save_path = path
        # ToDo: Maybe do it in a more general fashion. One folder for saving all the things (maybe) and automatically
        #  determine a subfolder /img/ for the images - maybe done in the future for the complete structure class

    def evaluate(self, background_hologram=None, save_img=False, compensate=True):
        """
        All-in-one method for investigating a taken hologram.
        If save_img = True , then the plotted images will be safed to self.img_save_path
                                - set the path with set_save_path(path)
        """
        # ToDo Title der Auswertungen ändern.
        # ToDo save_img und savepath überarbeiten

        if save_img and self.img_save_path is None:
            raise Exception("A path for saving the images is needed! \nUse the method set_save_path for this purpose.")
        if background_hologram is None and compensate is True:
            assert self.background_available == True
        else:
            self.calc_background(background_hologram)

        self.run(compensation=compensate)

        # Plotting of spectrum - normal, shifted, masked (2 different r)
        self.plotImage(self.intensity(self.spectrum_not_shifted), "Intensity not shifted", save=save_img)
        self.plotImage(self.intensity(self.spectrum_shifted), "Intensity shifted", save=save_img)
        self.plotImage(self.intensity(self.spectrum_masked), "Intensity shifted and masked", save=save_img)
        # Plotting of field and phase after masking and iFFT
        # Plotting of field and phase after compensation
        self.plotImage(self.intensity(self.reconstructed_field), "intensity after prop", save=save_img)
        self.plotImage(self.phase(self.reconstructed_field), "phase after prop, before unwrapping", save=save_img)
        # Plotting of unwrapped images
        self.plotImage(self.phase_unwrapped, "Phase after unwrapping", save=save_img)
        if compensate:
            self.plotImage(self.phase_compensated, "Phase after compensation", save=save_img)
            self.plot_height(-self.height_profile, title="Height profile of the Structure", save=save_img)

        # ToDo Add variable for this evaluation
        # self.plotImage(self.intensity(self.reconstructed_field), "intensity before prop", save=save_img)
        # self.plotImage(self.phase(self.reconstructed_field), "phase before prop and unwrapping", save=save_img)
        # Plotting of unwrapped images BEFORE PROPAGATION
        # self.plotImage(phase_unwrapped_rmax_not_prop, "phase after unwrapping BEFORE prop (correct r)", save=save_img)
        # self.plotImage(phase_unwrapped_rcalc_not_prop, "phase after unwrapping BEFORE prop (false r)", save=save_img)

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    def debugging(self, save_img=False):
        spectrum = self.spectrum_not_shifted
        fx, fy = self.first_diffraction_order_pos
        r = self.radius_mask

        spectrum_shifted = self.shift_spectrum(spectrum, fx, fy)
        spectrum_masked = self.masked_spectrum(spectrum_shifted, r=r)

        field = getField(spectrum_masked)

        distance = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        # ToDo Propagation testing!
        for i in range(2):
            if i == 0:
                a = 1
            else:
                a = -1
            for j in range(len(distance)):
                z = a * distance[j]

                prop_field = self._propagation(field=field, z=z)

                phase_unwrapped_not_prop = self.phase_unwrapping(self.phase(field))
                phase_unwrapped = self.phase_unwrapping(self.phase(prop_field))
                self.phase_unwrapped = phase_unwrapped

                phase_compensated = self.compensate_background()
                height = self.phase_to_height(phase_compensated)

                self.plotImage(self.intensity(prop_field), f"Intensity with z={z}", save=save_img)
                self.plot_height(-height, title=f"Height profile with z={z}", save=save_img)

        '''
        # Plotting of spectrum - normal, shifted, masked (2 different r)
        self.plotImage(self.intensity(spectrum), "Intensity not shifted", save=save_img)
        self.plotImage(self.intensity(spectrum_shifted), "Intensity shifted", save=save_img)
        self.plotImage(self.intensity(spectrum_masked), "Intensity shifted and (correctly) masked", save=save_img)
        # Plotting of field and phase after masking and iFFT
        self.plotImage(self.intensity(field), "intensity before prop", save=save_img)
        self.plotImage(self.phase(field), "phase before prop and unwrapping", save=save_img)
        # Plotting of field and phase after compensation
        self.plotImage(self.intensity(prop_field), "intensity after prop", save=save_img)
        self.plotImage(self.phase(prop_field), "phase after prop, before unwrapping", save=save_img)
        # Plotting of unwrapped images BEFORE PROPAGATION
        # self.plotImage(phase_unwrapped_rmax_not_prop, "phase after unwrapping BEFORE prop (correct r)", save=save_img)
        # self.plotImage(phase_unwrapped_rcalc_not_prop, "phase after unwrapping BEFORE prop (false r)", save=save_img)
        # Plotting of unwrapped images
        self.plotImage(phase_unwrapped, "phase after prop, after unwrapping", save=save_img)
        '''


if __name__ == "__main__":
    # Testing the Reconstruction class with the evaluate-methode
    ''' JUST EDIT THE UPPER HALF '''
    ####################################################################
    ####################################################################

    user = "Hannes"
    objective = "Zeiss 20x"

    path = "C:/Users/hanne/Desktop/Test4DHMReconstruction"
    file_name = "dhm_stair_galvo_after"
    eval_folder = "DEBUGGING_" + file_name
    path_pic = os.path.join(path, file_name)
    background_path_pic = os.path.join(path, "dhm_stair_galvo_before")

    ####################################################################
    ####################################################################
    # check if saving path exists
    save_path = os.path.join(path, eval_folder)
    if not os.path.exists(save_path): os.mkdir(save_path)

    # loading the holograms
    holo = get_hologram(path=path_pic)
    background_holo = get_hologram(path=background_path_pic)

    # Initialize the Reconstruction object
    DHM_obj = Reconstruction(hologram=holo.data, background_hologram=background_holo.data, objective=objective)
    # setting saving path
    DHM_obj.set_save_path(path=save_path)
    # evaluate whole Hologram
    DHM_obj.debugging(save_img=True)
