import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scidatacontainer import Container

from Auswertung.container import StructureContainer
from Auswertung.container import update_container

from offaxisholo import Hologram, HologramReconstructor, ReferenceHologram, HologramPostProcessor

"""
Ergebnis Propagationstest:
1. wellenlänge ist falsch hinterlegt in den Daten! 6.749e-6 anstatt 0.6749e-6
2. Es scheint als wäre das Hologram bereits in der Focusebene.
3. änderung der Wurzel von numpy.sqrt zu einer funktion die complexe wurzeln benutzen kann
4. bei einer propagationsdistanz von z<=1.4e-6 führt es dazu, dass es zu unendlich wird und damit nicht nutzbar.
5. negative propagationsdistanzen sind nicht sinnvoll - keine nutzbaren ergebnisse.
6. positive propagationsdistanzen werden anscheinend korrekt berechnet, allerdings wird es immer unschärfer mit 
    steigender propagationsdistanz
7. es ist egal ob erst propagiert und dann kompensiert wird oder anders herum

Die genutzen Extrafunktionen (z.B. run_comp_then_prop) kann man am ende dieses Scriptes auskommentiert finden.
"""


def test_propagation_distance(structure_path):
    if structure_path.endswith(".zdc"):
        data_container = StructureContainer(file=structure_path)
    else:
        path = structure_path + ".zdc"
        data_container = Container(file=path)

    tmp_dict = {}
    params = data_container.dhm_params
    background_hologram = data_container.background_hologram
    finished_structure = data_container.complete_hologram

    structure_complete = Hologram(data=finished_structure, dhm_parameter=params)
    pos = structure_complete.first_diffraction_order_pos
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=pos,
                                   dhm_parameter=params)

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete, background)

    # testing the propagation distance with a loop
    print("starting propagation search...")
    prop_dist = np.arange(0, 5e-3, 10e-6)    # maximum 0.5 cm , step 10µm
    prop_dist2 = np.arange(5e-3, 10e-3, 10e-6)    # maximum 1 cm , step 10µm
    prop_dist3 = np.arange(10e-3, 15e-3, 10e-6)    # maximum 1.5 cm , step 10µm
    prop_dist4 = np.arange(15e-3, 20e-3, 10e-6)    # maximum 2 cm , step 10µm
    field_img_liste, prop_dist_liste = reconstructor.run(prop_dist=prop_dist)
    field_img_liste2, prop_dist_liste2 = reconstructor.run(prop_dist=prop_dist2)
    field_img_liste3, prop_dist_liste3 = reconstructor.run(prop_dist=prop_dist3)
    field_img_liste4, prop_dist_liste4 = reconstructor.run(prop_dist=prop_dist4)
    for i in range(len(field_img_liste)):
        print(f"updating dictionary - step {i}/{len(field_img_liste)}")
        tmp_dict.update({f"eval/propagation_test/distance_{prop_dist_liste[i]}_m.png": field_img_liste[i]})
    # update container after complete reconstruction
    update_container(structure_path, tmp_dict)

    for i in range(len(field_img_liste2)):
        print(f"updating dictionary - step {i}/{len(field_img_liste2)}")
        tmp_dict.update({f"eval/propagation_test/distance_{prop_dist_liste2[i]}_m.png": field_img_liste2[i]})
    # update container after complete reconstruction
    update_container(structure_path, tmp_dict)

    for i in range(len(field_img_liste3)):
        print(f"updating dictionary - step {i}/{len(field_img_liste3)}")
        tmp_dict.update({f"eval/propagation_test/distance_{prop_dist_liste3[i]}_m.png": field_img_liste3[i]})
    # update container after complete reconstruction
    update_container(structure_path, tmp_dict)

    for i in range(len(field_img_liste4)):
        print(f"updating dictionary - step {i}/{len(field_img_liste4)}")
        tmp_dict.update({f"eval/propagation_test/distance_{prop_dist_liste4[i]}_m.png": field_img_liste4[i]})
    # update container after complete reconstruction
    update_container(structure_path, tmp_dict)


def test_prop_methode(structure_path):
    if structure_path.endswith(".zdc"):
        data_container = StructureContainer(file=structure_path)
    else:
        path = structure_path + ".zdc"
        data_container = Container(file=path)

    save1=os.path.join(structure_path[:-4])
    if not os.path.exists(save1):
        os.mkdir(save1)

    save_int=os.path.join(structure_path[:-4], "intensity")
    if not os.path.exists(save_int):
        os.mkdir(save_int)
    save_phase=os.path.join(structure_path[:-4], "phase")
    if not os.path.exists(save_phase):
        os.mkdir(save_phase)

    tmp_dict = {}
    params = data_container.dhm_params
    background_hologram = data_container.background_hologram
    finished_structure = data_container.complete_hologram

    structure_complete = Hologram(data=finished_structure, dhm_parameter=params)
    pos = structure_complete.first_diffraction_order_pos
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=pos,
                                   dhm_parameter=params)

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete, background)

    # testing the propagation distance with a loop
    prop_dist = np.arange(-5e-5, 5e-5, 1e-6)    # maximum 0.5 mm , step 1µm
    print("starting propagation search...")
    field_comp_prop, prop_dist_liste = reconstructor.run_comp_then_prop(prop_dist=prop_dist)
    for i in range(len(field_comp_prop)):
        print(f"start saving {prop_dist_liste[i]}----")
        intensity_comp_prop = structure_complete.intensity(field_comp_prop[i])
        phase_comp_prop = structure_complete.phase(field_comp_prop[i])

        if np.isnan(intensity_comp_prop).any() or np.isnan(phase_comp_prop).any():
            print(f"is not a number: {i}")
        save_img = save_int+f"/{i}_{prop_dist_liste[i]}.png"
        plt.imsave(save_img, intensity_comp_prop)
        save_img = save_phase+f"/{i}_{prop_dist_liste[i]}.png"
        plt.imsave(save_img, phase_comp_prop)

    # update container after complete reconstruction
    # update_container(structure_path, tmp_dict)


def test_recon_pic(pic_path,structure_path):
    data_container = StructureContainer(file=structure_path)
    save1=os.path.join(pic_path[:-4])
    if not os.path.exists(save1):
        os.mkdir(save1)

    save_int=os.path.join(save1, "intensity")
    if not os.path.exists(save_int):
        os.mkdir(save_int)
    save_phase=os.path.join(save1, "phase")
    if not os.path.exists(save_phase):
        os.mkdir(save_phase)

    tmp_dict = {}
    params = data_container.dhm_params
    finished_structure = plt.imread(pic_path)
    img = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
    # Convert to grayscale
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f'dtype: {g_img.dtype}, shape: {g_img.shape}, min: {np.min(g_img)}, max: {np.max(g_img)}')

    structure_complete = Hologram(data=g_img, dhm_parameter=params)

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete)

    # testing the propagation distance with a loop
    prop_dist = np.arange(-5e-5, 5e-5, 1e-6)    # maximum 0.5 mm , step 1µm
    print("starting propagation search...")
    field_comp_prop, prop_dist_liste = reconstructor.run_comp_then_prop(prop_dist=prop_dist, compensate=False)
    for i in range(len(field_comp_prop)):
        print(f"start saving {prop_dist_liste[i]}----")
        intensity_comp_prop = structure_complete.intensity(field_comp_prop[i])
        phase_comp_prop = structure_complete.phase(field_comp_prop[i])

        if np.isnan(intensity_comp_prop).any() or np.isnan(phase_comp_prop).any():
            print(f"is not a number: {i}")
        save_img = save_int+f"/{i}_{prop_dist_liste[i]}.png"
        plt.imsave(save_img, intensity_comp_prop)
        save_img = save_phase+f"/{i}_{prop_dist_liste[i]}.png"
        plt.imsave(save_img, phase_comp_prop)

    # update container after complete reconstruction
    # update_container(structure_path, tmp_dict)


def external_reconstruction_method(hologram, hologram_background):
    """
    Um die übersicht zu behalten
    Was used for an easy implementation of the current code. additional todos
    """
    # for a later investigation/ usage
    # processor = HologramPostProcessor(hologram)
    processor = None
    reconstructor = HologramReconstructor(hologram=hologram, reference=hologram_background, processor=processor)

    phase = reconstructor.run()
    intensity = hologram.propagated_intensity

    # Todo liste
    # DONE_todo 1 - dhm_parameter überall als dict ersetzten und die parameter dann mit einer neuen funktion in die variablen speichern
    # DONE_todo 2 - reconstruction einbauen in code oben
    # todo 3 - Rekonstruktion mittels simulation untersuchen
    # todo 4 - investigate Filtering methods and reconstruction data

    return intensity, phase


def hologram_3d_modelling(structure_path):
    # 1. get all the reconstructed phase information
    # 2. calculation
    # 3. savind in eval/reconstruction/3D_model
    # have fun
    print("3D modelling not yet implemented")
    pass


def problem_solver(structure_path, save_path):
    if structure_path.endswith(".zdc"):
        data_container = StructureContainer(file=structure_path)
    else:
        path = structure_path + ".zdc"
        data_container = Container(file=path)

    tmp_dict = {}
    params = data_container.dhm_params
    background_hologram = data_container.background_hologram
    finished_structure = data_container.complete_hologram

    structure_complete = Hologram(data=finished_structure, dhm_parameter=params)
    pos = structure_complete.first_diffraction_order_pos
    background = ReferenceHologram(data=background_hologram,
                                   first_diffraction_order_pos=pos,
                                   dhm_parameter=params)

    # Reconstruction of complete structure
    reconstructor = HologramReconstructor(structure_complete, background)

    reconstructor.set_save_path(save_path)
    reconstructor.evaluate(propagate=True, save_img=True)


# Funktionen die für testzwecke geschrieben wurden und nun nicht mehr gebraucht werden
"""

    def run_comp_then_prop(self, hologram: Hologram = None, background_hologram: ReferenceHologram = None, prop_dist=None,
            propagate=True,
            compensate=True) -> np.ndarray | tuple[Any, Any]:
        '''Full reconstruction pipeline including filtering and compensation.'''
        if hologram is None:
            hologram = self.hologram
        if background_hologram is None:
            if self.background_available:
                background_hologram = self.background
        if prop_dist is None:
            prop_dist = self.propagation_distance

        # Reconstruction of Hologram
        holo_field = hologram.reconstruct()
        holo_phase = self.phase_unwrapping(hologram.phase(holo_field))

        # Reconstruction of Background
        if background_hologram is not None:
            background_field = background_hologram.reconstruct()
            background_phase = self.phase_unwrapping(hologram.phase(background_field))

        # Aberration Compensation of Optics with Background image
        if compensate:
            phase = self.compensate(original=holo_phase, reference=background_phase)
            field = self.compensate(original=holo_field, reference=background_field)
            self.phase_compensated = phase
        else:
            phase = holo_phase
            field = holo_field

        # Propagation of the electrical field to the focal plane
        if propagate:
            if isinstance(prop_dist, list) or isinstance(prop_dist, np.ndarray):
                prop_list_field = []  # field after propagation
                prop_dist_list = []  # propagation distance
                for i in range(len(prop_dist)):
                    print(f"propagation search nr.{i} with distance {prop_dist[i]}")
                    field_propagated = self.propagate(field=field, distance=prop_dist[i])
                    prop_list_field.append(field_propagated)
                    prop_dist_list.append(prop_dist[i])

                return prop_list_field, prop_dist_list
            else:
                field_propagated = self.propagate(field=field, distance=prop_dist)

        # hologram.set_full_reconstruction(re_field=field, propagated_field=field_propagated if propagate else None,
        #                                  phase_unwrapped=holo_phase)
        # Filering of the phase
        # ToDo: Filtering needs rework or postprocessor needs rework
        # filtered = self.processor.filter(compensated)
        # self.height_profile = self.phase_to_height(filtered)
        # return filtered
        return


    def angularSpectrum(self, field, z, wavelength, dx, dy):
        '''
        # Function to diffract a complex field using the angular spectrum approximation
        # Inputs:
        # field - complex field
        # z - propagation distance [unit like wavelength]
        # wavelength - wavelength [unit like propagation distance]
        # dx, dy - pixel pitch
        '''
        field = np.array(field)

        if float(z) == 0.0:
            return field

        # Sanity check
        assert len(field.shape) == 2, "2D hologram image required!"
        M, N = field.shape

        # Spatial frequency coordinates
        fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(M, d=dy))
        FX, FY = np.meshgrid(fx, fy)

        root = (2 * np.pi) ** 2 * ((1. / wavelength) ** 2 - FX ** 2 - FY ** 2)

        # Calculate the propagating and the evanescent (complex) modes
        kz = sqrt(root)

        # Compute the transfer function (Angular Spectrum)
        H = np.exp(1j * kz * z)

        # Fourier transform of the input field
        spectrum = self.hologram.getSpectrum(field)

        # Multiply by transfer function in frequency domain
        propagated_ft = spectrum * H

        # Inverse Fourier transform to get back to spatial domain
        propagated_field = self.hologram.getField(propagated_ft)

        return propagated_field

    def angular_spectrum_propagation(self, field, z, wavelength, dx, dy):
        '''
        Propagate a complex optical field using the Angular Spectrum Method.

        Parameters:
        -----------
        field : 2D numpy array (complex)
            Input complex field amplitude
        z : float
            Propagation distance [same unit as wavelength]
        wavelength : float
            Wavelength of light [same unit as z]
        dx, dy : float
            Pixel pitch in x and y directions [same unit as wavelength]

        Returns:
        --------
        propagated_field : 2D numpy array (complex)
            Propagated complex field amplitude
        '''
        if z == 0:
            return field

        # Verify input is 2D
        if len(field.shape) != 2:
            raise ValueError("Input field must be 2D!")

        M, N = field.shape
        k = 2 * np.pi / wavelength  # wavenumber

        # Spatial frequencies
        fx = np.fft.fftfreq(N, dx)
        fy = np.fft.fftfreq(M, dy)
        FX, FY = np.meshgrid(fx, fy)

        # Calculate kz component of wavevector
        # Note: we use broadcasting to avoid explicit meshgrid
        kz = sqrt(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2 + 0j)

        # Transfer function
        H = np.exp(1j * kz * z)

        # Apply bandlimit to avoid aliasing
        # Maximum allowed spatial frequency based on sampling
        fx_max = 1 / (2 * dx)
        fy_max = 1 / (2 * dy)

        # Create frequency filter
        freq_mask = (np.abs(FX) <= fx_max * 0.9) & (np.abs(FY) <= fy_max * 0.9)
        H *= freq_mask

        # Propagate field
        spectrum = np.fft.fft2(field)
        propagated_spectrum = spectrum * np.fft.fftshift(H)
        propagated_field = np.fft.ifft2(propagated_spectrum)

        return propagated_field

    def run_prop_then_comp(self, hologram: Hologram = None, background_hologram: ReferenceHologram = None, prop_dist=None,
            propagate=True,
            compensate=True) -> np.ndarray | tuple[Any, Any]:
        '''Full reconstruction pipeline including filtering and compensation.'''
        if hologram is None:
            hologram = self.hologram
        if background_hologram is None:
            background_hologram = self.background
        if prop_dist is None:
            prop_dist = self.propagation_distance

        # Reconstruction of Hologram
        holo_field = hologram.reconstruct()
        holo_phase = self.phase_unwrapping(hologram.phase(holo_field))

        # Reconstruction of Background
        background_field = background_hologram.reconstruct()
        background_phase = self.phase_unwrapping(hologram.phase(background_field))

        # Propagation of the electrical field to the focal plane
        if propagate:
            if isinstance(prop_dist, list) or isinstance(prop_dist, np.ndarray):
                holo_field_prop = []  # field after propagation hologram
                back_field_prop = []  # field after propagation background
                prop_dist_list = []  # propagation distance
                for i in range(len(prop_dist)):
                    print(f"propagation search nr.{i} with distance {prop_dist[i]}")
                    holo_propagated = self.propagate(field=holo_field, distance=prop_dist[i])
                    back_propagated = self.propagate(field=background_field, distance=prop_dist[i])
                    holo_field_prop.append(holo_propagated)
                    back_field_prop.append(back_propagated)
                    prop_dist_list.append(prop_dist[i])

        # Aberration Compensation of Optics with Background image
        if compensate:
            field_compensated = np.asarray(holo_field_prop) - np.asarray(back_field_prop)
            return field_compensated, prop_dist_list
        else:
            phase = holo_phase
            field = holo_field

        # hologram.set_full_reconstruction(re_field=field, propagated_field=field_propagated if propagate else None,
        #                                  phase_unwrapped=holo_phase)
        # Filering of the phase
        # ToDo: Filtering needs rework or postprocessor needs rework
        # filtered = self.processor.filter(compensated)
        # self.height_profile = self.phase_to_height(filtered)
        # return filtered
        return
"""