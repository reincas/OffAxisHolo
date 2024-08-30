import fnmatch
import os

from offaxisholo.Holo_class import Reconstruction
from offaxisholo.utils import get_hologram
from offaxisholo.filtering import hybrid_median_mean_filter as hm2f


def test_evalutate():
    # Testing the Reconstruction class with the evaluate-methode
    ''' JUST EDIT THE UPPER HALF '''
    ####################################################################
    ####################################################################

    user = "Hannes"
    objective = "Zeiss 20x"

    path = "C:/Users/hanne/Desktop/Test4DHMReconstruction"
    file_name = "dhm_lens_galvo_before"
    eval_folder = "Evaluation_" + "Background_images"
    path_pic = os.path.join(path, file_name)
    background_path_pic = os.path.join(path, "dhm_lens_galvo_before")

    ####################################################################
    ####################################################################
    # check if saving path exists
    save_path = os.path.join(path, eval_folder)
    if not os.path.exists(save_path): os.mkdir(save_path)

    # loading the holograms
    holo = get_hologram(path=path_pic)
    # background_holo = get_hologram(path=background_path_pic)

    # Initialize the Reconstruction object
    DHM_obj = Reconstruction(hologram=holo.data, objective=objective)
    # setting saving path
    DHM_obj.set_save_path(path=save_path)
    # evaluate whole Hologram
    DHM_obj.evaluate(save_img=True, compensate=False)


def main():
    ####################################################################
    ####################################################################

    user = "Hannes"
    objective = "Zeiss 20x"

    path = "C:/Users/hanne/Desktop/Test4DHMReconstruction"
    file_name = "dhm_stair_galvo_after"
    eval_folder = "Filtering_eval" + "stair"
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
    DHM_obj.run(compensation=True)
    image = DHM_obj.phase_compensated
    filtered_image = hm2f(image, kernel_size=5)

    DHM_obj.plotImage(image, cmap='coolwarm')
    DHM_obj.plotImage(filtered_image, cmap='coolwarm')

    height_normal = DHM_obj.phase_to_height(-image)
    height_filtered = DHM_obj.phase_to_height(-filtered_image)

    DHM_obj.plot_height(height_normal)
    DHM_obj.plot_height(height_filtered)

    diff = height_filtered - height_normal

    DHM_obj.plotImage(diff, cmap='flag')


if __name__ == '__main__':
    main()
