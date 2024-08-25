import fnmatch
import os

from offaxisholo.Holo_class import Reconstruction
from offaxisholo.utils import get_hologram


def main():
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


if __name__ == '__main__':
    main()