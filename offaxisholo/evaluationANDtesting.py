import fnmatch
import os

from offaxisholo.Holo_class import Reconstruction
from offaxisholo.utils import get_hologram

dhm_test_list = []
root_directory = ("C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/test/devices/.test"
                  "/objective_test")
objective = ["Zeiss 20x", "Zeiss 63x"]

for i in range(len(objective)):
    path = os.path.join(root_directory, objective[i])
    dhm_test_list = fnmatch.filter(os.listdir(path), '*.zdc')

    for j in range(len(dhm_test_list)):
        holo_path = os.path.join(path, dhm_test_list[j])

        save_dir = holo_path[:-4]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        holo = get_hologram(path=holo_path)
        DHM_obj = Reconstruction(hologram=holo, objective=objective[i])
        DHM_obj.set_save_path(path=save_dir)
        DHM_obj.evaluation(save_img=True)
