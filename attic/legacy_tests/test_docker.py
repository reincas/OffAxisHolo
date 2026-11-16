import os

import numpy as np

from src.offaxisholo.pipeline.reconstruction_pipeline import DockerImageFile

container_path = (
    r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\rawdata"
    r"\DHM_Print\structures\DOE1_ABZ_Zeiss 63x.zdc"
)
image_path = (
    r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method"
    r"\rawdata\DHM_Developed\lens1.tif"
)
back_img_path = (
    r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\rawdata"
    r"\DHM_Developed\lens_background.tif"
)
# container_path = r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\
# rawdata\DHM_Print\structures\lens1_ABZ_Zeiss 63x.zdc"
eval_path = os.path.join(os.getcwd(), "test", "docker")
eval_path_img = os.path.join(os.getcwd(), "test", "docker_image")
os.makedirs(eval_path, exist_ok=True)

# test_docker = DockerSciDataContainer()
# test_docker.plot_reconstruction(container_path=container_path,
#                                 save_img=True, save_path_plot=eval_path,
#                                 save_data=True, save_path_data=eval_path,
#                                 data_name="data_test_docker",
#                                 compensate=True, propagate=True,
#                                 cmap="viridis")
test_docker = DockerImageFile()
a = np.linspace(start=-1e-6, stop=1e-6, num=10)
for i in range(10):
    length = a[i]
    print(length)
    test_docker.reconstruct(
        img_path=image_path,
        background_img_path=back_img_path,
        save_data=True,
        save_path=eval_path_img,
        save_name=f"data_test_docker_{length}",
        propagate=True,
        prop_dist=length,
    )

# length=0.0
# test_docker.reconstruct(img_path=image_path, background_img_path=back_img_path,
#                         save_data=True, save_path=eval_path_img,
#                         save_name=f"data_test_docker_{0}",
#                         propagate=True, prop_dist=length)
