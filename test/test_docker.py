import os
from OffAxisHolo.docker import DockerSciDataContainer

container_path = r"C:\Users\hanne\Documents\Projekte Offline PC\DHM as a QPI method\rawdata\DHM_Print\structures\lens1_ABZ_Zeiss 63x.zdc"
eval_path = os.path.join(os.getcwd(), "test", "docker")
os.makedirs(eval_path, exist_ok=True)

test_docker = DockerSciDataContainer()
test_docker.plot_reconstruction(container_path=container_path,
                                save_img=True, save_path_plot=eval_path,
                                save_data=True, save_path_data=eval_path, data_name="data_test_docker",
                                compensate=True, propagate=True,
                                cmap="viridis")
