import fnmatch
import os

# from evaluate import evaluate_structure
# from test_propagation import test_propagation_distance as evaluate_structure
from test_propagation import problem_solver as evaluate_structure
from container.collector import ExperimentCollector

exp_path = (r"C:\Users\hanne\Documents\Seafile\Nanoproduction_Hannes\Code\NanoFactorySystem\mains\.output\dhm_paper\FINAL_print_20241029_Zeiss 63x")
s_path = (r"C:\Users\hanne\Desktop\DHM Propagationstest")

# experiment = ExperimentCollector(root_path=exp_path)
# experiment.collect()

###

structure_path = os.path.join(exp_path, "structures")
structure_content = os.listdir(structure_path)

for i in range(len(structure_content)):
    # skipping corner and qr code evaluation
    if fnmatch.fnmatch(structure_content[i], "corner*") or fnmatch.fnmatch(structure_content[i], "qr*"):
        continue

    # make sure only the container files are used
    if fnmatch.fnmatch(structure_content[i], "*.zdc"):
        # for purpose of testing it - using only one structure
        if fnmatch.fnmatch(structure_content[i], "DOE1*") or fnmatch.fnmatch(structure_content[i], "lens1*"):
            print(f"Struktur {structure_content[i]} wird berechnet.")
            structure_container_path = os.path.join(structure_path, structure_content[i])
            speicherung = os.path.join(s_path, structure_content[i])
            if not os.path.exists(speicherung):
                os.makedirs(speicherung)
            evaluate_structure(structure_container_path, speicherung)
            print(f"Struktur {structure_content[i]} ist beendet.")
