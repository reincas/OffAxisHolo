from Auswertung.container import ExperimentCollector

PATH = r"C:\Users\hanne\Documents\Seafile\Nanoproduction_Hannes\Code\NanoFactorySystem\mains\.output\dhm_paper\FINAL_print_20241029_Zeiss 63x"

if __name__ == "__main__":
    experiment = ExperimentCollector(root_path=PATH)
    experiment.collect()
