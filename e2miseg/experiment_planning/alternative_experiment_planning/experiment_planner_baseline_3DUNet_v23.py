from e2miseg.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from e2miseg.paths import *


class ExperimentPlanner3D_v23(ExperimentPlanner3D_v21):
    """
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v23, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnFormerData_plans_v2.3"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnFormerPlansv2.3_plans_3D.pkl")
        self.preprocessor_name = "Preprocessor3DDifferentResampling"
