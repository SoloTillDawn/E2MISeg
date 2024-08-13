from e2miseg.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from e2miseg.paths import *


class ExperimentPlanner2D_v21_RGB_scaleTo_0_1(ExperimentPlanner2D_v21):
    """
    used by tutorial e2miseg.tutorials.custom_preprocessing
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnFormer_RGB_scaleTo_0_1"
        self.plans_fname = join(self.preprocessed_output_folder, "nnFormer_RGB_scaleTo_0_1" + "_plans_2D.pkl")

        self.preprocessor_name = 'GenericPreprocessor_scale_uint8_to_0_1'
