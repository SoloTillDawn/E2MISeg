import numpy as np

from e2miseg.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from e2miseg.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from e2miseg.paths import *


class ExperimentPlanner3D_v21_customTargetSpacing_2x2x2(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnUNetData_plans_v2.1_trgSp_2x2x2"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_trgSp_2x2x2_plans_3D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([2., 2., 2.]) # make sure this is float!!!! Not int!

class ExperimentPlanner2D_v21_customTargetSpacing_2x2x2(ExperimentPlanner2D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnFormerData_plans_v2.1_2D_trgSp_2x2x2"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnFormerPlansv2.1_trgSp_2x2x2_plans_2D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([2., 2., 2.])  # make sure this is float!!!! Not int!

