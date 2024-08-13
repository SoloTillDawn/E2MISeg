from copy import deepcopy

import numpy as np
from e2miseg.experiment_planning.common_utils import get_pool_and_conv_props
from e2miseg.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from e2miseg.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from e2miseg.network_architecture.generic_UNet import Generic_UNet
from e2miseg.paths import *


class ExperimentPlanner3D_v21_3cps(ExperimentPlanner3D_v21):
    """
    have 3x conv-in-lrelu per resolution instead of 2 while remaining in the same memory budget

    This only works with 3d fullres because we use the same data as ExperimentPlanner3D_v21. Lowres would require to
    rerun preprocesing (different patch size = different 3d lowres target spacing)
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21_3cps, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnFormerPlansv2.1_3cps_plans_3D.pkl")
        self.unet_base_num_features = 32
        self.conv_per_stage = 3

    def run_preprocessing(self, num_threads):
        pass
