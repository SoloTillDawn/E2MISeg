from collections import OrderedDict

from e2miseg.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from e2miseg.paths import *


class ExperimentPlannernonCT(ExperimentPlanner):
    """
    Preprocesses all data in nonCT mode (this is what we use for MRI per default, but here it is applied to CT images
    as well)
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannernonCT, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnFormer_nonCT"
        self.plans_fname = join(self.preprocessed_output_folder, "nnFormerPlans" + "nonCT_plans_3D.pkl")

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "nonCT"
            else:
                schemes[i] = "nonCT"
        return schemes

