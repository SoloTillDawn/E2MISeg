import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

my_output_identifier = "e2miseg"
default_plans_identifier = "unetr_pp_Plansv2.1"
default_data_identifier = 'unetr_pp_Data_plans_v2.1'
default_trainer = "e2miseg_trainer_acdc"

base = os.environ['e2miseg_raw_data_base'] if "e2miseg_raw_data_base" in os.environ.keys() else None
preprocessing_output_dir = os.environ['e2miseg_preprocessed'] if "e2miseg_preprocessed" in os.environ.keys() else None
network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None


if base is not None:
    nnFormer_raw_data = join(base, "e2miseg_raw_data")
    nnFormer_cropped_data = join(base, "e2miseg_cropped_data")
    maybe_mkdir_p(nnFormer_raw_data)
    maybe_mkdir_p(nnFormer_cropped_data)
else:
    print("e2miseg_raw_data_base error ")
    nnFormer_cropped_data = nnFormer_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("e2miseg_preprocessed error ")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER errpr")
    network_training_output_dir = None
