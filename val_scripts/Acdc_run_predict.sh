#!/bin/sh

DATASET_PATH=../DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=../e2miseg/evaluation/e2miseg_acdc_checkpoint
export e2miseg_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export e2miseg_raw_data_base="$DATASET_PATH"/unetr_pp_raw


python ../e2miseg/run/run_training.py 3d_fullres e2miseg_trainer_acdc 1 0 --gpu_ids 1 -val