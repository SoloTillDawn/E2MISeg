#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../e2miseg/evaluation/e2miseg_tumor_checkpoint
export e2miseg_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export e2miseg_raw_data_base="$DATASET_PATH"/unetr_pp_raw


python ../e2miseg/run/run_training.py 3d_fullres e2miseg_trainer_tumor 3 0  --gpu_ids 1 -val
python ../e2miseg/inference_tumor.py 0

