#!/bin/sh

DATASET_PATH=../DATASET_Mcl

export PYTHONPATH=.././
export RESULTS_FOLDER=../e2miseg/evaluation/e2miseg_mcl_checkpoint
export e2miseg_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task16_Mcl
export e2miseg_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../unetr_pp/inference/predict.py -i ..../Task016_Mcl/imagesTs/ -o ..../Task016_Mcl/inferTs -m ../Task016_Mcl/e2miseg_trainer_mcl__unetr_pp_Plansv2.1 -f 0
python ../unetr_pp/inference_mcl.py 0
