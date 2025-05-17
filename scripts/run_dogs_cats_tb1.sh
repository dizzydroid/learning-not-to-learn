#!/bin/bash
# Runs the Dogs and Cats experiment with TB1 bias (Bright dogs, Dark cats).

# --- Configuration ---
PROJECT_ROOT="."
DATA_ROOT_BASE="${PROJECT_ROOT}/data"
OUTPUT_DIR_BASE="${PROJECT_ROOT}/outputs"

SEED=42
BATCH_SIZE=32
NUM_WORKERS=4
EPOCHS=100
PRETRAIN_EPOCHS=5 # Example: pretrain for 5 epochs
LR_FG=0.0001
LR_H=0.0001
ADV_MU=1.0
ADV_LAMBDA=0.5
GRL_GAMMA=10.0
SAVE_EVERY=10
IMAGE_SIZE_DAC=224

# DogsAndCats specific
DAC_LIST_BRIGHT="list_bright.txt" # Expected in ./data/dogs_vs_cats/lists/
DAC_LIST_DARK="list_dark.txt"     # Expected in ./data/dogs_vs_cats/lists/
DAC_LIST_TEST="list_test_unbiased.txt" # Expected in ./data/dogs_vs_cats/lists/
BIAS_TYPE_TRAIN_DAC="TB1" # "TB1" or "TB2"
NUM_MAIN_CLASSES_DAC=2 # Dog vs Cat
NUM_BIAS_CLASSES_DAC=2 # Bright vs Dark

EXPERIMENT_NAME="dac_debias_${BIAS_TYPE_TRAIN_DAC}_mu${ADV_MU}_lambda${ADV_LAMBDA}"
# --- End Configuration ---

mkdir -p "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}/checkpoints"
mkdir -p "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}/logs"

echo "========================================================================"
echo "Starting Dogs and Cats (${BIAS_TYPE_TRAIN_DAC}) experiment: ${EXPERIMENT_NAME}"
echo "Project Root: $(pwd)"
echo "Data will be sourced from subdirectories within: ${DATA_ROOT_BASE}"
echo "  (Ensure images are in '${DATA_ROOT_BASE}/dogs_vs_cats/images/' and lists in '${DATA_ROOT_BASE}/dogs_vs_cats/lists/')"
echo "Outputs will be in: ${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}"
echo "========================================================================"

python3 main.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_dir "${OUTPUT_DIR_BASE}" \
    --seed ${SEED} \
    --mode "train" \
    --use_cuda True \
    \
    --dataset_name "DogsAndCats" \
    --data_root_base "${DATA_ROOT_BASE}" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --image_size ${IMAGE_SIZE_DAC} \
    --input_channels 3 \
    --dac_list_bright "${DAC_LIST_BRIGHT}" \
    --dac_list_dark "${DAC_LIST_DARK}" \
    --dac_list_test "${DAC_LIST_TEST}" \
    --bias_type_train "${BIAS_TYPE_TRAIN_DAC}" \
    \
    --num_main_classes ${NUM_MAIN_CLASSES_DAC} \
    --num_bias_classes ${NUM_BIAS_CLASSES_DAC} \
    --f_network_name "ResNet18" \
    --f_pretrained True \
    --g_network_name "ResNet18" \
    --h_network_name "ResNet18_FCBias" \
    \
    --epochs ${EPOCHS} \
    --pretrain_fg_epochs ${PRETRAIN_EPOCHS} \
    --lr_fg ${LR_FG} \
    --lr_h ${LR_H} \
    --adv_mu ${ADV_MU} \
    --adv_lambda ${ADV_LAMBDA} \
    --grl_gamma ${GRL_GAMMA} \
    --save_every_epochs ${SAVE_EVERY} \
    --do_eval True

echo "========================================================================"
echo "Dogs and Cats (${BIAS_TYPE_TRAIN_DAC}) experiment finished."
echo "========================================================================"
