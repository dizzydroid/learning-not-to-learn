#!/bin/bash
# Runs the Colored MNIST experiment.

# --- Configuration ---
PROJECT_ROOT="." # Assuming you run this script from the project root
DATA_ROOT_BASE="${PROJECT_ROOT}/data" # Base directory for all datasets
OUTPUT_DIR_BASE="${PROJECT_ROOT}/outputs"

SEED=42
BATCH_SIZE=128
NUM_WORKERS=4
EPOCHS=50           # Total epochs
PRETRAIN_EPOCHS=0   # Epochs to pretrain f and g without adversarial loss
LR_FG=0.001
LR_H=0.001
ADV_MU=1.0          # Weight for adversarial loss (mu)
ADV_LAMBDA=0.5      # Weight for entropy loss (lambda), set to 0 to disable
GRL_GAMMA=10.0
SAVE_EVERY=10

# ColoredMNIST specific
SIGMA_BIAS=0.05     # Std dev for color noise
CMNIST_TRAIN_BIAS_TYPE=True # Apply bias to training set
CMNIST_BIAS_LABEL_TYPE="mean_color_idx" # 'mean_color_idx' or 'quantized_sampled_color'
NUM_MAIN_CLASSES_CMNIST=10
NUM_BIAS_CLASSES_CMNIST=10 # For mean_color_idx

EXPERIMENT_NAME="cmnist_debias_sigma${SIGMA_BIAS}_mu${ADV_MU}_lambda${ADV_LAMBDA}"
# --- End Configuration ---

# Ensure output directory for this specific experiment exists
mkdir -p "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}/checkpoints"
mkdir -p "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}/logs"

echo "========================================================================"
echo "Starting Colored MNIST experiment: ${EXPERIMENT_NAME}"
echo "Project Root: $(pwd)" # Should be project root
echo "Data will be sourced from subdirectories within: ${DATA_ROOT_BASE}"
echo "Outputs will be in: ${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}"
echo "========================================================================"

python3 main.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_dir "${OUTPUT_DIR_BASE}" \
    --seed ${SEED} \
    --mode "train" \
    --use_cuda True \
    \
    --dataset_name "ColoredMNIST" \
    --data_root_base "${DATA_ROOT_BASE}" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --download_data True \
    --sigma_bias ${SIGMA_BIAS} \
    --cmnist_train_bias_type ${CMNIST_TRAIN_BIAS_TYPE} \
    --cmnist_bias_label_type "${CMNIST_BIAS_LABEL_TYPE}" \
    \
    --num_main_classes ${NUM_MAIN_CLASSES_CMNIST} \
    --num_bias_classes ${NUM_BIAS_CLASSES_CMNIST} \
    --f_network_name "SimpleCNN" \
    --g_network_name "SimpleCNN" \
    --h_network_name "SimpleCNN_ConvBias" \
    --input_channels 3 \
    --image_size 28 \
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
echo "Colored MNIST experiment finished."
echo "========================================================================"
