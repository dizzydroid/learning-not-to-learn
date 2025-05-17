#!/bin/bash

# scripts/run_imdb_eb1_gender.sh
# Runs IMDB-Face, EB1 bias (Women 0-29, Men 40+), Gender Classification Task.

# --- Configuration ---
PROJECT_ROOT="."
DATA_ROOT_BASE="${PROJECT_ROOT}/data"
OUTPUT_DIR_BASE="${PROJECT_ROOT}/outputs"

SEED=42
BATCH_SIZE=32
NUM_WORKERS=4
EPOCHS=100
PRETRAIN_EPOCHS=10 # Example: pretrain for 10 epochs
LR_FG=0.0001
LR_H=0.0001
ADV_MU=1.0
ADV_LAMBDA=0.5
GRL_GAMMA=10.0
SAVE_EVERY=10
IMAGE_SIZE_IMDB=224

# IMDB-Face specific
# These are filenames, expected in ./data/imdb_face/manifests/
IMDB_TRAIN_MANIFEST="train_eb1_gender.csv" # You need to create this manifest
IMDB_TEST_MANIFEST="test_unbiased_gender.csv"   # You need to create this manifest
IMDB_TASK="classify_gender" # "classify_gender" or "classify_age"
BIAS_TYPE_TRAIN_IMDB="EB1"  # "EB1" or "EB2"

# For classify_gender task:
NUM_MAIN_CLASSES_IMDB=2 # Gender (Female/Male)
NUM_BIAS_CLASSES_IMDB=2 # Age group (Young/Old)

EXPERIMENT_NAME="imdb_${IMDB_TASK}_${BIAS_TYPE_TRAIN_IMDB}_mu${ADV_MU}_lambda${ADV_LAMBDA}"
# --- End Configuration ---

mkdir -p "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}/checkpoints"
mkdir -p "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}/logs"

echo "========================================================================"
echo "Starting IMDB-Face (${IMDB_TASK}, ${BIAS_TYPE_TRAIN_IMDB}) experiment: ${EXPERIMENT_NAME}"
echo "Project Root: $(pwd)"
echo "Data will be sourced from subdirectories within: ${DATA_ROOT_BASE}"
echo "  (Ensure filtered images are in '${DATA_ROOT_BASE}/imdb_face/filtered_images/')"
echo "  (Ensure manifests are in '${DATA_ROOT_BASE}/imdb_face/manifests/')"
echo "Outputs will be in: ${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}"
echo "========================================================================"

python3 main.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_dir "${OUTPUT_DIR_BASE}" \
    --seed ${SEED} \
    --mode "train" \
    --use_cuda True \
    \
    --dataset_name "IMDBFace" \
    --data_root_base "${DATA_ROOT_BASE}" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --image_size ${IMAGE_SIZE_IMDB} \
    --input_channels 3 \
    --train_manifest "${IMDB_TRAIN_MANIFEST}" \
    --test_manifest "${IMDB_TEST_MANIFEST}" \
    --imdb_task "${IMDB_TASK}" \
    --bias_type_train "${BIAS_TYPE_TRAIN_IMDB}" \
    \
    --num_main_classes ${NUM_MAIN_CLASSES_IMDB} \
    --num_bias_classes ${NUM_BIAS_CLASSES_IMDB} \
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
echo "IMDB-Face (${IMDB_TASK}, ${BIAS_TYPE_TRAIN_IMDB}) experiment finished."
echo "========================================================================"
