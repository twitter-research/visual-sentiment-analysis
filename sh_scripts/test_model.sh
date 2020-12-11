#!/usr/bin/env bash

MODEL_FILE=$1

DATE_TIME=$(date +%m%d%k%M)
SAVE_DIR="experiments/model_test/${DATE_TIME}"

python scripts/model_test.py \
 --model_file "${MODEL_FILE}" \
 --test_csv "${DATA_DIR}/test.csv" \
 --category_csv "${DATA_DIR}/categories.csv" \
 --save_dir "${SAVE_DIR}" \
 --seed 42 \
 --batch_size 128 \
 --test_steps -1 \
 --multilabel_k 2
