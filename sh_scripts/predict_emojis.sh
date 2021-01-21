#!/usr/bin/env bash
# Copyright 2020 Twitter, Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0


MODEL_FILE=$1
IMAGE_DIR=$2

DATE_TIME=$(date +%m%d%k%M)
SAVE_DIR="experiments/model_predict/${DATE_TIME}"

python scripts/model_predict.py \
 --model_file "${MODEL_FILE}" \
 --image_dir "${IMAGE_DIR}" \
 --category_csv "${DATA_DIR}/categories.csv" \
 --save_dir "${SAVE_DIR}" \
 --batch_size 128
