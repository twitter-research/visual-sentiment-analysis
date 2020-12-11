#!/usr/bin/env bash

DATE_TIME=$(date +%m%d%k%M)
SAVE_DIR="experiments/model_train/${DATE_TIME}"

python scripts/model_train.py \
  --category_csv "${DATA_DIR}/categories.csv" \
  --train_csv "${DATA_DIR}/train.csv" \
  --val_csv "${DATA_DIR}/valid.csv" \
  --save_dir "${SAVE_DIR}" \
  --batch_size 128 \
  --num_workers 0 \
  --log_interval 1 \
  --optimizer "Adam" \
  --lr 1e-4 \
  --train_epochs 2 \
  --sampler_type "rnd"
