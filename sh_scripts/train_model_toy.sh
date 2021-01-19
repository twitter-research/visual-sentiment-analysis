#!/usr/bin/env bash
# Copyright 2020 Twitter, Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0


DATE_TIME=$(date +%m%d%k%M)
SAVE_DIR="experiments/model_train/${DATE_TIME}"

# Subsample dataset to small toy sample
for SPLIT in train valid
do
    head -n 1 "${DATA_DIR}/${SPLIT}.csv" > "${DATA_DIR}/tmp_header.csv"
    tail -n +2 "${DATA_DIR}/${SPLIT}.csv" | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .0001) print $0}' > "${DATA_DIR}/tmp_${SPLIT}.csv"
    cat "${DATA_DIR}/tmp_header.csv" "${DATA_DIR}/tmp_${SPLIT}.csv" > "${DATA_DIR}/${SPLIT}_toy.csv"
    rm "${DATA_DIR}/tmp_header.csv" "${DATA_DIR}/tmp_${SPLIT}.csv"
done

python scripts/model_train.py \
  --category_csv "${DATA_DIR}/categories.csv" \
  --train_csv "${DATA_DIR}/train_toy.csv" \
  --val_csv "${DATA_DIR}/valid_toy.csv" \
  --save_dir "${SAVE_DIR}" \
  --batch_size 128 \
  --num_workers 0 \
  --log_interval 1 \
  --optimizer "Adam" \
  --lr 1e-4 \
  --train_epochs 1 \
  --sampler_type "rnd"
