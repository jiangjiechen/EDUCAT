#!/usr/bin/env bash

DATE=$(date +%m%d)
SEED=1111

MODEL_TYPE=roberta
MAX_SEQ_LENGTH=256

PER_GPU_EVAL_BATCH_SIZE=32

DATA_DIR=$PJ_HOME/data/TimeTravel/nli_metrics
OUTPUT_DIR=$1
LOG_DIR=$OUTPUT_DIR

python3 train_classifier.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${OUTPUT_DIR} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --data_dir ${DATA_DIR} \
  --log_dir ${LOG_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --do_eval \
  --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
  --seed ${SEED}

