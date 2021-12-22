#!/usr/bin/env bash

DATE=$(date +%m%d)
SEED=1111

MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=roberta-base
MAX_SEQ_LENGTH=128

NUM_TRAIN_EPOCH=20
GRADIENT_ACCUMULATION_STEPS=1
PER_GPU_TRAIN_BATCH_SIZE=32
PER_GPU_EVAL_BATCH_SIZE=32
LOGGING_STEPS=200
SAVE_STEPS=200
LEARNING_RATE=2e-5

DATA_DIR=$PJ_HOME/data/TimeTravel/nli_metrics
OUTPUT_DIR=$PJ_HOME/models/nli_metrics/${MODEL_NAME_OR_PATH}_${DATE}
LOG_DIR=$OUTPUT_DIR

mkdir -p $OUTPUT_DIR

python3 convert_data.py

python3 train_classifier.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${LOG_DIR} \
  --do_train \
  --save_best \
  --overwrite_output_dir \
  --num_train_epochs ${NUM_TRAIN_EPOCH} \
  --learning_rate ${LEARNING_RATE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
  --logging_steps ${LOGGING_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --seed ${SEED}

python3 cjjpy.py --lark "$OUTPUT_DIR metrics done"
