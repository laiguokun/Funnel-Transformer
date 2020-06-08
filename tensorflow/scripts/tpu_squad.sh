#!/bin/bash

TPU_NAME=

# TPU v2-8 (base models) or v3-8 (large models)
NUM_HOSTS=1
NUM_CORE_PER_HOST=8

GS_ROOT=
GS_INIT_CKPT_DIR=${GS_ROOT}/pretrained_ckpt
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/squad
GS_MODEL_DIR=${GS_ROOT}/exp/squad

uncased=True
tokenizer_type=word_piece
tokenizer_path=${GS_INIT_CKPT_DIR}/vocab.uncased.txt
init_checkpoint=${GS_INIT_CKPT_DIR}/model.ckpt
model_config=${GS_INIT_CKPT_DIR}/net_config.json

python squad.py \
    --use_tpu=True \
    --tpu=${TPU_NAME} \
    --use_bfloat16=True \
    --num_hosts=${NUM_HOSTS} \
    --num_core_per_host=${NUM_CORE_PER_HOST} \
    --output_dir=${output_dir} \
    --model_dir=${GS_MODEL_DIR} \
    --predict_dir=${GS_MODeL_DIR}/prediction \
    --init_checkpoint=${init_checkpoint} \
    --model_config=${model_config} \
    --uncased=${uncased} \
    --tokenizer_type=${tokenizer_type} \
    --tokenizer_path=${tokenizer_path} \
    --train_version="v2" \
    --eval_version="v2" \
    --learning_rate=3e-5 \
    --lr_layer_decay_rate=0.75 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --iterations=1000 \
    --save_steps=1000 \
    --train_batch_size=48 \
    --eval_batch_size=8 \
    --do_train=True \
    --do_predict=True \
    $@
