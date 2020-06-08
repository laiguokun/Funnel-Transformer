#!/bin/bash

pretrain_dir=PATH_TO_PRETRAINED_DIR

task_name=CoLA
data_dir=../data/glue/${task_name}
output_dir=../proc_data/glue/${task_name}
model_dir=exp/glue/${task_name}

lr=1e-5
train_bsz=16
epochs=10
max_length=128

python classifier.py \
  --device_id=0 \
  --data_dir=${data_dir} \
  --output_dir=${output_dir} \
  --model_dir=${model_dir} \
  --tokenizer_type=word_piece \
  --tokenizer_path=${pretrain_dir}/vocab.uncased.txt \
  --init_ckpt_config=${pretrain_dir}/net_config.pytorch.json \
  --init_ckpt=${pretrain_dir}/model.pt \
  --attn_type=rel_shift \
  --dataset=${task_name} \
  --max_length=${max_length} \
  --lr=${lr} \
  --train_bsz=${train_bsz} \
  --epochs=${epochs} \
  $@

#   --fp16 \
#   --amp_opt="O2" \
