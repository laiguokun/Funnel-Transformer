#!/bin/bash

pretrain_dir=PATH_TO_DOWNLOADED_CKPT

task_name=mnli_matched
data_dir=../data/glue/MNLI
output_dir=../proc_data/glue/${task_name}
model_dir=exp/glue/${task_name}

lr=2e-5
train_bsz=8
epochs=3
max_length=128

python multigpu_classifier.py \
  --ddp_backend=apex \
  --data_dir=${data_dir} \
  --output_dir=${output_dir} \
  --model_dir=${model_dir} \
  --tokenizer_type=word_piece \
  --tokenizer_path=${pretrain_dir}/vocab.uncased.txt \
  --init_ckpt_config=${pretrain_dir}/net_config.pytorch.json \
  --init_ckpt=${pretrain_dir}/model.pt \
  --attn_type=rel_shift \
  --dataset=${task_name} \
  --max_length=${max_legnth} \
  --lr=${lr} \
  --train_bsz=${train_bsz} \
  --epochs=${epochs} \
  $@

#   --fp16 \
#   --amp_opt="O2" \
