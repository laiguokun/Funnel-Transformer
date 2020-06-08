#!/bin/bash

bsz=16
num_core=1

root=${PWD}/..
pretrain_dir=${root}/funnel_ckpts_new/B4-4-4H768-ELEC-TF
init_checkpoint=${pretrain_dir}/model.ckpt
model_config=${pretrain_dir}/net_config.json

# task=sts-b
# data_dir=${root}/data/glue/STS-B

task=rte
data_dir=${root}/data/glue/RTE

# task=mrpc
# data_dir=${root}/data/glue/MRPC

# task=mnli_matched
# data_dir=${root}/data/glue/MNLI

output_dir=${root}/proc_data/glue/${task}
model_dir=${root}/exp/glue/${task}

uncased=True
tokenizer_type=word_piece
tokenizer_path=${pretrain_dir}/vocab.uncased.txt

rm -rf ${model_dir}

python classifier.py \
    --data_dir=${data_dir} \
    --output_dir=${output_dir} \
    --uncased=${uncased} \
    --tokenizer_type=${tokenizer_type} \
    --tokenizer_path=${tokenizer_path} \
    --model_dir=${model_dir} \
    --init_checkpoint=${init_checkpoint} \
    --model_config=${model_config} \
    --learning_rate=0.00001 \
    --warmup_steps=155 \
    --train_steps=1550 \
    --train_batch_size=${bsz} \
    --num_core_per_host=${num_core} \
    --iterations=155 \
    --save_steps=155 \
    --do_train=True \
    --do_eval=True \
    --do_submit=True \
    --use_tpu=False \
    --task_name=${task} \
    $@

