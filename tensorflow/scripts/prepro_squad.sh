#!/bin/bash

# (1) Down load the raw data to some local dir
data_dir=

mkdir -p ${data_dir}
cd ${data_dir}

if [ -f train-v2.0.json ]; then
  echo "train-v2.0.json exists. Skip downloading."
else
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
fi
if [ -f dev-v2.0.json ]; then
  echo "dev-v2.0.json exists. Skip downloading."
else
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
fi

cd -

# (2) Specific the tokenizer path (local or GS)
pretrain_dir=
uncased=True
tokenizer_type=word_piece
tokenizer_path=${pretrain_dir}/vocab.uncased.txt

# (3) Set up google storage to save processed data
GS_ROOT=
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/squad

for split in train eval; do
  python squad.py \
    --output_dir=${GS_PROC_DATA_DIR} \
    --train_version="v2" \
    --train_file=${data_dir}/train-v2.0.json \
    --eval_version="v2" \
    --predict_file=${data_dir}/dev-v2.0.json \
    --uncased=${uncased} \
    --tokenizer_type=${tokenizer_type} \
    --tokenizer_path=${tokenizer_path} \
    --prepro_split=${split} \
    $@
done
