#!/bin/bash

root="../"
git clone https://github.com/wasiahmad/paraphrase_identification.git
mkdir -p ${root}/data/glue 
python ${root}/misc/download_glue_data.py \
  --data_dir ${root}/data/glue \
  --tasks all \
  --path_to_mrpc=paraphrase_identification/dataset/msr-paraphrase-corpus

rm -rf paraphrase_identification
