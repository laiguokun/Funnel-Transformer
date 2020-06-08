#!/bin/bash

# Example code of converting TensorFlow checkpoint to PyTorch checkpoint
TF_dir=
PT_dir=
mkdir -p ${PT_dir}

python ckpt_to_pt.py \
  --input_ckpt_path="${TF_dir}/model.ckpt" \
  --input_config_path="${TF_dir}/net_config.json" \
  --output_pt_path="${PT_dir}/model.pt" \
  --output_config_path="${PT_dir}/net_config.pytorch.json"
