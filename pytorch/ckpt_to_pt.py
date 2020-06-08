"""Convert a tf checkpoint to pytorch pt"""

from collections import OrderedDict
import json

from absl import flags
import torch
import tensorflow.compat.v1 as tf
import numpy as np

import modeling
flags.DEFINE_string("input_ckpt_path", "",
                    help="input ckpt for cleaning")
flags.DEFINE_string("input_config_path", "",
                    help="input ckpt for cleaning")
flags.DEFINE_string("output_pt_path", "",
                    help="output path for torch version ckpt")
flags.DEFINE_string("output_config_path", "",
                    help="output path for torch version config")

FLAGS = flags.FLAGS


def convert_to_tensor(x, idx=None):
  if x["dtype"] == np.float32:
    dtype = torch.float32
  else:
    dtype = torch.float16

  if idx is None:
    return torch.from_numpy(x["weight"]).type(dtype)
  else:
    return torch.from_numpy(x["weight"][idx]).dtype(dtype)

def embed_param(new_model, tf_model, prefix, tgt_prefix):
  # remove the mask embed
  new_model[tgt_prefix + "input_layer.0.lookup_table"] = \
      convert_to_tensor(tf_model[prefix+"input/word_embedding/lookup_table"])
  # layer norm
  new_model[tgt_prefix + "input_layer.1.weight"] = \
      convert_to_tensor(tf_model[prefix+"input/layer_norm/gamma"])
  new_model[tgt_prefix + "input_layer.1.bias"] = \
      convert_to_tensor(tf_model[prefix+"input/layer_norm/beta"])


def attn_param(new_model, tf_model, prefix, tgt_prefix, idx):
  # qkv proj
  q = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/q/kernel'.format(idx)])
  k = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/k/kernel'.format(idx)])
  v = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/v/kernel'.format(idx)])
  # qkv bias
  k_bias = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/k/bias'.format(idx)])
  v_bias = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/v/bias'.format(idx)])
  # o proj
  o = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/o/kernel'.format(idx)])
  o_bias = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/o/bias'.format(idx)])

  new_model[tgt_prefix+"attn_layers.{}.q_head.weight".format(idx)] = q
  new_model[tgt_prefix+"attn_layers.{}.k_head.weight".format(idx)] = k
  new_model[tgt_prefix+"attn_layers.{}.v_head.weight".format(idx)] = v
  new_model[tgt_prefix+"attn_layers.{}.post_proj.weight".format(idx)] = o
  new_model[tgt_prefix+"attn_layers.{}.k_head.bias".format(idx)] = k_bias
  new_model[tgt_prefix+"attn_layers.{}.v_head.bias".format(idx)] = v_bias
  new_model[tgt_prefix+"attn_layers.{}.post_proj.bias".format(idx)] = o_bias
  # rel attn
  r_w_bias = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/r_w_bias'.format(idx)])
  r_s_bias = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/r_s_bias'.format(idx)])
  r_r_bias = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/r_r_bias'.format(idx)])
  seg_embed = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/seg_embed'.format(idx)])
  r_kernel = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/rel_attn/r/kernel'.format(idx)])
  new_model[tgt_prefix+"attn_layers.{}.r_w_bias".format(idx)] = r_w_bias
  new_model[tgt_prefix+"attn_layers.{}.r_s_bias".format(idx)] = r_s_bias
  new_model[tgt_prefix+"attn_layers.{}.r_r_bias".format(idx)] = r_r_bias
  new_model[tgt_prefix+"attn_layers.{}.seg_embed".format(idx)] = seg_embed
  new_model[tgt_prefix+"attn_layers.{}.r_kernel".format(idx)] = r_kernel

def ffn_param(new_model, tf_model, prefix, tgt_prefix, idx):
  kernel_1 = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/ff/layer_1/kernel'.format(idx)])
  kernel_2 = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/ff/layer_2/kernel'.format(idx)])
  bias_1 = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/ff/layer_1/bias'.format(idx)])
  bias_2 = convert_to_tensor(
      tf_model[prefix+'encoder/layer_{}/ff/layer_2/bias'.format(idx)])
  new_model[tgt_prefix+"pffn_layers.{}.pffn.0.weight".format(idx)] = kernel_1
  new_model[tgt_prefix+"pffn_layers.{}.pffn.3.weight".format(idx)] = kernel_2
  new_model[tgt_prefix+"pffn_layers.{}.pffn.0.bias".format(idx)] = bias_1
  new_model[tgt_prefix+"pffn_layers.{}.pffn.3.bias".format(idx)] = bias_2

def ln_param(new_model, tf_model, prefix, tgt_prefix, idx):
  # attn layer norm
  gamma = convert_to_tensor(
      tf_model[prefix+"encoder/layer_{}/rel_attn/layer_norm/gamma".format(idx)])
  beta = convert_to_tensor(
      tf_model[prefix+"encoder/layer_{}/rel_attn/layer_norm/beta".format(idx)])
  new_model[tgt_prefix+"attn_layers.{}.layer_norm.weight".format(idx)] = gamma
  new_model[tgt_prefix+"attn_layers.{}.layer_norm.bias".format(idx)] = beta
  # ffn layer norm
  gamma = convert_to_tensor(
      tf_model[prefix+"encoder/layer_{}/ff/layer_norm/gamma".format(idx)])
  beta = convert_to_tensor(
      tf_model[prefix+"encoder/layer_{}/ff/layer_norm/beta".format(idx)])
  new_model[tgt_prefix+"pffn_layers.{}.layer_norm.weight".format(idx)] = gamma
  new_model[tgt_prefix+"pffn_layers.{}.layer_norm.bias".format(idx)] = beta

def modify_json_data(json_data):
  keys = ["vocab_size", "d_embed", "d_model", "n_head", "d_head",
          "d_inner", "dropout", "dropatt", "dropact", "block_size",
          "pooling_type", "pooling_size", "separate_cls", "pool_q_only"]
  key_list = list(json_data.keys())
  for item in key_list:
    if item not in keys:
      json_data.pop(item, None)
  print(json_data)
  return json_data

def clean_ckpt(_):
  """Core function."""
  input_ckpt = FLAGS.input_ckpt_path
  model = {}
  tf.reset_default_graph()

  tf.logging.info("Loading from %s", input_ckpt)
  var_list = tf.train.list_variables(input_ckpt)
  reader = tf.train.load_checkpoint(input_ckpt)
  var_values, var_dtypes = {}, {}

  # clean optimizer
  for (name, _) in var_list:
    if name.startswith("global_step") or "adam" in name.lower():
      continue
    tensor = reader.get_tensor(name)
    var_dtypes[name] = tensor.dtype
    var_values[name] = tensor

    model[name] = {'dtype': tensor.dtype, 'weight': tensor}

  tf_model = model
  new_model = OrderedDict()
  prefix = "model/"
  tgt_prefix = ""
  with open(FLAGS.input_config_path) as f:
    json_data = json.load(f)
  json_data = modify_json_data(json_data)
  with open(FLAGS.output_config_path, "w") as f:
    json.dump(json_data, f, indent=4, sort_keys=True)
  net_config = modeling.ModelConfig(**json_data)
  embed_param(new_model, tf_model, prefix, tgt_prefix)
  cum_layer_idx = 0
  for block_idx in range(net_config.n_block):
    for _ in range(net_config.block_param[block_idx]):
      attn_param(new_model, tf_model, prefix, tgt_prefix, cum_layer_idx)
      ffn_param(new_model, tf_model, prefix, tgt_prefix, cum_layer_idx)
      ln_param(new_model, tf_model, prefix, tgt_prefix, cum_layer_idx)
      cum_layer_idx += 1
  torch.save(new_model, FLAGS.output_pt_path)

if __name__ == "__main__":
  tf.app.run(clean_ckpt)
