"""Funnel-Transformer."""

import json
import os

import torch
import torch.nn as nn
from ops import LayerNorm
from ops import EmbeddindLookup
from ops import RelMultiheadAttention
from ops import PositionwiseFFN
from ops import Dense
from ops import AttentionStructure


def parse_depth_string(depth_str):
  depth_config = depth_str.split("x")
  if len(depth_config) == 1:
    depth_config.append(1)
  assert len(depth_config) == 2, "Require two-element depth config."

  return list(map(int, depth_config))


class ModelConfig(object):
  """ModelConfig contains fixed hyperparameters of a FunnelTFM model."""

  keys = ["vocab_size", "d_embed", "d_model", "n_head", "d_head",
          "d_inner", "dropout", "dropatt", "dropact", "block_size",
          "pooling_type", "pooling_size", "separate_cls", "pool_q_only"]

  def __init__(self, vocab_size, d_embed, d_model, n_head, d_head,
               d_inner, dropout, dropatt, dropact, block_size,
               pooling_type, pooling_size,
               separate_cls, pool_q_only):

    self.vocab_size = vocab_size
    self.d_embed = d_embed
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner

    self.dropout = dropout
    self.dropatt = dropatt
    self.dropact = dropact
    self.block_size = block_size
    block_size = block_size.split("_")
    self.n_block = len(block_size)
    self.block_rep = []
    self.block_param = []
    for i, _ in enumerate(block_size):
      block_size_i = parse_depth_string(block_size[i])
      self.block_param.append(block_size_i[0])
      self.block_rep.append(block_size_i[1])


    self.pooling_type = pooling_type
    self.pooling_size = pooling_size
    self.separate_cls = separate_cls
    self.pool_q_only = pool_q_only


  @staticmethod
  def init_from_text(file_path, args, sep_symbol=None):
    """Initialize ModelConfig from a text file."""
    print("Initialize ModelConfig from text file %s.", file_path)
    conf_args = {}
    with open(file_path) as f:
      for line in f:
        k, v = line.strip().split(sep_symbol)
        if k in ModelConfig.keys:
          conf_args[k] = v
        else:
          print("Unused key %s", k)

    net_config = ModelConfig(**conf_args)

    # Merge loaded config and args
    for key in ModelConfig.keys:
      overwrite_keys = set(args.overwrite_keys.split(","))
      if key in overwrite_keys:
        setattr(net_config, key, getattr(args, key))
      else:
        setattr(args, key, getattr(net_config, key))

    return net_config

  @staticmethod
  def init_from_json(file_path, args):
    """Initialize ModelConfig from a json file."""
    print("Initialize ModelConfig from json file %s.", file_path)
    with open(file_path) as f:
      json_data = json.load(f)

    net_config = ModelConfig(**json_data)

    # Merge loaded config and args
    for key in ModelConfig.keys:
      overwrite_keys = set(args.overwrite_keys.split(","))
      if key in overwrite_keys:
        setattr(net_config, key, getattr(args, key))
      else:
        setattr(args, key, getattr(net_config, key))

    return net_config

  @staticmethod
  def init_from_args(args):
    """Initialize ModelConfig from args."""
    print("Initialize ModelConfig from args.")
    conf_args = {}
    for key in ModelConfig.keys:
      conf_args[key] = getattr(args, key)

    return ModelConfig(**conf_args)

  def to_json(self, json_path):
    """Save ModelConfig to a json file."""
    print("Save ModelConfig to json file {}.".format(json_path))
    json_data = {}
    for key in ModelConfig.keys:
      json_data[key] = getattr(self, key)

    json_dir = os.path.dirname(json_path)
    if not os.path.exists(json_dir):
      os.makedirs(json_dir)
    with open(json_path, "w") as f:
      json.dump(json_data, f, indent=4, sort_keys=True)


class FunnelTFM(nn.Module):
  """FunnelTFM model."""

  def __init__(self, net_config, args, cls_target=True):
    super(FunnelTFM, self).__init__()
    self.net_config = net_config
    self.args = args
    self.input_layer = nn.Sequential(
        EmbeddindLookup(net_config.vocab_size,
                        net_config.d_embed),
        LayerNorm(net_config.d_embed),
        nn.Dropout(net_config.dropout))

    self.pos_drop = nn.Dropout(net_config.dropout)

    self.attn_info = AttentionStructure(net_config, args)
    self.attn_layers = nn.ModuleList()
    self.pffn_layers = nn.ModuleList()
    for block_idx in range(net_config.n_block):
      for _ in range(net_config.block_param[block_idx]):
        self.attn_layers.append(
            RelMultiheadAttention(
                net_config,
                args,
                net_config.d_model,
                net_config.n_head,
                net_config.d_head,
                net_config.dropout,
                net_config.dropatt,
                block_idx,
            )
        )
        self.pffn_layers.append(
            PositionwiseFFN(
                net_config.d_model,
                net_config.d_inner,
                net_config.dropout,
                net_config.dropact,
            )
        )
    if cls_target:
      self.build_cls_head()

  def build_cls_head(self):
    if not hasattr(self, "cls_head"):
      net_config = self.net_config
      self.cls_head = nn.Sequential(
          Dense(net_config.d_model, net_config.d_model),
          nn.Tanh(),
          nn.Dropout(net_config.dropout),
          Dense(net_config.d_model, self.args.num_class))
      self.cls_loss = nn.CrossEntropyLoss()

  def tfmxl_layers(self, inputs, seg_id=None, input_mask=None):
    net_config = self.net_config
    output = inputs

    ret_dict = {}
    ##### TFM-XL layers
    hiddens = []
    layer_idx = 0
    attn_struct = self.attn_info.init_attn_structure(output, seg_id, input_mask)
    for block_idx in range(net_config.n_block):
      if net_config.separate_cls:
        pooling_flag = output.size(1) > 2
      else:
        pooling_flag = output.size(1) > 1

      if block_idx > 0 and pooling_flag:
        pooled_out, attn_struct, _ = self.attn_info.pre_attn_pooling(
            output, attn_struct)
      for param_idx in range(net_config.block_param[block_idx]):
        for rep_idx in range(net_config.block_rep[block_idx]):
          sub_idx = param_idx * net_config.block_rep[block_idx] + rep_idx
          do_pooling = sub_idx == 0 and block_idx > 0 and pooling_flag

          q = k = v = output
          # update q, k, v for the first sub layer of a pooling block
          if do_pooling:
            if net_config.pool_q_only:
              q = pooled_out
              k = v = output
            else:
              q = k = v = pooled_out
          output = self.attn_layers[layer_idx](q, k, v, attn_struct)
          output = self.pffn_layers[layer_idx](output)
          if do_pooling:
            attn_struct = self.attn_info.post_attn_pooling(attn_struct)

          hiddens.append(output)

        layer_idx += 1
    #print(torch.max(hiddens[-1][0][0]))
    return hiddens, ret_dict

  def extract_hiddens(self, inputs, seg_id=None, input_mask=None):
    word_embed = self.input_layer(inputs)
    hiddens, tfm_dict = self.tfmxl_layers(
        word_embed,
        seg_id=seg_id,
        input_mask=input_mask)
    return [word_embed] + hiddens, tfm_dict

  def forward(self, inputs, input_mask=None, seg_id=None,
              cls_target=None):

    if input_mask is None and self.args.pad_id is not None:
      input_mask = inputs == self.args.pad_id
      input_mask = input_mask.float()

    hiddens, tfm_dict = self.extract_hiddens(
        inputs, seg_id=seg_id, input_mask=input_mask)

    ret_dict = {}
    if cls_target is not None:
      last_hidden = hiddens[-1][:, 0]
      cls_logits = self.cls_head(last_hidden)
      prediction = torch.argmax(cls_logits, -1)
      ret_dict["cls_pred"] = prediction
      cls_loss = self.cls_loss(cls_logits, cls_target)
      ret_dict["cls_loss"] = cls_loss
      cls_correct = prediction == cls_target
      cls_correct = cls_correct.type(torch.float32).sum()
      ret_dict["cls_corr"] = cls_correct
    update_monitor_dict(ret_dict, tfm_dict)
    return hiddens, ret_dict


def update_monitor_dict(tgt, src, prefix=None):
  if prefix is None:
    tgt.update(src)
  else:
    for k, v in src.items():
      tgt["{}/{}".format(prefix, k)] = v

  return tgt
