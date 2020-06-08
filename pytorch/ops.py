"""Common operations used to construct model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


INF = 1e6

try:
  import apex
  LayerNorm = apex.normalization.FusedLayerNorm
except ImportError as e:
  class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
      super(LayerNorm, self).__init__(*args, **kwargs)
      self.eps = 1e-9

    def forward(self, inputs):
      dtype = torch.float32
      if self.elementwise_affine:
        weight = self.weight.type(dtype)
        bias = self.bias.type(dtype)
      else:
        weight = self.weight
        bias = self.bias
      input_dtype = inputs.dtype
      inputs = inputs.type(dtype)
      output = F.layer_norm(inputs, self.normalized_shape, weight, bias, self.eps)
      if output.dtype != input_dtype:
        output = output.type(input_dtype)
      return output

class GELU(nn.Module):
  def forward(self, x):
    cdf = 0.5 * (1.0 + torch.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf


class EmbeddindLookup(nn.Module):
  def __init__(self, n_embed, d_embed):
    super(EmbeddindLookup, self).__init__()
    self.lookup_table = nn.Parameter(torch.zeros([n_embed, d_embed]))
    self.reset_parameters()

  def reset_parameters(self):
    init.normal_(self.lookup_table)

  def forward(self, inputs):
    return F.embedding(inputs, self.lookup_table)


def maybe_convert_to_list(x):
  if isinstance(x, (int, float)):
    return [x]
  elif isinstance(x, (list, tuple)):
    return list(x)


def get_einsum_string(ndims, einsum_symbols=None):
  if einsum_symbols is None:
    einsum_symbols = ["u", "v", "w", "x", "y", "z"]
  assert ndims <= len(einsum_symbols)
  einsum_prefix = ""
  for i in range(ndims):
    einsum_prefix += einsum_symbols[i]

  return einsum_prefix

class Dense(nn.Module):
  """Dense layer."""

  def __init__(self, inp_shape, out_shape, bias=True, reverse_order=False):
    super(Dense, self).__init__()

    self.inp_shape = maybe_convert_to_list(inp_shape)
    self.out_shape = maybe_convert_to_list(out_shape)

    self.reverse_order = reverse_order
    if self.reverse_order:
      self.einsum_str = "...{0},{1}{0}->...{1}".format(
          get_einsum_string(len(self.inp_shape), ["a", "b", "c", "d"]),
          get_einsum_string(len(self.out_shape), ["e", "f", "g", "h"]))
      weight_shape = self.out_shape + self.inp_shape
    else:
      self.einsum_str = "...{0},{0}{1}->...{1}".format(
          get_einsum_string(len(self.inp_shape), ["a", "b", "c", "d"]),
          get_einsum_string(len(self.out_shape), ["e", "f", "g", "h"]))
      weight_shape = self.inp_shape + self.out_shape

    self.weight = nn.Parameter(torch.zeros(weight_shape))
    if bias:
      self.bias = nn.Parameter(torch.zeros(self.out_shape))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self):
    fan_in = np.prod(self.inp_shape)
    fan_out = np.prod(self.out_shape)
    std = np.sqrt(1.0 / float(fan_in + fan_out))

    nn.init.normal_(self.weight, std=std)
    if self.bias is not None:
      nn.init.constant_(self.bias, 0.)

  def forward(self, inputs):
    output = torch.einsum(self.einsum_str, inputs, self.weight)
    if self.bias is not None:
      output = output + self.bias
    return output

  def extra_repr(self):
    return "inp_shape={}, out_shape={}, bias={}".format(self.inp_shape,
                                                        self.out_shape,
                                                        self.bias is not None)

class PositionwiseFFN(nn.Module):
  """Positionwas Feed-forward Net."""

  def __init__(self, d_model, d_inner, dropout, dropact):
    super(PositionwiseFFN, self).__init__()
    self.pffn = nn.Sequential(
        Dense(d_model, d_inner),
        GELU(),
        nn.Dropout(dropact),
        Dense(d_inner, d_model),
        nn.Dropout(dropout))
    self.layer_norm = LayerNorm(d_model)

  def forward(self, inputs):
    pffn_out = self.pffn(inputs)
    output = self.layer_norm(inputs + pffn_out)
    return output


class RelMultiheadAttention(nn.Module):
  """Relative multi-head attention."""

  def __init__(self, net_config, args, d_model, n_head, d_head, dropout,
               dropatt, bidx):
    super(RelMultiheadAttention, self).__init__()

    self.net_config = net_config
    self.args = args
    self.attn_type = args.attn_type
    self.bidx = bidx

    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head

    self.dropout = dropout
    self.dropatt = dropatt

    self.att_drop = nn.Dropout(self.dropatt)
    self.hid_drop = nn.Dropout(self.dropout)

    self.q_head = Dense(d_model, [n_head, d_head], bias=False,)
    self.k_head = Dense(d_model, [n_head, d_head])
    self.v_head = Dense(d_model, [n_head, d_head])

    self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
    self.r_r_bias = nn.Parameter(torch.zeros([n_head, d_head]))
    self.r_kernel = nn.Parameter(torch.zeros([d_model, n_head, d_head]))
    self.r_s_bias = nn.Parameter(torch.zeros([n_head, d_head]))
    self.seg_embed = nn.Parameter(torch.zeros([2, n_head, d_head]))

    self.post_proj = Dense([n_head, d_head], d_model,)
    self.layer_norm = LayerNorm(d_model)
    self.scale = 1. / np.sqrt(d_head)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.uniform_(self.r_w_bias, b=0.1)
    nn.init.uniform_(self.r_r_bias, b=0.1)
    nn.init.uniform_(self.r_kernel, b=0.1)
    nn.init.uniform_(self.r_s_bias, b=0.1)
    nn.init.uniform_(self.seg_embed, b=0.1)

  def rel_shift(self, x, row_axis, key_len, shift=1):
    """Perform relative shift to form the relative attention score."""
    # Deal with negative indexing
    row_axis = row_axis % x.ndim

    # Assume `col_axis` = `row_axis + 1`
    col_axis = row_axis + 1
    assert col_axis < x.ndim

    tgt_shape_1, tgt_shape_2 = [], []
    for i in range(x.ndim):
      if i == row_axis:
        tgt_shape_1.append(x.shape[col_axis])
        tgt_shape_2.append(x.shape[row_axis])
      elif i == col_axis:
        tgt_shape_1.append(x.shape[row_axis])
        tgt_shape_2.append(x.shape[col_axis] - shift)
      else:
        tgt_shape_1.append(x.shape[i])
        tgt_shape_2.append(x.shape[i])

    y = torch.reshape(x, tgt_shape_1)
    y = torch.narrow(y, row_axis, shift, x.shape[col_axis] - shift)
    y = torch.reshape(y, tgt_shape_2)
    y = torch.narrow(y, col_axis, 0, key_len)

    return y

  def rel_pos_bias(self, pos_enc, q_head, k_len, func_mask=None):
    n_head = self.n_head
    d_head = self.d_head
    net_config = self.net_config
    scale = self.scale
    r_r_bias = self.r_r_bias
    r_kernel = self.r_kernel
    if self.attn_type == "factorized":
      enc_q_1, enc_q_2, enc_k_1, enc_k_2 = pos_enc
      q_head_r = torch.einsum("...inh,dnh->...ind",
                              q_head + r_r_bias * scale,
                              r_kernel)
      q_head_r_1 = q_head_r * torch.unsqueeze(enc_q_1, -2)
      q_head_r_2 = q_head_r * torch.unsqueeze(enc_q_2, -2)
      prefix_k = get_einsum_string(len(enc_k_1.shape) - 2)
      einsum_str = "...ind,{0}jd->...nij".format(prefix_k)
      bd = (torch.einsum(einsum_str, q_head_r_1, enc_k_1) +
            torch.einsum(einsum_str, q_head_r_2, enc_k_2))
    elif self.attn_type == "rel_shift":
      if k_len != q_head.size(1):
        # pooling case
        shift = 2
        pos_enc = pos_enc[self.bidx][1]
      else:
        shift = 1
        pos_enc = pos_enc[self.bidx][0]
      q_head = q_head + r_r_bias * scale
      r_head = torch.einsum("td,dnh->tnh", pos_enc, r_kernel)
      bd = torch.einsum("bfnh,tnh->bnft", q_head, r_head)
      bd = self.rel_shift(bd, -2, k_len, shift)
    else:
      raise NotImplementedError
    if func_mask is not None:
      bd = bd * func_mask
    return bd

  def rel_seg_bias(self, seg_mat, q_head, func_mask=None):
    # segment based attention score

    if seg_mat is None:
      seg_bias = 0
    else:
      r_s_bias = self.r_s_bias * self.scale
      seg_embed = self.seg_embed

      seg_bias = torch.einsum("...ind,snd->...nis",
                              q_head + r_s_bias, seg_embed)
      tgt_shape = list(seg_mat.size())
      tgt_shape.insert(-2, self.n_head)
      seg_mat = torch.unsqueeze(seg_mat, -3).expand(tgt_shape)
      _diff, _same = torch.split(seg_bias, 1, dim=-1)
      _diff = _diff.expand(tgt_shape)
      _same = _same.expand(tgt_shape)
      seg_bias = torch.where(seg_mat, _same, _diff)
      if func_mask is not None:
        seg_bias *= func_mask
    return seg_bias

  def forward(self, q, k, v, attn_struct):
    pos_enc, seg_mat, input_mask, attn_mask, func_mask = attn_struct
    q_head = self.q_head(q)
    k_head = self.k_head(k)
    v_head = self.v_head(v)

    q_head = q_head * self.scale
    r_w_bias = self.r_w_bias * self.scale
    # content based attention score
    content_score = torch.einsum("...ind,...jnd->...nij",
                                 q_head + r_w_bias, k_head)
    pos_bias = self.rel_pos_bias(pos_enc, q_head, k_head.size(1), func_mask)

    seg_bias = self.rel_seg_bias(seg_mat, q_head, func_mask)
    # merge attention scores
    attn_score = content_score + pos_bias + seg_bias

    # precision safe
    dtype = attn_score.dtype
    attn_score = attn_score.float()
    # perform masking
    if attn_mask is not None:
      attn_score = attn_score - INF * attn_mask.float()
    # attention probability
    attn_prob = torch.softmax(attn_score, dim=-1)
    attn_prob = attn_prob.type(dtype)

    attn_prob = self.att_drop(attn_prob)
    # attention output
    attn_vec = torch.einsum("...nij,...jnd->...ind", attn_prob, v_head)

    attn_out = self.post_proj(attn_vec)
    attn_out = self.hid_drop(attn_out)

    output = self.layer_norm(q + attn_out)
    return output


class AttentionStructure(nn.Module):
  """Relative multi-head attention."""

  def __init__(self, net_config, args, dtype=torch.float32, device=None):
    super(AttentionStructure, self).__init__()

    self.net_config = net_config
    self.dtype = dtype
    self.device = device
    self.sin_drop = nn.Dropout(net_config.dropout)
    self.cos_drop = nn.Dropout(net_config.dropout)
    self.args = args
    self.attn_type = args.attn_type
    self.delta = None

  def stride_pool_pos(self, pos_id, bidx):
    net_config = self.net_config
    if net_config.separate_cls:
      # Under separate [cls], we treat the [cls] as the first token in
      # the previous block of the 1st real block. Since the 1st real
      # block always has position 1, the position of the previous block
      # will 1 - 2**bidx, where `2 ** bidx` is the current stride.
      cls_pos = pos_id.new_tensor([-2**bidx + 1])
      if self.args.truncate_seq:
        pooled_pos_id = pos_id[1:-1]
      else:
        pooled_pos_id = pos_id[1:]
      pooled_pos_id = torch.cat([cls_pos, pooled_pos_id[::2]], 0)
    else:
      pooled_pos_id = pos_id[::2]

    return pooled_pos_id

  def construct_rel_pos_seq(self, q_pos, q_stride, k_pos, k_stride):
    net_config = self.net_config
    shift = q_stride // k_stride
    pool_size = net_config.pooling_size

    ref_point = q_pos[0] - k_pos[0]
    num_remove = shift * len(q_pos)
    max_dist = ref_point + num_remove * k_stride
    min_dist = q_pos[0] - k_pos[-1]

    rel_pos_id = torch.arange(max_dist,
                              min_dist - 1,
                              -k_stride,
                              dtype=torch.long,
                              device=q_pos.device)

    return rel_pos_id

  def get_pos_enc(self, seq_len, dtype, device):
    """Create inputs related to relative position encoding."""
    net_config = self.net_config
    if self.attn_type == "factorized":
      pos_seq = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
      pos_seq_q, pos_seq_k = pos_seq, pos_seq
      d_model = self.net_config.d_model
      d_model_half = d_model // 2
      freq_seq = torch.arange(0, d_model_half, 1.0,
                              dtype=dtype, device=device)
      inv_freq = 1 / (10000 ** (freq_seq / d_model_half))
      sinusoid_q = torch.einsum("...i,d->...id", pos_seq_q, inv_freq)
      sinusoid_k = torch.einsum("...i,d->...id", pos_seq_k, inv_freq)
      sin_enc_q = torch.sin(sinusoid_q)
      cos_enc_q = torch.cos(sinusoid_q)
      sin_enc_q = self.sin_drop(sin_enc_q)
      cos_enc_q = self.cos_drop(cos_enc_q)
      sin_enc_k = torch.sin(sinusoid_k)
      cos_enc_k = torch.cos(sinusoid_k)
      enc_q_1 = torch.cat([sin_enc_q, sin_enc_q], dim=-1)
      enc_k_1 = torch.cat([cos_enc_k, sin_enc_k], dim=-1)
      enc_q_2 = torch.cat([cos_enc_q, cos_enc_q], dim=-1)
      enc_k_2 = torch.cat([-sin_enc_k, cos_enc_k], dim=-1)
      return [enc_q_1, enc_q_2, enc_k_1, enc_k_2]
    elif self.attn_type == "rel_shift":
      d_model = self.net_config.d_model
      d_model_half = d_model // 2
      freq_seq = torch.arange(0, d_model_half, 1.0,
                              dtype=dtype, device=device)
      inv_freq = 1 / (10000 ** (freq_seq / d_model_half))

      # initialize an extra long position sequnece
      rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0,
                                dtype=dtype, device=device)
      zero_offset = seq_len * 2

      sinusoid = torch.einsum("...i,d->...id", rel_pos_id, inv_freq)
      sin_enc = torch.sin(sinusoid)
      cos_enc = torch.cos(sinusoid)
      sin_enc = self.sin_drop(sin_enc)
      cos_enc = self.cos_drop(cos_enc)
      pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)

      # Pre-compute and cache the rel_pos_id for all blocks
      pos_id = torch.arange(0, seq_len, dtype=dtype, device=device)
      pooled_pos_id = pos_id
      pos_enc_list = []

      for bidx in range(0, self.net_config.n_block):
        # For each block with bidx > 0, we need two types pos_encs:
        #   - Attn(pooled-q, unpooled-kv)
        #   - Attn(pooled-q, pooled-kv)

        #### First type: Attn(pooled-q, unpooled-kv)
        if bidx > 0:
          pooled_pos_id = self.stride_pool_pos(pos_id, bidx)

          # construct rel_pos_id
          q_stride = self.net_config.pooling_size ** bidx
          k_stride = self.net_config.pooling_size ** (bidx - 1)
          rel_pos_id = self.construct_rel_pos_seq(
              q_pos=pooled_pos_id, q_stride=q_stride,
              k_pos=pos_id, k_stride=k_stride)

          # gather relative positional encoding
          rel_pos_id = rel_pos_id[:, None] + zero_offset
          rel_pos_id = rel_pos_id.expand(rel_pos_id.size(0), d_model)
          pos_enc_2 = torch.gather(pos_enc, 0, rel_pos_id)
        else:
          pos_enc_2 = None

        #### Second type: Attn(pooled-q, pooled-kv)
        # construct rel_pos_id
        pos_id = pooled_pos_id
        stride = self.net_config.pooling_size ** bidx
        rel_pos_id = self.construct_rel_pos_seq(
            q_pos=pos_id, q_stride=stride,
            k_pos=pos_id, k_stride=stride)

        # gather relative positional encoding
        rel_pos_id = rel_pos_id[:, None] + zero_offset
        rel_pos_id = rel_pos_id.expand(rel_pos_id.size(0), d_model)
        pos_enc_1 = torch.gather(pos_enc, 0, rel_pos_id)

        pos_enc_list.append([pos_enc_1, pos_enc_2])
      return pos_enc_list
    else:
      raise NotImplementedError

  def seg_id_to_mat(self, seg_q, seg_k):
    """Convert `seg_id` to `seg_mat`."""
    seg_mat = torch.eq(torch.unsqueeze(seg_q, -1), torch.unsqueeze(seg_k, -2))

    # Treat [cls] as in the same segment as both A & B
    cls_mat = torch.unsqueeze(torch.eq(seg_q, self.args.seg_id_cls), -1) | \
        torch.unsqueeze(torch.eq(seg_k, self.args.seg_id_cls), -2)
    seg_mat = cls_mat | seg_mat

    return seg_mat

  def get_attn_mask(self, input_mask):
    if input_mask is None:
      attn_mask = None
    else:
      attn_mask = input_mask[:, None, None, :]
    return attn_mask

  def init_attn_structure(self, hidden, seg_id=None, input_mask=None):
    net_config = self.net_config
    self.delta = 1
    seq_len = hidden.size(1)
    self.seq_len = seq_len
    pos_enc = self.get_pos_enc(seq_len, hidden.dtype, hidden.device)

    if seg_id is None:
      seg_mat = None
    else:
      seg_mat = self.seg_id_to_mat(seg_id, seg_id)

    if net_config.separate_cls:
      func_mask = F.pad(
          torch.ones([seq_len - 1, seq_len - 1],
                     dtype=hidden.dtype,
                     device=hidden.device),
          (1, 0, 1, 0))
    else:
      func_mask = None

    attn_mask = self.get_attn_mask(input_mask)
    attn_struct = (pos_enc, seg_mat, input_mask, attn_mask, func_mask)
    return attn_struct

  def stride_pool(self, tensor, axis):
    """Perform pooling by stride slicing the tensor along the given axis."""
    if tensor is None:
      return None

    net_config = self.net_config
    if isinstance(tensor, (tuple, list)):
      ndims = tensor[0].dim()
    else:
      ndims = tensor.dim()
    axis = axis % ndims

    enc_slice = []
    for i in range(ndims):
      if i == axis:
        if net_config.separate_cls and self.args.truncate_seq:
          enc_slice.append(slice(None, -1, 2))
        else:
          enc_slice.append(slice(None, None, 2))
        break
      else:
        enc_slice.append(slice(None))

    if net_config.separate_cls:
      cls_slice = []
      for i in range(ndims):
        if i == axis:
          cls_slice.append(slice(None, 1))
          break
        else:
          cls_slice.append(slice(None))

    def _pool_func(enc):
      # separate_cls = True
      #   trunc = False
      #     [0 1 2 3 4 5 6 7] => [0] & [1 2 3 4 5 6 7] => [0] & [1 3 5 7]
      #     [0 1 3 5 7] => [0] & [1 3 5 7] => [0] & [1 5]
      #     [0 1 5] => [0] & [1 5] =>  [0] & [1]
      #   trunc = True
      #     [0 1 2 3 4 5 6 7] => [0] & [1 2 3 4 5 6] => [0] & [1 3 5]
      #     [0 1 3 5] => [0] & [1 3] => [0] & [1]
      #     [0 1] => [0] & [] => [0]
      # separate_cls = False
      #   [0 1 2 3 4 5 6 7] => [0 2 4 6]
      #   [0 2 4 6] => [0 4]
      #   [0 4] => [0]

      if net_config.separate_cls:
        enc = torch.cat([enc[cls_slice], enc], axis=axis)
      return enc[enc_slice]

    if isinstance(tensor, (tuple, list)):
      return list(map(_pool_func, tensor))
    else:
      return _pool_func(tensor)

  def pool_tensor(self, tensor, mode="mean", stride=(2, 1)):
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
      return None

    net_config = self.net_config
    ndims = tensor.dim()
    if net_config.separate_cls:
      if self.args.truncate_seq:
        tensor = torch.cat([tensor[:, :1], tensor[:, :-1]], dim=1)
      else:
        tensor = torch.cat([tensor[:, :1], tensor], dim=1)

    assert ndims == 2 or ndims == 3 or ndims == 4

    if ndims == 2:
      tensor = tensor[:, None, :, None]
    elif ndims == 3:
      tensor = tensor[:, None, :, :]

    if mode == "mean":
      tensor = F.avg_pool2d(
          tensor, stride, stride=stride, ceil_mode=True)
    elif mode == "max":
      tensor = F.max_pool2d(
          tensor, stride, stride=stride, ceil_mode=True)
    elif mode == "min":
      tensor = -F.max_pool2d(
          -tensor, stride, stride=stride, ceil_mode=True)
    else:
      raise NotImplementedError
    if ndims == 2:
      tensor = tensor.squeeze(-1).squeeze(1)
    elif ndims == 3:
      tensor = tensor.squeeze(1)

    return tensor

  def pre_attn_pooling(self, output, attn_struct):
    pos_enc, seg_mat, input_mask, attn_mask, func_mask = attn_struct
    net_config = self.net_config
    ret_dict = {}
    if net_config.pool_q_only:
      if self.args.attn_type == "factorized":
        pos_enc = self.stride_pool(pos_enc[:2], 0) + pos_enc[2:]
      seg_mat = self.stride_pool(seg_mat, 1)
      func_mask = self.stride_pool(func_mask, 0)
      output = self.pool_tensor(output, mode=net_config.pooling_type)
    else:
      self.delta *= 2
      if self.args.attn_type == "factorized":
        pos_enc = self.stride_pool(pos_enc, 0)
      seg_mat = self.stride_pool(seg_mat, 1)
      seg_mat = self.stride_pool(seg_mat, 2)
      func_mask = self.stride_pool(func_mask, 1)
      func_mask = self.stride_pool(func_mask, 2)
      input_mask = self.pool_tensor(input_mask, mode="min")
      output = self.pool_tensor(output, mode=net_config.pooling_type)
    attn_mask = self.get_attn_mask(input_mask)
    attn_struct = (pos_enc, seg_mat, input_mask, attn_mask, func_mask)
    return output, attn_struct, ret_dict

  def post_attn_pooling(self, attn_struct):
    net_config = self.net_config
    pos_enc, seg_mat, input_mask, attn_mask, func_mask = attn_struct

    if net_config.pool_q_only:
      self.delta *= 2
      if self.args.attn_type == "factorized":
        pos_enc = pos_enc[:2] + self.stride_pool(pos_enc[2:], 0)
      seg_mat = self.stride_pool(seg_mat, 2)
      func_mask = self.stride_pool(func_mask, 1)
      input_mask = self.pool_tensor(input_mask, mode="min")
    attn_mask = self.get_attn_mask(input_mask)
    attn_struct = (pos_enc, seg_mat, input_mask, attn_mask, func_mask)
    return attn_struct
