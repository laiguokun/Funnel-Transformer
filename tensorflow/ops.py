"""Common operations used to construct model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS


INF = 1e6
EPS = 1e-9


###############################################################################
##### Utils
###############################################################################
def safe_precision(func):
  """Safe precision decorator."""
  @functools.wraps(func)
  def wrapped_func(inputs, *args, **kwargs):
    """Turn inputs into float32 for computation."""
    if inputs.dtype != tf.float32:
      fp32_inputs = tf.cast(inputs, tf.float32)
    else:
      fp32_inputs = inputs

    output = func(fp32_inputs, *args, **kwargs)

    if output.dtype != inputs.dtype:
      output = tf.cast(output, inputs.dtype)

    return output

  return wrapped_func


def get_einsum_prefix(ndims, einsum_symbols=None):
  if einsum_symbols is None:
    einsum_symbols = ["u", "v", "w", "x", "y", "z"]
  assert ndims <= len(einsum_symbols)
  einsum_prefix = ""
  for i in range(ndims):
    einsum_prefix += einsum_symbols[i]

  return einsum_prefix


def update_ret_dict(tgt, src, prefix=None):
  if prefix is None:
    tgt.update(src)
  else:
    for k, v in src.items():
      tgt["{}/{}".format(prefix, k)] = v

  return tgt


###############################################################################
##### Common ops
###############################################################################
safe_softmax = safe_precision(tf.nn.softmax)


def embedding_lookup(x, n_embed, d_embed, initializer, lookup_table=None,
                     use_tpu=True, scope="embedding", reuse=None,
                     dtype=tf.float32):
  """tpu and gpu embedding_lookup function."""
  with tf.variable_scope(scope, reuse=reuse):
    if lookup_table is None:
      lookup_table = tf.get_variable("lookup_table", shape=[n_embed, d_embed],
                                     dtype=dtype, initializer=initializer)

    if use_tpu:
      one_hot_idx = tf.one_hot(x, n_embed, dtype=dtype)
      einsum_prefix = get_einsum_prefix(x.shape.ndims)
      einsum_str = "{0}n,nd->{0}d".format(einsum_prefix)
      output = tf.einsum(einsum_str, one_hot_idx, lookup_table)
    else:
      output = tf.nn.embedding_lookup(lookup_table, x)

    return output, lookup_table


def dense(x, out_shape, initializer, inp_shape=None, begin_axis=-1,
          use_bias=True, activation=None, scope="dense", reuse=False):
  """A more flexible dense layer."""
  if isinstance(out_shape, int):
    out_shape = [out_shape]
  if inp_shape is None:
    inp_shape = x.shape.as_list()[begin_axis:]
  elif isinstance(inp_shape, int):
    inp_shape = [inp_shape]

  inp_syms = ["a", "b", "c", "d"]
  out_syms = ["e", "f", "g", "h"]

  prefix = get_einsum_prefix(x.shape.ndims - len(inp_shape))
  inp_str = get_einsum_prefix(len(inp_shape), inp_syms)
  out_str = get_einsum_prefix(len(out_shape), out_syms)

  with tf.variable_scope(scope, reuse=reuse):
    kernel_shape = inp_shape + out_shape
    kernel = tf.get_variable("kernel",
                             kernel_shape,
                             dtype=x.dtype,
                             initializer=initializer)

    output = tf.einsum(
        "{0}{1},{1}{2}->{0}{2}".format(prefix, inp_str, out_str), x, kernel)

    if use_bias:
      bias = tf.get_variable("bias",
                             out_shape,
                             dtype=x.dtype,
                             initializer=tf.zeros_initializer())
      output += bias

    if activation is not None:
      output = activation(output)

  return output


@safe_precision
def layer_norm_op(inputs,
                  norm_shape=None,
                  begin_norm_axis=-1,
                  center=True,
                  scale=True,
                  activation_fn=None,
                  reuse=None,
                  trainable=True,
                  name=None):
  """Custom Layer Normalization layer."""

  if norm_shape is None:
    # If `norm_shape` is not provided, use `begin_norm_axis` to infer
    norm_shape = inputs.shape[begin_norm_axis:]
  elif isinstance(norm_shape, int):
    # If `norm_shape` is provided as int, convert it to list
    norm_shape = [norm_shape]

  with tf.variable_scope(name, "layer_norm", [inputs], reuse=reuse):
    inputs_rank = inputs.shape.ndims
    if inputs_rank is None:
      raise ValueError("Inputs %s has undefined rank." % inputs.name)
    dtype = inputs.dtype.base_dtype
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = tf.get_variable(
          "beta",
          shape=norm_shape,
          dtype=dtype,
          initializer=tf.zeros_initializer(),
          trainable=trainable)
    if scale:
      gamma = tf.get_variable(
          "gamma",
          shape=norm_shape,
          dtype=dtype,
          initializer=tf.ones_initializer(),
          trainable=trainable)
    # By default, compute the moments across all the dimensions except the one
    # with index 0.
    norm_axes = list(range(inputs_rank - len(norm_shape), inputs_rank))
    mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    # Note that epsilon must be increased for float16 due to the limited
    # representable range.
    variance_epsilon = 1e-8 if dtype != tf.float16 else 1e-3
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs.shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def dropout_op(tensor, rate, training, *args, **kwargs):
  kwargs["dtype"] = tensor.dtype
  dropout_func = tf.keras.layers.Dropout(rate, *args, **kwargs)
  return dropout_func(tensor, training=training)


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_type):
  """Get the corresponding activation function from string."""
  if activation_type == "relu":
    activation = tf.nn.relu
  elif activation_type == "gelu":
    activation = gelu
  elif activation_type == "tanh":
    activation = tf.tanh
  else:
    raise ValueError("Unsupported activation type {}".format(activation_type))

  return activation


###############################################################################
##### Trasnformer ops
###############################################################################
def residual_and_layer_norm(residual, hidden, norm_shape=None):
  """Perform residual & layer normalization."""
  ret_dict = {}

  if residual is not None:
    output = hidden + residual
  else:
    output = hidden

  output = layer_norm_op(output, norm_shape=norm_shape, name="layer_norm")

  return output, ret_dict


def positionwise_ffn(inp, d_model, d_inner, dropout, dropact, initializer,
                     activation_type="gelu", scope="ff", is_training=True,
                     reuse=None):
  """Position-wise Feed-forward Network."""
  ret_dict = {}

  activation = get_activation(activation_type)

  output = inp
  with tf.variable_scope(scope, reuse=reuse):
    # mlp part
    output = dense(output, d_inner, inp_shape=d_model, activation=activation,
                   initializer=initializer, scope="layer_1")
    output = dropout_op(output, dropact, training=is_training, name="drop_1")
    output = dense(output, d_model, initializer=initializer, inp_shape=d_inner,
                   scope="layer_2")
    output = dropout_op(output, dropout, training=is_training, name="drop_2")

    # post ffn process
    output, res_lnorm_dict = residual_and_layer_norm(inp, output,
                                                     norm_shape=d_model)

    # add to monitor dict
    ret_dict = update_ret_dict(ret_dict, res_lnorm_dict)

  return output, ret_dict


def rel_attn_core(
    d_model, n_head, d_head, q, k, v, pos_enc, seg_mat, attn_mask, attn_bias,
    dropatt, is_training, initializer, func_mask=None,
    rel_attn_type="factorized"):
  """Core relative positional attention operations."""

  ret_dict = {}
  tf_float = q.dtype

  q_head = dense(q, out_shape=[n_head, d_head], inp_shape=d_model,
                 initializer=initializer, scope="q", use_bias=False)
  k_head = dense(k, out_shape=[n_head, d_head], inp_shape=d_model,
                 initializer=initializer, scope="k")
  v_head = dense(v, out_shape=[n_head, d_head], inp_shape=d_model,
                 initializer=initializer, scope="v")

  # scale `q_head`
  scale = tf.cast(1.0 / np.sqrt(d_head), tf_float)
  q_head = q_head * scale

  # content based attention score
  r_w_bias = tf.get_variable("r_w_bias", [n_head, d_head],
                             dtype=tf_float, initializer=initializer)
  content_bias = tf.einsum("...ind,...jnd->...nij",
                           q_head + r_w_bias * scale, k_head)

  # position based attention score
  if pos_enc is None:
    pos_bias = 0
  else:
    ##### Utilize the decomposed version when using TPU #####
    if rel_attn_type == "factorized":
      if FLAGS.verbose:
        tf.logging.info("Compute rel-pos attn with factorized implementation.")
      pos_bias = rel_pos_bias(q_head, pos_enc, d_model, n_head, d_head,
                              initializer, func_mask=func_mask, dtype=tf_float)
    elif rel_attn_type == "rel_shift":
      if FLAGS.verbose:
        tf.logging.info("Compute rel-pos attn with rel-shift implementation.")
      klen = tf.shape(content_bias)[-1]
      pos_bias = rel_pos_bias_gpu(q_head, pos_enc, d_model, n_head, d_head,
                                  klen, initializer, func_mask=func_mask,
                                  dtype=tf_float)
    else:
      raise NotImplementedError

  # segment based attention score
  if seg_mat is None:
    seg_bias = 0
  else:
    if FLAGS.verbose:
      tf.logging.info("Compute rel-seg attn.")
    seg_bias = rel_seg_bias(q_head, seg_mat, n_head, d_head, initializer,
                            func_mask=func_mask, dtype=tf_float)

  # merge attention scores
  attn_score = content_bias + pos_bias + seg_bias

  # add extra attention score if provided
  if attn_bias is not None:
    if FLAGS.verbose:
      tf.logging.info("Attention bias shape: %s", attn_bias.shape)
    attn_score += attn_bias * scale

  # perform masking
  if attn_mask is not None:
    if FLAGS.verbose:
      tf.logging.info("Attention mask shape: %s", attn_mask.shape)
    ret_dict["attn_mask"] = attn_mask
    attn_score = attn_score - INF * attn_mask

  # attention probability
  attn_prob = safe_softmax(attn_score, -1)
  ret_dict["attn_prob"] = attn_prob
  attn_prob = dropout_op(attn_prob, dropatt, training=is_training)

  # attention output
  attn_vec = tf.einsum("...nij,...jnd->...ind", attn_prob, v_head)

  # things to monitor in attention
  ret_dict["content_bias"] = content_bias
  if pos_enc is not None:
    ret_dict["pos_bias"] = pos_bias
  if seg_mat is not None:
    ret_dict["seg_bias"] = seg_bias

  return attn_vec, ret_dict


def rel_multihead_attn(q, k, v, pos_enc, seg_mat, attn_mask, d_model, n_head,
                       d_head, dropout, dropatt, is_training, initializer,
                       attn_bias=None, func_mask=None, scope="rel_attn",
                       reuse=None, rel_attn_type="factorized"):
  """Multi-head attention with relative positional encoding."""

  ret_dict = {}

  with tf.variable_scope(scope, reuse=reuse) as scope:
    # attention core
    attn_vec, attn_core_dict = rel_attn_core(
        d_model, n_head, d_head, q, k, v, pos_enc, seg_mat, attn_mask,
        attn_bias, dropatt, is_training, initializer, func_mask=func_mask,
        rel_attn_type=rel_attn_type)

    # post projection
    attn_out = dense(attn_vec, d_model, initializer=initializer,
                     inp_shape=[n_head, d_head], scope="o")
    attn_out = dropout_op(attn_out, dropout, training=is_training)

    # residual + layer normalization
    output, post_dict = residual_and_layer_norm(q, attn_out,
                                                norm_shape=d_model)

    # things to monitor
    ret_dict = update_ret_dict(ret_dict, attn_core_dict)
    ret_dict = update_ret_dict(ret_dict, post_dict)

  return output, ret_dict


###############################################################################
##### relative positional attention ops
###############################################################################
def rel_shift(x, row_dim, klen=-1, shift=1):
  """Perform relative shift to form the relative attention score."""
  ndims = x.shape.ndims
  x_shape = tf.shape(x)

  # Deal with negative indexing
  if row_dim < 0:
    row_dim = ndims + row_dim
  assert row_dim >= 0

  # Assume `col_dim` = `row_dim + 1`
  col_dim = row_dim + 1
  assert col_dim < ndims

  tgt_shape_1, slice_begin_1, slice_len_1 = [], [], []
  tgt_shape_2, slice_begin_2, slice_len_2 = [], [], []
  for i in range(ndims):
    slice_len_1.append(-1)
    slice_begin_2.append(0)

    if i == row_dim:
      tgt_shape_1.append(x_shape[col_dim])
      tgt_shape_2.append(x_shape[row_dim])
      slice_begin_1.append(shift)
      slice_len_2.append(-1)
    elif i == col_dim:
      tgt_shape_1.append(x_shape[row_dim])
      tgt_shape_2.append(x_shape[col_dim] - shift)
      slice_begin_1.append(0)
      slice_len_2.append(klen)
    else:
      tgt_shape_1.append(x_shape[i])
      tgt_shape_2.append(x_shape[i])
      slice_begin_1.append(0)
      slice_len_2.append(-1)

  x = tf.reshape(x, tgt_shape_1)
  x = tf.slice(x, slice_begin_1, slice_len_1)
  x = tf.reshape(x, tgt_shape_2)
  x = tf.slice(x, slice_begin_2, slice_len_2)

  return x


def rel_pos_bias_gpu(q_head, pos_enc, d_model, n_head, d_head, klen,
                     initializer, func_mask=None, dtype=tf.float32):
  """Relative attention positional bias via relative shift for GPU."""
  enc, shift = pos_enc
  scale = tf.cast(1.0 / np.sqrt(d_head), dtype)

  # parameters
  r_r_bias = tf.get_variable("r_r_bias", [n_head, d_head],
                             dtype=dtype, initializer=initializer)
  r_head = dense(enc, out_shape=[n_head, d_head], inp_shape=d_model,
                 initializer=initializer, scope="r", use_bias=False)

  # [B x T x N x D]
  pos_bias = tf.einsum("...inh,jnh->...nij", q_head + r_r_bias * scale,
                       r_head)
  pos_bias = rel_shift(pos_bias, -2, klen, shift)

  if func_mask is not None:
    pos_bias *= func_mask

  return pos_bias


def rel_pos_bias(q_head, pos_enc, d_model, n_head, d_head, initializer,
                 func_mask=None, dtype=tf.float32):
  """Relative attention positional bias."""
  # [(B) x T x D]
  enc_q_1, enc_q_2, enc_k_1, enc_k_2 = pos_enc

  # parameters
  r_r_bias = tf.get_variable("r_r_bias", [n_head, d_head],
                             dtype=dtype, initializer=initializer)
  r_kernel = tf.get_variable("r/kernel", [d_model, n_head, d_head],
                             dtype=dtype, initializer=initializer)

  scale = tf.cast(1.0 / np.sqrt(d_head), dtype)
  # [B x T x N x D]
  q_head_r = tf.einsum("...inh,dnh->...ind", q_head + r_r_bias * scale,
                       r_kernel)

  # [(B) x T x N x D]
  q_head_r_1 = q_head_r * tf.expand_dims(enc_q_1, -2)
  q_head_r_2 = q_head_r * tf.expand_dims(enc_q_2, -2)
  # tf.logging.info("%s, %s, %s", q_head_r, q_head_r_1, q_head_r_2)

  # [(B) x T x N x D]
  prefix_k = get_einsum_prefix(enc_k_1.shape.ndims - 2)
  einsum_str = "...ind,{0}jd->...nij".format(prefix_k)
  pos_bias = (tf.einsum(einsum_str, q_head_r_1, enc_k_1) +
              tf.einsum(einsum_str, q_head_r_2, enc_k_2))

  if func_mask is not None:
    pos_bias *= func_mask

  return pos_bias


def rel_seg_bias(q_head, seg_mat, n_head, d_head, initializer, func_mask=None,
                 dtype=tf.float32):
  """Relative attention segmentation bias."""
  # Expand seg_mat: [... x N x T x T]
  tgt_shape = []
  for i in range(seg_mat.shape.ndims):
    tgt_shape.append(tf.shape(seg_mat)[i])
  tgt_shape.insert(-2, n_head)
  seg_mat = tf.expand_dims(seg_mat, -3)

  # Compute same / diff biases
  r_s_bias = tf.get_variable("r_s_bias", [n_head, d_head],
                             dtype=dtype, initializer=initializer)
  seg_embed = tf.get_variable("seg_embed", [2, n_head, d_head],
                              dtype=dtype, initializer=initializer)

  scale = tf.cast(1.0 / np.sqrt(d_head), dtype)
  q_head_s = q_head + r_s_bias * scale
  # [... x N x T x 2]
  seg_biases = tf.einsum("...inh,snh->...nis", q_head_s, seg_embed)

  # Split into `diff` & `same`: [... x N x T x 1]
  seg_bias_diff, seg_bias_same = tf.split(seg_biases, 2, axis=-1)

  # Broadcast
  seg_mat = tf.broadcast_to(seg_mat, tgt_shape)
  seg_bias_diff = tf.broadcast_to(seg_bias_diff, tgt_shape)
  seg_bias_same = tf.broadcast_to(seg_bias_same, tgt_shape)
  seg_bias = tf.where(seg_mat, seg_bias_same, seg_bias_diff)

  if func_mask is not None:
    seg_bias *= func_mask

  return seg_bias


def seg_id_to_mat(seg_q, seg_k):
  """Convert `seg_id` to `seg_mat`."""
  if seg_q is None or seg_k is None:
    return None

  seg_mat = tf.equal(tf.expand_dims(seg_q, -1), tf.expand_dims(seg_k, -2))

  # Treat [cls] as in the same segment as both A & B
  cls_mat = tf.logical_or(
      tf.expand_dims(tf.equal(seg_q, tf.constant([FLAGS.seg_id_cls])), -1),
      tf.expand_dims(tf.equal(seg_k, tf.constant([FLAGS.seg_id_cls])), -2))
  seg_mat = tf.logical_or(cls_mat, seg_mat)

  return seg_mat


def get_pos_enc(pos_id_q, pos_id_k, d_model, dropout, is_training,
                clamp_len=-1, dtype=tf.float32):
  """Create inputs related to relative position encoding."""
  pos_id_q = tf.cast(pos_id_q, dtype)
  pos_id_k = tf.cast(pos_id_k, dtype)
  if clamp_len > 0:
    pos_id_q = tf.clamp(pos_id_q, -clamp_len, clamp_len)
    pos_id_k = tf.clamp(pos_id_k, -clamp_len, clamp_len)

  d_model_half = d_model // 2
  freq_seq = tf.cast(tf.range(0, d_model_half, 1.0), dtype=dtype)
  inv_freq = 1 / (10000 ** (freq_seq / d_model_half))

  sinusoid_q = tf.einsum("...i,d->...id", pos_id_q, inv_freq)
  sinusoid_k = tf.einsum("...i,d->...id", pos_id_k, inv_freq)

  sin_enc_q = tf.sin(sinusoid_q)
  cos_enc_q = tf.cos(sinusoid_q)
  sin_enc_q = dropout_op(sin_enc_q, dropout, training=is_training)
  cos_enc_q = dropout_op(cos_enc_q, dropout, training=is_training)

  sin_enc_k = tf.sin(sinusoid_k)
  cos_enc_k = tf.cos(sinusoid_k)

  enc_q_1 = tf.concat([sin_enc_q, sin_enc_q], axis=-1)
  enc_k_1 = tf.concat([cos_enc_k, sin_enc_k], axis=-1)

  enc_q_2 = tf.concat([cos_enc_q, cos_enc_q], axis=-1)
  enc_k_2 = tf.concat([-sin_enc_k, cos_enc_k], axis=-1)

  return [enc_q_1, enc_q_2, enc_k_1, enc_k_2]


def get_pos_enc_gpu(rel_pos_id, d_model, dropout, is_training,
                    clamp_len=-1, dtype=tf.float32):
  """Create inputs related to relative position encoding."""
  rel_pos_id = tf.cast(rel_pos_id, dtype)
  if clamp_len > 0:
    rel_pos_id = tf.clamp(rel_pos_id, -clamp_len, clamp_len)

  d_model_half = d_model // 2
  freq_seq = tf.cast(tf.range(0, d_model_half, 1.0), dtype=dtype)
  inv_freq = 1 / (10000 ** (freq_seq / d_model_half))

  sinusoid = tf.einsum("...i,d->...id", rel_pos_id, inv_freq)
  sin_enc = tf.sin(sinusoid)
  cos_enc = tf.cos(sinusoid)
  sin_enc = dropout_op(sin_enc, dropout, training=is_training)
  cos_enc = dropout_op(cos_enc, dropout, training=is_training)
  pos_enc = tf.concat([sin_enc, cos_enc], axis=-1)

  return pos_enc
