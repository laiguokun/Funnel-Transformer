# Lint as: python3
"""Funnel-Transformer Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
import tensorflow.compat.v1 as tf

import ops


##### Model configuration
flags.DEFINE_string("overwrite_keys", default="",
                    help="Comma separated keys to indicate model configs that "
                    "will always be overwritten by the FLAGS values.")

# Size
flags.DEFINE_string("block_size", default="3_3_3",
                    help="Depth of blocks with potential parameter sharing.")
flags.DEFINE_integer("d_model", default=512,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=512,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=8,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=64,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=2048,
                     help="Dimension of inner hidden size in FFN.")

# Dropouts
flags.DEFINE_float("dropout", default=0.1,
                   help="Model dropout.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout.")
flags.DEFINE_float("dropact", default=0.0,
                   help="Activation dropout.")

flags.DEFINE_string("ff_activation", default="gelu",
                    help="Activation type used in position-wise feed-forward.")
flags.DEFINE_string("rel_attn_type", default="factorized",
                    help="Type of the relative attention.")

##### Parameter initialization
flags.DEFINE_enum("init", default="truncated_normal",
                  enum_values=["normal", "uniform", "truncated_normal"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

##### Funnel-Transformer specific
# encoder
flags.DEFINE_enum("pooling_type", default="mean",
                  enum_values=["mean", "max"],
                  help="choose from [max, mean].")
flags.DEFINE_integer("pooling_size", default=2,
                     help="Kernel size for max and mean pooling.")
flags.DEFINE_bool("pool_q_only", default=True,
                  help="Only perform pooling on query")
flags.DEFINE_bool("separate_cls", default=True,
                  help="Whether to isolate the [cls]")

# decoder
flags.DEFINE_string("decoder_size", default="2",
                    help="Size configuration of the decoder.")

# This can be directly modifiable at runtime
flags.DEFINE_bool("truncate_seq", default=True,
                  help="Truncate the last few tokens according to the max "
                  "stride in the network to make separate [cls] efficient.")

FLAGS = flags.FLAGS


INF = 1e8


class ModelConfig(object):
  """ModelConfig contains fixed hyperparameters of a Funnel-Transformer."""

  keys = ["block_size", "vocab_size", "d_embed", "d_model", "n_head", "d_head",
          "d_inner", "ff_activation", "dropout", "dropatt", "dropact",
          "init", "init_std", "init_range", "rel_attn_type", "separate_cls",
          "pooling_type", "pooling_size", "pool_q_only", "decoder_size"]

  def __init__(self, block_size, vocab_size, d_embed, d_model, n_head,
               d_head, d_inner, dropout, dropatt, dropact, ff_activation,
               init="truncated_normal", init_std=0.02, init_range=0.1,
               rel_attn_type="factorized", separate_cls=True,
               pooling_type="mean", pooling_size=2, pool_q_only=True,
               decoder_size="0"):

    """Initialize model config."""
    assert vocab_size == FLAGS.vocab_size, "Vocabulary size does not match."

    self.vocab_size = vocab_size
    self.d_embed = d_embed
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner

    self.dropout = dropout
    self.dropatt = dropatt
    self.dropact = dropact

    self.ff_activation = ff_activation
    self.init = init
    self.init_std = init_std
    self.init_range = init_range

    self.rel_attn_type = rel_attn_type

    self.block_size = block_size
    self.block_depth = []
    self.block_param_size = []
    self.block_repeat_size = []
    for cur_block_size in block_size.split("_"):
      cur_block_size = ModelConfig.parse_depth_string(cur_block_size)
      self.block_depth.append(cur_block_size[0] * cur_block_size[1])
      self.block_param_size.append(cur_block_size[0])
      self.block_repeat_size.append(cur_block_size[1])
    self.n_block = len(self.block_depth)

    assert not (self.n_block == 1 and decoder_size != "0"), \
        "Models with only 1 block does NOT need a decoder."
    self.decoder_size = decoder_size
    decoder_size = ModelConfig.parse_depth_string(decoder_size)
    self.decoder_depth = decoder_size[0] * decoder_size[1]
    self.decoder_param_size = decoder_size[0]
    self.decoder_repeat_size = decoder_size[1]

    self.pooling_type = pooling_type
    self.pooling_size = pooling_size
    self.pool_q_only = pool_q_only

    self.separate_cls = separate_cls

  @staticmethod
  def parse_depth_string(depth_str):
    depth_config = depth_str.split("x")
    if len(depth_config) == 1:
      depth_config.append(1)
    assert len(depth_config) == 2, "Require two-element depth config."

    return list(map(int, depth_config))

  @staticmethod
  def overwrite_args(args):
    """Overwrite args."""
    # Merge loaded config and FLAGS
    for key in list(set(ModelConfig.keys) & set(args.keys())):
      overwrite_keys = set(FLAGS.overwrite_keys.split(","))
      if key in overwrite_keys:
        args[key] = getattr(FLAGS, key)
      else:
        setattr(FLAGS, key, args[key])

    return args

  @staticmethod
  def init_from_text(file_path, sep_symbol=None):
    """Initialize ModelConfig from a text file."""
    tf.logging.info("Initialize ModelConfig from text file %s.", file_path)
    args = {}
    with tf.io.gfile.GFile(file_path) as f:
      for line in f:
        k, v = line.strip().split(sep_symbol)
        if k in ModelConfig.keys:
          args[k] = v
        else:
          tf.logging.warning("Unused key %s", k)

    args = ModelConfig.overwrite_args(args)
    net_config = ModelConfig(**args)

    return net_config

  @staticmethod
  def init_from_json(file_path):
    """Initialize ModelConfig from a json file."""
    tf.logging.info("Initialize ModelConfig from json file %s.", file_path)
    with tf.io.gfile.GFile(file_path) as f:
      json_data = json.load(f)
      if (not getattr(FLAGS, "use_tpu", False) and
          json_data["rel_attn_type"] == "factorized"):
        json_data["rel_attn_type"] = "rel_shift"
        tf.logging.info("Change rel_attn_type to `rel_shift` for non-TPU env.")

    json_data = ModelConfig.overwrite_args(json_data)
    net_config = ModelConfig(**json_data)

    return net_config

  @staticmethod
  def init_from_flags():
    """Initialize ModelConfig from FLAGS."""
    tf.logging.info("Initialize ModelConfig from FLAGS.")
    args = {}
    for key in ModelConfig.keys:
      args[key] = getattr(FLAGS, key)

    return ModelConfig(**args)

  def to_json(self, json_path):
    """Save ModelConfig to a json file."""
    tf.logging.info("Save ModelConfig to json file %s.", json_path)
    json_data = {}
    for key in ModelConfig.keys:
      json_data[key] = getattr(self, key)

    json_dir = os.path.dirname(json_path)
    if not tf.io.gfile.exists(json_dir):
      tf.io.gfile.makedirs(json_dir)
    with tf.io.gfile.GFile(json_path, "w") as f:
      json.dump(json_data, f, indent=4, sort_keys=True)


class FunnelTFM(object):
  """Funnel-Transformer model used during both pretraining and finetuning."""

  def __init__(self, net_config):
    """Initialize the model with config."""

    self.net_config = net_config
    self.attn_structures = None

  def get_initializer(self):
    """Get variable intializer."""
    net_config = self.net_config
    if net_config.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-net_config.init_range,
          maxval=net_config.init_range,
          seed=None)
    elif net_config.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=net_config.init_std,
          seed=None)
    elif net_config.init == "truncated_normal":
      initializer = tf.initializers.truncated_normal(
          stddev=net_config.init_std,
          seed=None)
    else:
      raise ValueError("Initializer {} not supported".format(net_config.init))
    return initializer

  def get_embedding_table(self, scope="input", dtype=tf.float32):
    """Get the corresponding embeeding table."""
    net_config = self.net_config
    with tf.variable_scope(scope, reuse=True):
      with tf.variable_scope("word_embedding", reuse=True):
        lookup_table = tf.get_variable(
            "lookup_table", [net_config.vocab_size, net_config.d_model],
            dtype=dtype)
    return lookup_table

  def input_embedding(self, inputs, is_training, seg_id=None, pos_id=None,
                      word_embed_table=None, use_tpu=False, scope="input",
                      reuse=tf.AUTO_REUSE, dtype=tf.float32):
    """Turn input ids to input embedding."""

    net_config = self.net_config
    initializer = self.get_initializer()
    ret_dict = {}

    ##### Embedding
    def embed_func(x, pos_id, seg_id):
      """Word embed + Position embed + Segment embed (if provided)."""
      # Word embedding
      embed, word_embed_table = ops.embedding_lookup(
          x=x,
          n_embed=net_config.vocab_size,
          d_embed=net_config.d_embed,
          initializer=initializer,
          use_tpu=use_tpu,
          dtype=dtype,
          scope="word_embedding")

      if net_config.rel_attn_type == "null":
        # Position embedding
        if pos_id is None:
          pos_id = tf.cast(tf.range(tf.shape(x)[-1]), x.dtype)
        pos_emb, _ = ops.embedding_lookup(
            x=pos_id,
            n_embed=512,
            d_embed=net_config.d_embed,
            initializer=initializer,
            use_tpu=use_tpu,
            dtype=dtype,
            scope="position_embedding")
        embed += pos_emb

        # Segment embedding
        if seg_id is not None:
          seg_emb, _ = ops.embedding_lookup(
              x=seg_id % 2,
              n_embed=2,
              d_embed=net_config.d_embed,
              initializer=initializer,
              use_tpu=use_tpu,
              dtype=dtype,
              scope="segment_embedding")
          embed += seg_emb

      return embed, word_embed_table

    with tf.variable_scope(scope, reuse=reuse):
      ##### Input embedding layer normalization and dropout
      word_emb, word_embed_table = embed_func(x=inputs,
                                              pos_id=pos_id,
                                              seg_id=seg_id)
      word_emb = ops.layer_norm_op(word_emb, norm_shape=[net_config.d_embed])

      output = ops.dropout_op(word_emb,
                              net_config.dropout,
                              training=is_training)

    return output, word_embed_table, ret_dict

  def input_projection(self, input_embed):
    """Project input embedding to a proper dimension if needed."""
    net_config = self.net_config
    initializer = self.get_initializer()
    ret_dict = {}

    output = input_embed
    if net_config.d_embed != net_config.d_model:
      tf.logging.info("Project input embedding: %s -> %s",
                      net_config.d_embed, net_config.d_model)
      output = ops.dense(
          output,
          net_config.d_model,
          inp_shape=net_config.d_embed,
          initializer=initializer,
          scope="input_projection")

    return output, ret_dict

  def tfmxl_layer(self, q, k, v, pos_enc, seg_mat, attn_mask, is_training,
                  func_mask=None, attn_bias=None):

    """Single transformer-xl layer."""
    net_config = self.net_config
    initializer = self.get_initializer()

    ret_dict = {}
    output, attn_dict = ops.rel_multihead_attn(
        q=q,
        k=k,
        v=v,
        pos_enc=pos_enc,
        seg_mat=seg_mat,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        d_model=net_config.d_model,
        n_head=net_config.n_head,
        d_head=net_config.d_head,
        dropout=net_config.dropout,
        dropatt=net_config.dropatt,
        is_training=is_training,
        initializer=initializer,
        func_mask=func_mask,
        rel_attn_type=net_config.rel_attn_type)

    output, pffn_dict = ops.positionwise_ffn(
        inp=output,
        d_model=net_config.d_model,
        d_inner=net_config.d_inner,
        activation_type=net_config.ff_activation,
        dropout=net_config.dropout,
        dropact=net_config.dropact,
        is_training=is_training,
        initializer=initializer)

    ops.update_ret_dict(ret_dict, attn_dict, "attn")
    ops.update_ret_dict(ret_dict, pffn_dict, "pffn")
    return output, ret_dict

  def encoder(self,
              input_embed,
              is_training,
              seg_id=None,
              pos_id=None,
              input_mask=None,
              scope="encoder",
              reuse=tf.AUTO_REUSE):
    """Encoder of the Funnel-Transformer."""
    net_config = self.net_config
    ret_dict = {}

    with tf.variable_scope(scope, reuse=reuse):

      ##### Input projection
      output, _ = self.input_projection(input_embed)

      ##### Encoder layers
      hiddens = []
      layer_dict = {}
      for block_idx in range(net_config.n_block):
        # prepare structures for relative attention
        if block_idx == 0:
          pos_enc, seg_mat, func_mask = self.init_attn_structures(
              input_embed, seg_id, pos_id, is_training)
        else:
          pool_ret = self.pre_attn_pooling(
              output, pos_enc, seg_mat, input_mask, func_mask, block_idx,
              is_training)
          pooled_out, pos_enc, seg_mat, input_mask, func_mask = pool_ret
        attn_mask = None if input_mask is None else input_mask[:, None, None]

        for param_idx in range(net_config.block_param_size[block_idx]):
          ##### current layer idx
          layer_idx = sum(net_config.block_param_size[:block_idx]) + param_idx
          with tf.variable_scope("layer_{}".format(layer_idx), reuse=reuse):
            cur_repeat_size = net_config.block_repeat_size[block_idx]
            for repeat_idx in range(cur_repeat_size):
              sub_idx = (param_idx * cur_repeat_size + repeat_idx)
              do_pooling = block_idx > 0 and sub_idx == 0

              # prepare inputs to the current layer
              if do_pooling:
                if net_config.pool_q_only:
                  q = pooled_out
                  k = v = output
                else:
                  q = k = v = pooled_out
              else:
                q = k = v = output

              # attention layer
              output, layer_dict = self.tfmxl_layer(
                  q=q,
                  k=k,
                  v=v,
                  pos_enc=pos_enc,
                  seg_mat=seg_mat,
                  attn_mask=attn_mask,
                  is_training=is_training,
                  func_mask=func_mask)

              # post-attention pooling
              if do_pooling:
                pool_ret = self.post_attn_pooling(
                    pos_enc, seg_mat, input_mask, func_mask, block_idx,
                    is_training)
                pos_enc, seg_mat, input_mask, func_mask = pool_ret
                attn_mask = None if input_mask is None \
                    else input_mask[:, None, None]

              # update ret dict
              hiddens.append(output)
              prefix = "block_{}/layer_{}/repeat_{}".format(
                  block_idx, layer_idx, repeat_idx)
              ops.update_ret_dict(ret_dict, layer_dict, prefix)

    return output, hiddens, ret_dict

  ##############################################
  ##### Pooling related section            #####
  ##############################################
  def stride_pool(self, tensor, axis):
    """Perform pooling by stride slicing the tensor along the axis."""
    if tensor is None:
      return None

    net_config = self.net_config
    pool_size = net_config.pooling_size
    if isinstance(tensor, (tuple, list)):
      ndims = tensor[0].shape.ndims
    else:
      ndims = tensor.shape.ndims
    axis = axis % ndims

    slice_list = []
    for i in range(ndims):
      if i == axis:
        if FLAGS.separate_cls:
          if FLAGS.truncate_seq:
            slice_list.append(slice(1, -1, pool_size))
          else:
            slice_list.append(slice(1, None, pool_size))
        else:
          slice_list.append(slice(None, None, pool_size))
        break
      else:
        slice_list.append(slice(None))

    if net_config.separate_cls:
      cls_slice_list = []
      for i in range(ndims):
        if i == axis:
          cls_slice_list.append(slice(None, 1))
          break
        else:
          cls_slice_list.append(slice(None))

    def _pool_func(origin):
      pooled = origin[slice_list]
      if net_config.separate_cls:
        pooled = tf.concat([origin[cls_slice_list], pooled], axis=axis)
      return pooled

    if isinstance(tensor, (tuple, list)):
      return list(map(_pool_func, tensor))
    else:
      return _pool_func(tensor)

  def pool_tensor(self, tensor, mode="mean"):
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
      return None

    net_config = self.net_config
    ndims = tensor.shape.ndims
    pool_size = net_config.pooling_size

    if net_config.separate_cls:
      cls_tensor = tensor[:, :1]
      if FLAGS.truncate_seq:
        pooled = tensor[:, 1:-1]
      else:
        pooled = tensor[:, 1:]
    else:
      pooled = tensor

    if ndims == 2: pooled = pooled[:, :, None]
    if mode == "mean":
      pooled = tf.nn.avg_pool1d(
          pooled,
          ksize=pool_size,
          strides=pool_size,
          data_format="NWC",
          padding="SAME")
    elif mode == "max":
      pooled = tf.nn.max_pool1d(
          pooled,
          ksize=pool_size,
          strides=pool_size,
          data_format="NWC",
          padding="SAME")
    elif mode == "min":
      pooled = -tf.nn.max_pool1d(
          -pooled,
          ksize=pool_size,
          strides=pool_size,
          data_format="NWC",
          padding="SAME")
    else:
      raise NotImplementedError
    if ndims == 2: pooled = tf.squeeze(pooled, 2)

    if net_config.separate_cls:
      pooled = tf.concat([cls_tensor, pooled], axis=1)

    return pooled

  def rel_shift_pos_enc(self, q_len, q_pow, k_len, k_pow, is_training,
                        dtype):
    """Get positional encoding under the relative shift implementation."""
    net_config = self.net_config
    pool_size = net_config.pooling_size

    q_stride = pool_size ** q_pow
    k_stride = pool_size ** k_pow
    shift = q_stride // k_stride

    min_pos_k = 1 - k_stride
    max_pos_k = min_pos_k + (k_len - 1) * k_stride
    min_pos_q = 1 - q_stride

    ref_point = min_pos_q - min_pos_k
    num_to_remove = shift * q_len
    max_dist = ref_point + num_to_remove * k_stride
    min_dist = min_pos_q - max_pos_k
    rel_pos_id = tf.range(max_dist, min_dist - 1, -k_stride)

    enc = ops.get_pos_enc_gpu(
        rel_pos_id,
        net_config.d_model,
        net_config.dropout,
        is_training=is_training,
        dtype=dtype)

    pos_enc = (enc, shift)

    return pos_enc

  def init_attn_structures(self, hidden, seg_id, pos_id, is_training):
    """Initialize extra structures needed for attention."""
    net_config = self.net_config
    if net_config.rel_attn_type == "null":
      self.attn_structures = (None, None, None)
    else:
      if self.attn_structures is None:
        seq_len = tf.shape(hidden)[1]

        if net_config.rel_attn_type == "factorized":
          if pos_id is None:
            half_len = tf.cast(seq_len // 2, tf.float32)
            pos_id = tf.range(-half_len, half_len, 1.0)
          pos_enc = ops.get_pos_enc(
              pos_id,
              pos_id,
              net_config.d_model,
              net_config.dropout,
              is_training=is_training,
              dtype=hidden.dtype)
        elif net_config.rel_attn_type == "rel_shift":
          assert pos_id is None
          seq_len_fp = tf.cast(seq_len, tf.float32)
          rel_pos_id = tf.range(seq_len_fp, -seq_len_fp, -1.0)
          enc = ops.get_pos_enc_gpu(
              rel_pos_id,
              net_config.d_model,
              net_config.dropout,
              is_training=is_training,
              dtype=hidden.dtype)
          shift = 1
          pos_enc = (enc, shift)
        else:
          raise NotImplementedError
        seg_mat = ops.seg_id_to_mat(seg_id, seg_id)
        num_real_token = seq_len - 1
        func_mask = tf.pad(
            tf.ones([num_real_token, num_real_token], dtype=hidden.dtype),
            [[1, 0], [1, 0]])

        self.attn_structures = (pos_enc, seg_mat, func_mask)

    return self.attn_structures

  def pre_attn_pooling(self, output, pos_enc, seg_mat, input_mask,
                       func_mask, block_idx, is_training):
    """Perform pooling before the attention layer."""
    net_config = self.net_config
    if net_config.pool_q_only:
      seg_mat = self.stride_pool(seg_mat, 1)
      output = self.pool_tensor(output, mode=net_config.pooling_type)
      func_mask = self.stride_pool(func_mask, 0)
      if pos_enc is not None:
        if net_config.rel_attn_type == "factorized":
          pos_enc = self.stride_pool(pos_enc[:2], 0) + pos_enc[2:]
        elif net_config.rel_attn_type == "rel_shift":
          pos_enc = self.rel_shift_pos_enc(
              q_len=tf.shape(func_mask)[0], q_pow=block_idx,
              k_len=tf.shape(func_mask)[1], k_pow=block_idx-1,
              is_training=is_training, dtype=func_mask.dtype)
        else:
          raise NotImplementedError
    else:
      seg_mat = self.stride_pool(seg_mat, 1)
      seg_mat = self.stride_pool(seg_mat, 2)
      output = self.pool_tensor(output, mode=net_config.pooling_type)
      func_mask = self.stride_pool(func_mask, 0)
      func_mask = self.stride_pool(func_mask, 1)
      input_mask = self.pool_tensor(input_mask, mode="min")
      if pos_enc is not None:
        if net_config.rel_attn_type == "factorized":
          pos_enc = self.stride_pool(pos_enc, 0)
        elif net_config.rel_attn_type == "rel_shift":
          pos_enc = self.rel_shift_pos_enc(
              q_len=tf.shape(func_mask)[0], q_pow=block_idx,
              k_len=tf.shape(func_mask)[1], k_pow=block_idx,
              is_training=is_training, dtype=func_mask.dtype)
        else:
          raise NotImplementedError

    return output, pos_enc, seg_mat, input_mask, func_mask

  def post_attn_pooling(self, pos_enc, seg_mat, input_mask, func_mask,
                        block_idx, is_training):
    """Perform pooling after the attention layer."""
    net_config = self.net_config
    if net_config.pool_q_only:
      seg_mat = self.stride_pool(seg_mat, 2)
      func_mask = self.stride_pool(func_mask, 1)
      input_mask = self.pool_tensor(input_mask, mode="min")
      if pos_enc is not None:
        if net_config.rel_attn_type == "factorized":
          pos_enc = pos_enc[:2] + self.stride_pool(pos_enc[2:], 0)
        elif net_config.rel_attn_type == "rel_shift":
          pos_enc = self.rel_shift_pos_enc(
              q_len=tf.shape(func_mask)[1], q_pow=block_idx,
              k_len=tf.shape(func_mask)[1], k_pow=block_idx,
              is_training=is_training, dtype=func_mask.dtype)
        else:
          raise NotImplementedError

    return pos_enc, seg_mat, input_mask, func_mask

  ##############################################
  ##### Upsampling related section         #####
  ##############################################
  def upsample(self, output, stride, tgt_len):
    """Upsample a hidden state by stride."""
    if stride == 1:
      return output

    net_config = self.net_config

    if net_config.separate_cls:
      cls_output = output[:, :1]
      output = output[:, 1:]

    output = tf.repeat(output, repeats=stride, axis=1)

    if net_config.separate_cls:
      if FLAGS.truncate_seq:
        pad_len = stride - 1
        output = tf.pad(output, [[0, 0], [0, pad_len], [0, 0]])
      else:
        output = output[:, :tgt_len - 1]
      output = tf.concat([cls_output, output], axis=1)

    return output

  def bridge_layer(self, hiddens, input_mask, reuse=tf.AUTO_REUSE):
    """A bridge layer between encoder and decoder."""
    net_config = self.net_config
    ret_dict = {}

    tgt_len = tf.shape(input_mask)[1]
    with tf.variable_scope("upsampling_layer", reuse=reuse):
      # upsample hiddens based on the provided block indices
      upsampled_hids = []
      cum_num_layer = 0
      for block_idx in range(net_config.n_block):
        stride = 2 ** block_idx
        cum_num_layer += (net_config.block_repeat_size[block_idx] *
                          net_config.block_param_size[block_idx])
        layer_idx = cum_num_layer - 1
        upsampled_hid = self.upsample(
            hiddens[layer_idx], stride=stride, tgt_len=tgt_len)
        upsampled_hids.append(upsampled_hid)

      # add residual connection
      upsampled_hidden = upsampled_hids[-1]
      unpooled_hidden = upsampled_hids[0]
      output = upsampled_hidden + unpooled_hidden

    return output, ret_dict

  def decoder(self,
              hiddens,
              is_training,
              input_mask=None,
              seg_id=None,
              pos_id=None,
              scope="decoder",
              reuse=tf.AUTO_REUSE):
    """Decode a compressed sequence into a full sequence."""
    net_config = self.net_config
    ret_dict = {}

    output, bridge_dict = self.bridge_layer(
        hiddens, input_mask, reuse=reuse)
    ops.update_ret_dict(ret_dict, bridge_dict, "bridge")

    if net_config.decoder_depth == 0:
      return output, ret_dict

    # prepare structures for relative attention
    pos_enc, seg_mat, func_mask = self.init_attn_structures(
        output, seg_id, pos_id, is_training)
    attn_mask = None if input_mask is None else input_mask[:, None, None]

    # Decoder layers
    n_enc_param_layer = sum(net_config.block_param_size)
    with tf.variable_scope(scope, reuse=reuse):
      for param_idx in range(net_config.decoder_param_size):
        layer_idx = n_enc_param_layer + param_idx
        with tf.variable_scope("layer_{}".format(layer_idx), reuse=reuse):
          for repeat_idx in range(net_config.decoder_repeat_size):

            output, layer_dict = self.tfmxl_layer(
                q=output,
                k=output,
                v=output,
                pos_enc=pos_enc,
                seg_mat=seg_mat,
                attn_mask=attn_mask,
                is_training=is_training,
                func_mask=func_mask)
            ops.update_ret_dict(
                ret_dict, layer_dict,
                "layer_{}/repeat_{}".format(layer_idx, repeat_idx))

    return output, ret_dict

  ##############################################
  ##### Some commonly used APIS            #####
  ##############################################
  def summarize_sequence(self,
                         hidden,
                         scope="sequnece_summary",
                         reuse=tf.AUTO_REUSE):
    """Summarize hidden sequence into a vector."""
    net_config = self.net_config
    initializer = self.get_initializer()

    with tf.variable_scope(scope, reuse=reuse):
      # use another projection with `tanh` activation
      summary = ops.dense(
          hidden[:, 0],
          net_config.d_model,
          activation=tf.tanh,
          use_bias=True,
          initializer=initializer,
          scope="summary")

    return summary

  def extract_hiddens(self, inputs, is_training, seg_id=None, pos_id=None,
                      input_mask=None, use_decoder=False, use_tpu=False,
                      use_bfloat16=False):
    """Extract hidden states."""
    dtype = tf.float32 if not use_bfloat16 else tf.bfloat16
    input_embed, _, _ = self.input_embedding(
        inputs, is_training, seg_id=seg_id, use_tpu=use_tpu, dtype=dtype)

    output, hiddens, ret_dict = self.encoder(
        input_embed,
        is_training,
        seg_id=seg_id,
        pos_id=pos_id,
        input_mask=input_mask)

    # Decoding
    if use_decoder:
      output, _ = self.decoder(
          hiddens,
          input_mask=input_mask,
          seg_id=seg_id,
          pos_id=pos_id,
          is_training=is_training)

    return output, hiddens, ret_dict

  def get_pooled_output(self, inputs, is_training, seg_id=None, input_mask=None,
                        use_tpu=False, use_bfloat16=False):
    """Get pooled output."""
    output, _, _ = self.extract_hiddens(
        inputs,
        is_training,
        seg_id=seg_id,
        input_mask=input_mask,
        use_tpu=use_tpu,
        use_bfloat16=use_bfloat16)

    summary = self.summarize_sequence(output)

    return summary, output

  def lm_logits(self, hidden, lookup_table=None, mapping=None, scope="lm"):
    """Compute logits for language modeling cross entropy loss."""
    net_config = self.net_config
    initializer = self.get_initializer()

    # Extract relavant hidden states
    if mapping is not None:
      hidden = tf.einsum("...id,...ki->...kd", hidden, mapping)

    # Apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("{}_proj".format(scope)):
      hidden = ops.dense(
          hidden,
          out_shape=net_config.d_embed,
          inp_shape=net_config.d_model,
          activation=ops.get_activation(net_config.ff_activation),
          initializer=initializer)
      hidden = ops.layer_norm_op(hidden, norm_shape=[net_config.d_embed])

    with tf.variable_scope("{}_loss".format(scope)):
      if lookup_table is not None:
        softmax_w = lookup_table
      else:
        softmax_w = tf.get_variable("weight",
                                    [net_config.vocab_size, net_config.d_embed],
                                    dtype=hidden.dtype, initializer=initializer)

      softmax_b = tf.get_variable("bias", [net_config.vocab_size],
                                  dtype=hidden.dtype,
                                  initializer=tf.zeros_initializer())

      logits = tf.einsum("...d,nd->...n", hidden, softmax_w) + softmax_b
      if logits.dtype != tf.float32:
        # Always use float32 for LM loss
        logits = tf.cast(logits, tf.float32)

    return logits

  def lm_loss(self, hidden, target, lookup_table=None, mapping=None,
              return_logits=False, use_tpu=False, scope="lm"):
    """Compute language modeling cross entropy loss."""
    net_config = self.net_config
    logits = self.lm_logits(hidden, lookup_table, mapping, scope)

    if target.shape.as_list() == logits.shape.as_list():
      if use_tpu:
        target = tf.cast(target, logits.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * target, -1)
      else:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                          logits=logits)
    else:
      if use_tpu:
        target = tf.one_hot(target, net_config.vocab_size, dtype=logits.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * target, -1)
      else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                              logits=logits)

    if return_logits:
      return loss, logits
    else:
      return loss

  def classification_loss(self, hidden, labels, n_class, is_training, scope,
                          reuse=tf.AUTO_REUSE, return_logits=False):
    """Get classification loss."""
    net_config = self.net_config
    initializer = self.get_initializer()

    with tf.variable_scope(scope, reuse=reuse):
      hidden = ops.dropout_op(hidden, net_config.dropout, training=is_training)
      logits = ops.dense(
          hidden,
          n_class,
          initializer=initializer,
          scope="logit")

      # Always cast to float32 for softmax & loss
      if logits.dtype != tf.float32:
        logits = tf.cast(logits, tf.float32)

      one_hot_target = tf.one_hot(labels, n_class, dtype=logits.dtype)
      loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)

      if return_logits:
        return loss, logits

      return loss

  def regression_loss(self, hidden, labels, is_training, scope,
                      reuse=tf.AUTO_REUSE, return_logits=False):
    """Get regression loss."""
    net_config = self.net_config
    initializer = self.get_initializer()

    with tf.variable_scope(scope, reuse=reuse):
      hidden = ops.dropout_op(hidden, net_config.dropout, training=is_training)
      logits = ops.dense(
          hidden,
          1,
          initializer=initializer,
          scope="logit")

      # Always cast to float32 for loss
      logits = tf.squeeze(logits, axis=-1)
      if logits.dtype != tf.float32:
        logits = tf.cast(logits, tf.float32)

      loss = tf.square(logits - tf.cast(labels, logits.dtype))

      if return_logits:
        return loss, logits

      return loss

  ##############################################
  ##### APIs related to specific task loss #####
  ##############################################
  def get_mlm_loss(self, target, inputs, is_training, seg_id=None, mapping=None,
                   input_mask=None, use_tpu=False, use_bfloat16=False):
    """Get mlm pretrain output."""
    ret_dict = {}
    net_config = self.net_config

    dtype = tf.float32 if not use_bfloat16 else tf.bfloat16
    input_embed, word_embed_table, emb_dict = self.input_embedding(
        inputs, is_training, seg_id=seg_id, use_tpu=use_tpu, dtype=dtype)
    ops.update_ret_dict(ret_dict, emb_dict, "emb")

    # encoder
    output, hiddens, enc_dict = self.encoder(
        input_embed,
        is_training,
        seg_id=seg_id,
        input_mask=input_mask)
    ops.update_ret_dict(ret_dict, enc_dict, "enc")

    # decoder
    if net_config.n_block > 1:
      output, dec_dict = self.decoder(
          hiddens,
          input_mask=input_mask,
          seg_id=seg_id,
          is_training=is_training)
      ops.update_ret_dict(ret_dict, dec_dict, "dec")

    # mlm loss
    lm_loss, logits = self.lm_loss(
        output,
        target,
        mapping=mapping,
        lookup_table=word_embed_table,
        return_logits=True,
        use_tpu=use_tpu)

    ret_dict["lm_logits"] = logits
    ret_dict["hiddens"] = hiddens

    return lm_loss, ret_dict

  def get_classification_loss(self, labels, inputs, n_class, is_training, scope,
                              seg_id=None, input_mask=None, use_tpu=False,
                              use_bfloat16=False):
    """Classification loss."""
    summary, _ = self.get_pooled_output(inputs,
                                        is_training,
                                        seg_id=seg_id,
                                        input_mask=input_mask,
                                        use_tpu=use_tpu,
                                        use_bfloat16=use_bfloat16)
    cls_loss, logits = self.classification_loss(summary,
                                                labels,
                                                n_class,
                                                is_training,
                                                return_logits=True,
                                                scope=scope)

    return cls_loss, logits

  def get_regression_loss(self, labels, inputs, is_training, scope, seg_id=None,
                          input_mask=None, use_tpu=False, use_bfloat16=False):
    """Regression loss."""
    summary, _ = self.get_pooled_output(inputs,
                                        is_training,
                                        seg_id=seg_id,
                                        input_mask=input_mask,
                                        use_tpu=use_tpu,
                                        use_bfloat16=use_bfloat16)
    reg_loss, logits = self.regression_loss(summary,
                                            labels,
                                            is_training,
                                            return_logits=True,
                                            scope=scope)
    return reg_loss, logits

  def get_race_loss(self, labels, inputs, is_training, seg_id=None,
                    input_mask=None, use_tpu=False, use_bfloat16=False):
    """RACE loss."""
    net_config = self.net_config
    initializer = self.get_initializer()

    bsz_per_core = tf.shape(inputs)[0]
    inputs = tf.reshape(inputs, [bsz_per_core * 4, -1])
    labels = tf.reshape(labels, [bsz_per_core])

    if seg_id is not None:
      seg_id = tf.reshape(seg_id, [bsz_per_core * 4, -1])
    if input_mask is not None:
      input_mask = tf.reshape(input_mask, [bsz_per_core * 4, -1])

    summary, _ = self.get_pooled_output(inputs,
                                        is_training,
                                        seg_id=seg_id,
                                        input_mask=input_mask,
                                        use_tpu=use_tpu,
                                        use_bfloat16=use_bfloat16)

    with tf.variable_scope("race"):
      summary = ops.dropout_op(summary, net_config.dropout,
                               training=is_training)
      logits = ops.dense(
          summary,
          1,
          initializer=initializer,
          scope="logits")

      logits = tf.reshape(logits, [bsz_per_core, 4])
      logits = tf.cast(logits, tf.float32)
      one_hot_target = tf.one_hot(labels, 4, dtype=logits.dtype)
      per_example_loss = -tf.reduce_sum(
          tf.nn.log_softmax(logits) * one_hot_target, -1)

    return per_example_loss, logits

  def binary_logits(self, hidden, scope="binary", reuse=False):
    """Compute per-element bianry classification logits."""
    net_config = self.net_config
    initializer = self.get_initializer()
    with tf.variable_scope("{}_proj".format(scope), reuse=reuse):
      hidden = ops.dense(
          hidden,
          net_config.d_model,
          activation=ops.get_activation("gelu"),
          initializer=initializer)

    with tf.variable_scope("{}_loss".format(scope), reuse=reuse):
      binary_w = tf.get_variable("weight", [net_config.d_model],
                                 dtype=hidden.dtype, initializer=initializer)

      binary_b = tf.get_variable("bias", [1], dtype=hidden.dtype,
                                 initializer=tf.zeros_initializer())

      logits = tf.einsum("bid,d->bi", hidden, binary_w) + binary_b
      if logits.dtype != tf.float32:
        # Always use float32 for loss
        logits = tf.cast(logits, tf.float32)
    return logits

  def binary_loss(self, hidden, target, scope="binary", reuse=False):
    """Compute per-element bianry classification loss."""
    logits = self.binary_logits(hidden, scope=scope, reuse=reuse)

    target = tf.cast(target, dtype=logits.dtype)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=logits)

    # whether predictions are correct
    prediction = tf.cast(logits > 0, target.dtype)
    correct = tf.cast(tf.equal(prediction, target), tf.float32)

    return loss, correct

  def get_squad_loss(self, inputs, cls_index, para_mask, is_training,
                     seg_id=None, input_mask=None, start_positions=None,
                     use_tpu=False, use_bfloat16=False):
    """SQuAD loss."""
    net_config = self.net_config
    initializer = self.get_initializer()

    seq_len = tf.shape(inputs)[1]
    output, _, _ = self.extract_hiddens(
        inputs,
        is_training,
        seg_id=seg_id,
        input_mask=input_mask,
        use_decoder=True,
        use_tpu=use_tpu,
        use_bfloat16=use_bfloat16)

    with tf.variable_scope("start_logits"):
      # [B x L x D] -> [B x L x 1]
      start_logits = ops.dense(
          output,
          1,
          initializer=initializer)
      # [B x L x 1] -> [B x L]
      start_logits = tf.squeeze(start_logits, -1)
      start_logits_masked = start_logits * (1 - para_mask) - 1e30 * para_mask
      # [B x L]
      start_log_probs = tf.nn.log_softmax(
          tf.cast(start_logits_masked, tf.float32), -1)

    with tf.variable_scope("end_logits"):
      if FLAGS.conditional_end:
        if is_training:
          assert start_positions is not None
          start_index = tf.one_hot(start_positions, depth=seq_len, axis=-1,
                                   dtype=output.dtype)
          start_features = tf.einsum("blh,bl->bh", output, start_index)
          start_features = tf.tile(start_features[:, None], [1, seq_len, 1])
          end_logits = ops.dense(
              tf.concat([output, start_features], axis=-1),
              net_config.d_model,
              initializer=initializer,
              activation=tf.tanh,
              scope="dense_0")
          end_logits = ops.layer_norm_op(end_logits, begin_norm_axis=-1)

          end_logits = ops.dense(
              end_logits, 1,
              initializer=initializer,
              scope="dense_1")
          end_logits = tf.squeeze(end_logits, -1)
          end_logits_masked = end_logits * (1 - para_mask) - 1e30 * para_mask
          # [B x L]
          end_log_probs = tf.nn.log_softmax(
              tf.cast(end_logits_masked, tf.float32), -1)
        else:
          start_top_log_probs, start_top_index = tf.nn.top_k(
              start_log_probs, k=FLAGS.start_n_top)
          start_index = tf.one_hot(start_top_index,
                                   depth=seq_len, axis=-1, dtype=output.dtype)
          # [B x L x D] + [B x K x L] -> [B x K x D]
          start_features = tf.einsum("blh,bkl->bkh", output, start_index)
          # [B x L x D] -> [B x 1 x L x D] -> [B x K x L x D]
          end_input = tf.tile(output[:, None],
                              [1, FLAGS.start_n_top, 1, 1])
          # [B x K x D] -> [B x K x 1 x D] -> [B x K x L x D]
          start_features = tf.tile(start_features[:, :, None],
                                   [1, 1, seq_len, 1])
          # [B x K x L x 2D]
          end_input = tf.concat([end_input, start_features], axis=-1)
          end_logits = ops.dense(
              end_input,
              net_config.d_model,
              initializer=initializer,
              activation=tf.tanh,
              scope="dense_0")
          end_logits = ops.layer_norm_op(end_logits, begin_norm_axis=-1)
          # [B x K x L x 1]
          end_logits = ops.dense(
              end_logits,
              1,
              initializer=initializer,
              scope="dense_1")

          # [B x K x L]
          end_logits = tf.squeeze(end_logits, -1)
          if FLAGS.use_masked_loss:
            end_logits_masked = end_logits * (
                1 - para_mask[:, None]) - 1e30 * para_mask[:, None]
          else:
            end_logits_masked = end_logits
          # [B x K x L]
          end_log_probs = tf.nn.log_softmax(
              tf.cast(end_logits_masked, tf.float32), -1)
          # [B x K x K']
          end_top_log_probs, end_top_index = tf.nn.top_k(
              end_log_probs, k=FLAGS.end_n_top)
          # [B x K*K']
          end_top_log_probs = tf.reshape(
              end_top_log_probs,
              [-1, FLAGS.start_n_top * FLAGS.end_n_top])
          end_top_index = tf.reshape(
              end_top_index,
              [-1, FLAGS.start_n_top * FLAGS.end_n_top])
      else:
        end_logits = ops.dense(
            output,
            1,
            initializer=initializer)
        end_logits = tf.squeeze(end_logits, -1)
        end_logits_masked = end_logits * (1 - para_mask) - 1e30 * para_mask
        end_log_probs = tf.nn.log_softmax(
            tf.cast(end_logits_masked, tf.float32), -1)
        if not is_training:
          start_top_log_probs, start_top_index = tf.nn.top_k(
              start_log_probs, k=FLAGS.start_n_top)
          end_top_log_probs, end_top_index = tf.nn.top_k(
              end_log_probs, k=FLAGS.end_n_top)

    return_dict = {}
    if is_training:
      return_dict["start_log_probs"] = start_log_probs
      return_dict["end_log_probs"] = end_log_probs
    else:
      return_dict["start_top_log_probs"] = start_top_log_probs
      return_dict["start_top_index"] = start_top_index
      return_dict["end_top_log_probs"] = end_top_log_probs
      return_dict["end_top_index"] = end_top_index

    if FLAGS.use_answer_class:
      with tf.variable_scope("answer_class"):
        cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=output.dtype)
        cls_feature = tf.einsum("blh,bl->bh", output, cls_index)

        start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                                name="softmax_start")
        start_feature = tf.einsum("blh,bl->bh", output, start_p)

        ans_feature = tf.concat([start_feature, cls_feature], -1)
        ans_feature = ops.dense(
            ans_feature,
            FLAGS.d_model,
            activation=tf.tanh,
            initializer=initializer,
            scope="dense_0")
        ans_feature = ops.dropout_op(ans_feature, net_config.dropout,
                                     training=is_training)
        cls_logits = ops.dense(
            ans_feature,
            1,
            initializer=initializer,
            scope="dense_1",
            use_bias=False)
        cls_logits = tf.squeeze(cls_logits, -1)

        return_dict["cls_logits"] = tf.cast(cls_logits, tf.float32)
    else:
      cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
      cls_logits = tf.einsum("bl,bl->b", start_log_probs, cls_index)

      return_dict["cls_logits"] = tf.cast(cls_logits, tf.float32)

    return return_dict
