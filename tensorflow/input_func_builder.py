"""Create input function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow.compat.v1 as tf

import data_utils


FLAGS = flags.FLAGS
flags.DEFINE_enum("sample_strategy", default="single_token",
                  enum_values=["single_token", "whole_word", "token_span",
                               "word_span"],
                  help="Stragey used to sample prediction targets.")
flags.DEFINE_bool("shard_across_host", default=True,
                  help="Shard files across available hosts.")

flags.DEFINE_float("leak_ratio", default=0.1,
                   help="Percent of masked positions that are filled with "
                   "original tokens.")
flags.DEFINE_float("rand_ratio", default=0.1,
                   help="Percent of masked positions that are filled with "
                   "random tokens.")

flags.DEFINE_integer("max_tok", default=5,
                     help="Maximum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")
flags.DEFINE_integer("min_tok", default=1,
                     help="Minimum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")

flags.DEFINE_integer("max_word", default=5,
                     help="Maximum number of whole words to sample in a span."
                     "Effective when word_span strategy is used.")
flags.DEFINE_integer("min_word", default=1,
                     help="Minimum number of whole words to sample in a span."
                     "Effective when word_span strategy is used.")


def parse_files_to_dataset(parser, file_paths, split, num_hosts,
                           host_id, num_core_per_host, bsz_per_core,
                           num_threads=256, shuffle_buffer=20480):
  """Parse a list of file names into a single tf.dataset."""

  def get_options():
    options = tf.data.Options()
    # Forces map and interleave to be sloppy for enhance performance.
    options.experimental_deterministic = False
    return options

  if FLAGS.shard_across_host and len(file_paths) >= num_hosts:
    tf.logging.info("Shard %d files across %s hosts.", len(file_paths),
                    num_hosts)
    file_paths = file_paths[host_id::num_hosts]
  tf.logging.info("Host %d/%d handles %d files", host_id, num_hosts,
                  len(file_paths))

  assert split == "train"
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # file-level shuffle
  if len(file_paths) > 1:
    tf.logging.info("Perform file-level shuffle with size %d", len(file_paths))
    dataset = dataset.shuffle(len(file_paths))

  # `cycle_length` is the number of parallel files that get read.
  cycle_length = min(num_threads, len(file_paths))
  tf.logging.info("Interleave %d files", cycle_length)

  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      num_parallel_calls=cycle_length,
      cycle_length=cycle_length)
  dataset.with_options(get_options())

  tf.logging.info("Perform sample-level shuffle with size %d", shuffle_buffer)
  dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Note: since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  dataset = dataset.cache().map(
      parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset


def _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len, num_predict):
  """Turn beg and end indices into actual mask."""
  non_func_mask = tf.logical_and(
      tf.not_equal(inputs, FLAGS.sep_id),
      tf.not_equal(inputs, FLAGS.cls_id))
  all_indices = tf.where(
      non_func_mask,
      tf.range(tgt_len, dtype=tf.int64),
      tf.constant(-1, shape=[tgt_len], dtype=tf.int64))
  candidate_matrix = tf.cast(
      tf.logical_and(
          all_indices[None, :] >= beg_indices[:, None],
          all_indices[None, :] < end_indices[:, None]),
      tf.float32)
  cumsum_matrix = tf.reshape(
      tf.cumsum(tf.reshape(candidate_matrix, [-1])),
      [-1, tgt_len])
  masked_matrix = tf.cast(cumsum_matrix <= num_predict, tf.float32)
  target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
  is_target = tf.cast(target_mask, tf.bool)

  return is_target, target_mask


def _word_span_mask(inputs, tgt_len, num_predict, boundary, stride=1):
  """Sample whole word spans as prediction targets."""
  # Note: 1.2 is roughly the token-to-word ratio
  non_pad_len = tgt_len + 1 - stride
  chunk_len_fp = non_pad_len / num_predict / 1.2
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(FLAGS.min_word, FLAGS.max_word + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])
  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)

  # Sample `num_predict` words here: note that this is over sampling
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + FLAGS.min_word

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_fp = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)

  left_ctx_len = round_to_int(left_ctx_len)
  right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range `boundary` indices
  max_boundary_index = tf.cast(tf.shape(boundary)[0] - 1, tf.int64)
  valid_idx_mask = end_indices < max_boundary_index
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  beg_indices = tf.gather(boundary, beg_indices)
  end_indices = tf.gather(boundary, end_indices)

  # Shuffle valid `position` indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _token_span_mask(inputs, tgt_len, num_predict, stride=1):
  """Sample token spans as prediction targets."""
  non_pad_len = tgt_len + 1 - stride
  chunk_len_fp = non_pad_len / num_predict
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(FLAGS.min_tok, FLAGS.max_tok + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + FLAGS.min_tok

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_fp = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

  # Get the actual begin and end indices
  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range indices
  valid_idx_mask = end_indices < non_pad_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _whole_word_mask(inputs, tgt_len, num_predict, boundary):
  """Sample whole words as prediction targets."""
  pair_indices = tf.concat([boundary[:-1, None], boundary[1:, None]], axis=1)
  cand_pair_indices = tf.random.shuffle(pair_indices)[:num_predict]
  beg_indices = cand_pair_indices[:, 0]
  end_indices = cand_pair_indices[:, 1]

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _single_token_mask(inputs, tgt_len, num_predict, exclude_mask=None):
  """Sample individual tokens as prediction targets."""
  func_mask = tf.equal(inputs, FLAGS.cls_id)
  func_mask = tf.logical_or(func_mask, tf.equal(inputs, FLAGS.sep_id))
  func_mask = tf.logical_or(func_mask, tf.equal(inputs, FLAGS.pad_id))
  if exclude_mask is None:
    exclude_mask = func_mask
  else:
    exclude_mask = tf.logical_or(func_mask, exclude_mask)
  candidate_mask = tf.logical_not(exclude_mask)

  all_indices = tf.range(tgt_len, dtype=tf.int64)
  candidate_indices = tf.boolean_mask(all_indices, candidate_mask)
  masked_pos = tf.random.shuffle(candidate_indices)
  masked_pos = tf.sort(masked_pos[:num_predict])
  target_mask = tf.sparse_to_dense(
      sparse_indices=masked_pos,
      output_shape=[tgt_len],
      sparse_values=1.0,
      default_value=0.0)
  is_target = tf.cast(target_mask, tf.bool)

  return is_target, target_mask


def _online_sample_masks(
    inputs, tgt_len, num_predict, boundary=None, stride=1):
  """Sample target positions to predict."""
  tf.logging.info("Online sample with strategy: `%s`.", FLAGS.sample_strategy)
  if FLAGS.sample_strategy == "single_token":
    return _single_token_mask(inputs, tgt_len, num_predict)
  else:
    if FLAGS.sample_strategy == "whole_word":
      assert boundary is not None, "whole word sampling requires `boundary`"
      is_target, target_mask = _whole_word_mask(inputs, tgt_len, num_predict,
                                                boundary)
    elif FLAGS.sample_strategy == "token_span":
      is_target, target_mask = _token_span_mask(inputs, tgt_len, num_predict,
                                                stride=stride)
    elif FLAGS.sample_strategy == "word_span":
      assert boundary is not None, "word span sampling requires `boundary`"
      is_target, target_mask = _word_span_mask(inputs, tgt_len, num_predict,
                                               boundary, stride=stride)
    else:
      raise NotImplementedError

    # Fill in single tokens if not full
    cur_num_masked = tf.reduce_sum(tf.cast(is_target, tf.int64))
    extra_mask, extra_tgt_mask = _single_token_mask(
        inputs, tgt_len, num_predict - cur_num_masked, is_target)
    return tf.logical_or(is_target, extra_mask), target_mask + extra_tgt_mask


def discrepancy_correction(inputs, is_target, tgt_len):
  """Construct the masked input."""
  random_p = tf.random.uniform([tgt_len], maxval=1.0)
  mask_ids = tf.constant(FLAGS.mask_id, dtype=inputs.dtype, shape=[tgt_len])

  change_to_mask = tf.logical_and(random_p > FLAGS.leak_ratio, is_target)
  masked_ids = tf.where(change_to_mask, mask_ids, inputs)

  if FLAGS.rand_ratio > 0:
    change_to_rand = tf.logical_and(
        FLAGS.leak_ratio < random_p,
        random_p < FLAGS.leak_ratio + FLAGS.rand_ratio)
    change_to_rand = tf.logical_and(change_to_rand, is_target)
    rand_ids = tf.random.uniform([tgt_len], maxval=FLAGS.vocab_size,
                                 dtype=masked_ids.dtype)
    masked_ids = tf.where(change_to_rand, rand_ids, masked_ids)

  return masked_ids


def create_target_mapping(
    example, is_target, seq_len, num_predict, **kwargs):
  """Create target mapping and retrieve the corresponding kwargs."""
  if num_predict is not None:
    # Get masked indices
    indices = tf.range(seq_len, dtype=tf.int64)
    indices = tf.boolean_mask(indices, is_target)

    # Handle the case that actual_num_predict < num_predict
    actual_num_predict = tf.shape(indices)[0]
    pad_len = num_predict - actual_num_predict

    # Create target mapping
    target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
    paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
    target_mapping = tf.concat([target_mapping, paddings], axis=0)
    example["target_mapping"] = tf.reshape(target_mapping,
                                           [num_predict, seq_len])

    # Handle fields in kwargs
    for k, v in kwargs.items():
      pad_shape = [pad_len] + v.shape.as_list()[1:]
      tgt_shape = [num_predict] + v.shape.as_list()[1:]
      example[k] = tf.concat([
          tf.boolean_mask(v, is_target),
          tf.zeros(shape=pad_shape, dtype=v.dtype)], 0)
      example[k].set_shape(tgt_shape)
  else:
    for k, v in kwargs.items():
      example[k] = v


def get_dataset(
    params, num_hosts, num_core_per_host, split, file_paths, seq_len,
    num_predict, use_bfloat16=False, truncate_seq=False, stride=1):
  """Get one-stream dataset."""

  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  #### Function used to parse tfrecord
  def parser(record):
    """function used to parse tfrecord."""

    record_spec = {
        "input": tf.io.FixedLenFeature([seq_len], tf.int64),
        "seg_id": tf.io.FixedLenFeature([seq_len], tf.int64),
    }

    if FLAGS.sample_strategy in ["whole_word", "word_span"]:
      tf.logging.info("Add `boundary` spec for %s", FLAGS.sample_strategy)
      record_spec["boundary"] = tf.io.VarLenFeature(tf.int64)

    # retrieve serialized example
    example = tf.io.parse_single_example(
        serialized=record,
        features=record_spec)

    inputs = example.pop("input")
    if FLAGS.sample_strategy in ["whole_word", "word_span"]:
      boundary = tf.sparse.to_dense(example.pop("boundary"))
    else:
      boundary = None

    if truncate_seq and stride > 1:
      tf.logging.info("Truncate pretrain sequence with stride %d", stride)
      # seq_len = 8, stride = 2:
      #   [cls 1 2 sep 4 5 6 sep] => [cls 1 2 sep 4 5 sep pad]
      padding = tf.constant([FLAGS.sep_id] + [FLAGS.pad_id] * (stride - 1),
                            dtype=inputs.dtype)
      inputs = tf.concat([inputs[:-stride], padding], axis=0)
      if boundary is not None:
        valid_boundary_mask = boundary < seq_len - stride
        boundary = tf.boolean_mask(boundary, valid_boundary_mask)

    is_target, target_mask = _online_sample_masks(
        inputs, seq_len, num_predict, boundary=boundary, stride=stride)

    masked_input = discrepancy_correction(inputs, is_target, seq_len)
    masked_input = tf.reshape(masked_input, [seq_len])
    is_mask = tf.equal(masked_input, FLAGS.mask_id)
    is_pad = tf.equal(masked_input, FLAGS.pad_id)

    example["masked_input"] = masked_input
    example["origin_input"] = inputs
    example["is_target"] = is_target
    example["input_mask"] = tf.cast(tf.logical_or(is_mask, is_pad), tf.float32)
    example["pad_mask"] = tf.cast(is_pad, tf.float32)

    # create target mapping
    create_target_mapping(
        example, is_target, seq_len, num_predict,
        target_mask=target_mask, target=inputs)

    # type cast for example
    data_utils.convert_example(example, use_bfloat16)

    for k, v in example.items():
      tf.logging.info("%s: %s", k, v)

    return example

  # Get dataset
  dataset = parse_files_to_dataset(
      parser=parser,
      file_paths=file_paths,
      split=split,
      num_hosts=num_hosts,
      host_id=host_id,
      num_core_per_host=num_core_per_host,
      bsz_per_core=bsz_per_core)

  return dataset


def get_input_fn(
    tfrecord_dir,
    split,
    bsz_per_host,
    seq_len,
    num_predict,
    num_hosts=1,
    num_core_per_host=1,
    uncased=False,
    num_passes=None,
    use_bfloat16=False,
    num_pool=0,
    truncate_seq=False):
  """Create Estimator input function."""

  assert num_predict is not None and 0 < num_predict < seq_len - 3
  stride = 2 ** num_pool

  # Merge all record infos into a single one
  record_glob_base = data_utils.format_filename(
      prefix="meta.{}.pass-*".format(split),
      suffix="json*",
      seq_len=seq_len,
      uncased=uncased)

  def _get_num_batch(info):
    if "num_batch" in info:
      return info["num_batch"]
    elif "num_example" in info:
      return info["num_example"] / bsz_per_host
    else:
      raise ValueError("Do not have sample info.")

  record_info = {"num_batch": 0, "filenames": []}

  tfrecord_dirs = tfrecord_dir.split(",")
  tf.logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    tf.logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.io.gfile.glob(record_glob))
    tf.logging.info("[%d] Num of record info path: %d",
                    idx, len(record_paths))

    cur_record_info = {"num_batch": 0, "filenames": []}

    for record_info_path in record_paths:
      if num_passes is not None:
        record_info_name = os.path.basename(record_info_path)
        fields = record_info_name.split(".")[2].split("-")
        pass_id = int(fields[-1])
        if pass_id >= num_passes:
          tf.logging.debug("Skip pass %d: %s", pass_id, record_info_name)
          continue

      with tf.io.gfile.GFile(record_info_path, "r") as fp:
        info = json.load(fp)
        cur_record_info["num_batch"] += int(_get_num_batch(info))
        cur_record_info["filenames"] += info["filenames"]

    # overwrite directory for `cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    tf.logging.info("[Dir %d] Number of chosen batches: %s",
                    idx, cur_record_info["num_batch"])
    tf.logging.info("[Dir %d] Number of chosen files: %s",
                    idx, len(cur_record_info["filenames"]))
    tf.logging.debug(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  tf.logging.info("Total number of batches: %d", record_info["num_batch"])
  tf.logging.info("Total number of files: %d", len(record_info["filenames"]))
  tf.logging.debug(record_info["filenames"])

  def input_fn(params):
    """Input function wrapper."""
    dataset = get_dataset(
        params=params,
        num_hosts=num_hosts,
        num_core_per_host=num_core_per_host,
        split=split,
        file_paths=record_info["filenames"],
        seq_len=seq_len,
        use_bfloat16=use_bfloat16,
        num_predict=num_predict,
        truncate_seq=truncate_seq,
        stride=stride)

    return dataset

  return input_fn, record_info
