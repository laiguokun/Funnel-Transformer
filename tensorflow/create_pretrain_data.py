"""Create tfrecord for pretraining."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow.compat.v1 as tf

import data_utils
import tokenization

tf.disable_v2_behavior()


FLAGS = flags.FLAGS
flags.DEFINE_integer("min_doc_len", 1,
                     help="Minimum document length allowed.")
flags.DEFINE_integer("seq_len", 512,
                     help="Sequence length.")

flags.DEFINE_string("input_glob", "data/example/*.txt",
                    help="Input file glob.")
flags.DEFINE_string("save_dir", "proc_data/example",
                    help="Directory for saving the processed data.")
flags.DEFINE_enum("split", "train", ["train", "dev", "test"],
                  help="Save the data as which split.")

flags.DEFINE_integer("pass_id", 0, help="ID of the current pass."
                     "Different passes sample different negative segment.")
flags.DEFINE_integer("num_task", 1, help="Number of total tasks.")
flags.DEFINE_integer("task", 0, help="The Task ID. This value is used when "
                     "using multiple workers to identify each worker.")


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _tfrecord_path(save_dir):
  """Get tfrecord path."""
  data_prefix = "data.{}.pass-{}".format(FLAGS.split, FLAGS.pass_id)
  data_suffix = "tfrecord-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  tfrecord_name = data_utils.format_filename(
      prefix=data_prefix,
      suffix=data_suffix,
      seq_len=FLAGS.seq_len,
      uncased=FLAGS.uncased,
  )
  tfrecord_path = os.path.join(save_dir, tfrecord_name)

  return tfrecord_path


def _meta_path(save_dir):
  """Get meta path."""
  meta_prefix = "meta.{}.pass-{}".format(FLAGS.split, FLAGS.pass_id)
  meta_suffix = "json-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  meta_name = data_utils.format_filename(
      prefix=meta_prefix,
      suffix=meta_suffix,
      seq_len=FLAGS.seq_len,
      uncased=FLAGS.uncased,
  )
  meta_path = os.path.join(save_dir, meta_name)

  return meta_path


def create_pretrain_data(input_paths, tokenizer):
  """Load data and call corresponding create_func."""
  input_shards = []

  # working structure used to store each document
  input_data, sent_ids = [], []
  end_of_doc = False

  # monitor doc length and number of tokens
  doc_length = []
  total_num_tok = 0

  for input_path in input_paths:
    sent_id, line_cnt = True, 0

    tf.logging.info("Start processing %s", input_path)
    for line in tf.io.gfile.GFile(input_path):
      if line_cnt % 100000 == 0:
        tf.logging.info("Loading line %d", line_cnt)

      if not line.strip():
        # encounter an empty line (end of a document)
        end_of_doc = True
        cur_sent = []
      else:
        cur_sent = tokenizer.convert_text_to_ids(line.strip())

      if cur_sent:
        input_data.extend(cur_sent)
        sent_ids.extend([sent_id] * len(cur_sent))
        sent_id = not sent_id

      if end_of_doc:
        # monitor over doc lengths
        doc_length.append(len(input_data))

        # only retain docs longer than `min_doc_len`
        if len(input_data) >= max(FLAGS.min_doc_len, 1):
          input_data = np.array(input_data, dtype=np.int64)
          sent_ids = np.array(sent_ids, dtype=np.bool)
          input_shards.append((input_data, sent_ids))
          total_num_tok += len(input_data)

        # refresh working structs
        input_data, sent_ids = [], []
        end_of_doc = False

      line_cnt += 1

    tf.logging.info("Finish %s with %d lines.", input_path, line_cnt)

  tf.logging.info("[Task %d] Total number tokens: %d", FLAGS.task,
                  total_num_tok)

  hist, bins = np.histogram(doc_length,
                            bins=[0, 64, 128, 256, 512, 1024, 2048, 102400])
  percent = hist / np.sum(hist)
  tf.logging.info("***** Doc length histogram *****")
  for pct, l, r in zip(percent, bins[:-1], bins[1:]):
    tf.logging.info("  - [%d, %d]: %.4f", l, r, pct)

  # Randomly shuffle input shards (with a fixed but unique random seed)
  np.random.seed(100 * FLAGS.task + FLAGS.pass_id)
  perm_indices = np.random.permutation(len(input_shards))

  input_data_list, sent_ids_list = [], []
  prev_sent_id = None
  for perm_idx in perm_indices:
    input_data, sent_ids = input_shards[perm_idx]
    tf.logging.debug("Idx %d: data %s sent %s", perm_idx,
                     input_data.shape, sent_ids.shape)
    # make sure the `send_ids[0] == not prev_sent_id`
    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    # append to temporary list
    input_data_list.append(input_data)
    sent_ids_list.append(sent_ids)

    # update `prev_sent_id`
    prev_sent_id = sent_ids[-1]

  # concat into a flat np.ndarray
  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)

  create_tfrecords(
      save_dir=FLAGS.save_dir,
      data=[input_data, sent_ids],
      tokenizer=tokenizer,
  )


def main(_):
  """create pretraining data (tfrecords)."""
  # Load tokenizer
  tokenizer = tokenization.get_tokenizer()
  data_utils.setup_special_ids(tokenizer)

  # Make workdirs
  if not tf.io.gfile.exists(FLAGS.save_dir):
    tf.io.gfile.makedirs(FLAGS.save_dir)

  # Interleavely split the work into FLAGS.num_task splits
  file_paths = sorted(tf.io.gfile.glob(FLAGS.input_glob))
  tf.logging.info("Use glob: %s", FLAGS.input_glob)
  tf.logging.info("Find %d files: %s", len(file_paths), file_paths)

  task_file_paths = file_paths[FLAGS.task::FLAGS.num_task]
  if not task_file_paths:
    tf.logging.info("Exit: task %d has no file to process.", FLAGS.task)
    return

  tf.logging.info("Task %d process %d files: %s",
                  FLAGS.task, len(task_file_paths), task_file_paths)

  create_pretrain_data(task_file_paths, tokenizer)


def _split_a_and_b(data, sent_ids, begin_idx, tot_len):
  """Split two segments from `data` starting from the index `begin_idx`."""

  data_len = data.shape[0]
  if begin_idx + tot_len >= data_len:
    tf.logging.info("Not enough data: "
                    "begin_idx %d + tot_len %d >= data_len %d",
                    begin_idx, tot_len, data_len)
    return None

  end_idx = begin_idx + 1
  cut_points = []
  while end_idx < data_len:
    if sent_ids[end_idx] != sent_ids[end_idx - 1]:
      if end_idx - begin_idx >= tot_len: break
      cut_points.append(end_idx)
    end_idx += 1

  a_begin = begin_idx
  if not cut_points or random.random() < 0.5:
    # negative pair
    label = 0
    if not cut_points:
      a_end = end_idx
    else:
      a_end = random.choice(cut_points)

    b_len = max(1, tot_len - (a_end - a_begin))
    b_begin = random.randint(0, data_len - b_len)
    b_end = b_begin + b_len

    # locate a complete sentence for `b`
    while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
      b_begin -= 1
    while b_end < data_len and sent_ids[b_end - 1] == sent_ids[b_end]:
      b_end += 1

    new_begin = a_end
  else:
    # positive pair
    label = 1
    a_end = random.choice(cut_points)
    b_begin = a_end
    b_end = end_idx

    new_begin = b_end

  # truncate both a & b
  while a_end - a_begin + b_end - b_begin > tot_len:
    # truncate a (only right)
    if a_end - a_begin > b_end - b_begin:
      a_end -= 1
    # truncate b (both left and right)
    else:
      if random.random() < 0.5:
        b_end -= 1
      else:
        b_begin += 1

  ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

  return ret


def _get_boundary_indices(tokenizer, seg, reverse=False):
  """Get all boundary indices of whole words."""
  seg_len = len(seg)
  if reverse:
    seg = np.flip(seg, 0)

  boundary_indices = []
  for idx, token_id in enumerate(seg.tolist()):
    if tokenizer.is_start_id(token_id) and not tokenizer.is_func_id(token_id):
      boundary_indices.append(idx)
  boundary_indices.append(seg_len)

  if reverse:
    boundary_indices = [seg_len - idx for idx in boundary_indices]

  return boundary_indices


def create_tfrecords(save_dir, data, tokenizer):
  """create tfrecords from numpy array."""
  ##### Prepare data
  data, sent_ids = data[0], data[1]
  tf.logging.info("Raw data shape %s.", data.shape)

  ##### Create record writer
  tfrecord_path = _tfrecord_path(save_dir)
  record_writer = tf.python_io.TFRecordWriter(tfrecord_path)
  tf.logging.info("Start writing tfrecord to %s.", tfrecord_path)

  ##### Create tfrecord
  data_len = data.shape[0]

  sep_array = np.array([FLAGS.sep_id], dtype=np.int64)
  cls_array = np.array([FLAGS.cls_id], dtype=np.int64)

  i = 0
  num_example = 0
  while i + FLAGS.seq_len <= data_len:
    if num_example % 10000 == 0:
      tf.logging.info("Processing example %d", num_example)

    ##### sample two segments a & b and the corresponding `label` (pos | neg)
    results = _split_a_and_b(
        data,
        sent_ids,
        begin_idx=i,
        tot_len=FLAGS.seq_len - 3)

    if results is None:
      tf.logging.info("Break out at sequence position %d", i)
      break

    # unpack the results
    (a_data, b_data, label, new_begin) = tuple(results)

    ##### create `input` & `seg_id`
    cat_data = np.concatenate([
        cls_array, a_data, sep_array, b_data, sep_array])
    seg_id = ([FLAGS.seg_id_cls] +
              [FLAGS.seg_id_a] * (a_data.shape[0] + 1) +
              [FLAGS.seg_id_b] * (b_data.shape[0] + 1))

    ##### get word boundaries
    boundary = _get_boundary_indices(tokenizer, cat_data)

    ##### final check
    assert cat_data.shape[0] == FLAGS.seq_len

    feature = {
        "input": _int64_feature(cat_data),
        "seg_id": _int64_feature(seg_id),
        "boundary": _int64_feature(boundary),
        "label": _int64_feature([label]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    record_writer.write(example.SerializeToString())

    # update the number of examples
    num_example += 1

    # update the new begin index
    i = new_begin

  record_writer.close()
  tf.logging.info("Done writing %s. Num of examples: %d",
                  tfrecord_path, num_example)

  ##### dump record information
  meta_info = {
      "filenames": [os.path.basename(tfrecord_path)],
      "num_example": num_example
  }
  meta_path = _meta_path(save_dir)
  with tf.io.gfile.GFile(meta_path, "w") as fp:
    json.dump(meta_info, fp)

  return num_example


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
