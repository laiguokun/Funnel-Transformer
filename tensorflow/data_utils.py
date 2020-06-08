"""Common data utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow.compat.v1 as tf


flags.DEFINE_integer("vocab_size", default=None, help="")
flags.DEFINE_integer("unk_id", default=None, help="")
flags.DEFINE_integer("bos_id", default=None, help="")
flags.DEFINE_integer("eos_id", default=None, help="")
flags.DEFINE_integer("cls_id", default=None, help="")
flags.DEFINE_integer("sep_id", default=None, help="")
flags.DEFINE_integer("pad_id", default=None, help="")
flags.DEFINE_integer("mask_id", default=None, help="")
flags.DEFINE_integer("eod_id", default=None, help="")
flags.DEFINE_integer("eop_id", default=None, help="")

flags.DEFINE_integer("seg_id_a", default=0, help="segment id of segment A.")
flags.DEFINE_integer("seg_id_b", default=1, help="segment id of segment B.")
flags.DEFINE_integer("seg_id_cls", default=2, help="segment id of cls.")
flags.DEFINE_integer("seg_id_pad", default=0, help="segment id of pad.")


FLAGS = flags.FLAGS


special_symbols_mapping = collections.OrderedDict([
    ("<unk>", "unk_id"),
    ("<s>", "bos_id"),
    ("</s>", "eos_id"),
    ("<cls>", "cls_id"),
    ("<sep>", "sep_id"),
    ("<pad>", "pad_id"),
    ("<mask>", "mask_id"),
    ("<eod>", "eod_id"),
    ("<eop>", "eop_id")
])


def setup_special_ids(tokenizer):
  """Set up the id of special tokens."""
  FLAGS.vocab_size = tokenizer.get_vocab_size()
  tf.logging.info("Set vocab_size: %d.", FLAGS.vocab_size)
  for sym, sym_id_str in special_symbols_mapping.items():
    try:
      sym_id = tokenizer.get_token_id(sym)
      setattr(FLAGS, sym_id_str, sym_id)
      tf.logging.info("Set %s to %d.", sym_id_str, sym_id)
    except KeyError:
      tf.logging.warning("Skip %s: not found in tokenizer's vocab.", sym)


def format_filename(prefix, suffix, seq_len, uncased):
  """Format the name of the tfrecord/meta file."""
  seq_str = "seq-{}".format(seq_len)
  if uncased:
    case_str = "uncased"
  else:
    case_str = "cased"

  file_name = "{}.{}.{}.{}".format(prefix, seq_str, case_str, suffix)

  return file_name


def convert_example(example, use_bfloat16=False):
  """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    if val.dtype == tf.int64:
      val = tf.cast(val, tf.int32)
    if use_bfloat16 and val.dtype == tf.float32:
      val = tf.cast(val, tf.bfloat16)

    example[key] = val


def sparse_to_dense(example):
  """Convert sparse feature to dense ones."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    example[key] = val

  return example


def read_docs(file_path, tokenizer):
  """Read docs from a file separated by empty lines."""
  # working structure used to store each document
  all_docs = []
  doc, end_of_doc = [], False

  line_cnt = 0
  tf.logging.info("Start processing %s", file_path)
  for line in tf.io.gfile.GFile(file_path):
    if line_cnt % 100000 == 0:
      tf.logging.info("Loading line %d", line_cnt)

    if not line.strip():
      # encounter an empty line (end of a document)
      end_of_doc = True
      cur_sent = []
    else:
      cur_sent = tokenizer.convert_text_to_ids(line.strip())

    if cur_sent:
      line_cnt += 1
      doc.append(np.array(cur_sent))

    # form a doc
    if end_of_doc or sum(map(len, doc)) >= FLAGS.max_doc_len:
      # only retain docs longer than `min_doc_len`
      doc_len = sum(map(len, doc))
      if doc_len >= max(FLAGS.min_doc_len, 1):
        all_docs.append(doc)

      # refresh working structs
      doc, end_of_doc = [], False

  # deal with the leafover if any
  if doc:
    # only retain docs longer than `min_doc_len`
    doc_len = sum(map(len, doc))
    if doc_len >= max(FLAGS.min_doc_len, 1):
      all_docs.append(doc)

  tf.logging.info("Finish %s with %d docs from %d lines.", file_path,
                  len(all_docs), line_cnt)

  return all_docs
