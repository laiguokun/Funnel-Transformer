"""Finetune for squad."""
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gc
import json
import math
import os
import time

from absl import flags
import absl.logging as _logging

import numpy as np
import six
import tensorflow.compat.v1 as tf

if six.PY2:
  import cPickle as pickle
else:
  import pickle

import data_utils
import modeling
import model_utils
import optimization
import squad_utils_v1
import squad_utils_v2
import tokenization

logger = tf.get_logger()
logger.propagate = False
tf.disable_v2_behavior()


SPIECE_UNDERLINE = u"▁"


# Preprocessing
flags.DEFINE_bool("verbose", default=False,
                  help="Whether to print additional information.")

flags.DEFINE_string("prepro_split", default=None, help="run prepro")
flags.DEFINE_integer("num_proc", default=1,
                     help="Number of preprocessing processes.")
flags.DEFINE_integer("proc_id", default=0,
                     help="Process id.")

flags.DEFINE_integer("max_seq_length",
                     default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length",
                     default=64, help="Max query length")
flags.DEFINE_integer("doc_stride",
                     default=128, help="Doc stride")
flags.DEFINE_integer("max_answer_length",
                     default=64, help="Max answer length")
flags.DEFINE_bool("keep_accents", default=False,
                  help="Whether to keep accents.")

flags.DEFINE_string("train_version", default="v2", help="Train set version.")
flags.DEFINE_string("eval_version", default="v2", help="Eval set version.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")

flags.DEFINE_bool("retain_all", default=False,
                  help="Retain all checkpoints.")
flags.DEFINE_bool("retain_best", default=False,
                  help="Retain the best checkpoint.")

# Init checkpoint
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("model_config", "",
                    help="FunnelTFM model configuration.")
flags.DEFINE_bool("init_global_vars", default=False,
                  help="If true, init all global vars. If false, init "
                  "trainable vars only.")

flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("predict_dir", default="",
                    help="Dir for predictions.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("train_file", default="",
                    help="Path of train file.")
flags.DEFINE_string("predict_file", default="",
                    help="Path of prediction file.")

# TPUs and machines
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_bool("use_tpu", default=True, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2, 16 for TPU v3.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# training
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=5500,
                     help="Number of training steps")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. "
                     "Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                     "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=128,
                     help="batch size for training")

# prediction
flags.DEFINE_bool("do_predict", default=False, help="whether to do predict")
flags.DEFINE_integer("eval_start_step", default=-1,
                     help="Start step for evaluation")
flags.DEFINE_integer("eval_end_step", default=10000000,
                     help="End step for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128,
                     help="batch size for prediction")
flags.DEFINE_string("target_eval_key", "best_f1",
                    help="Use has_ans_f1 for Model selection.")
flags.DEFINE_integer("n_best_size", default=5,
                     help="n best size for predictions")
flags.DEFINE_integer("start_n_top", 5, "Beam size for span start.")
flags.DEFINE_integer("end_n_top", 5, "Beam size for span end.")

# ensemble
flags.DEFINE_bool("do_ensemble", default=False, help="whether to do ensemble.")
flags.DEFINE_string("ensemble_ckpts", None,
                    "Comma separated paths to the ensemble ckpts.")
flags.DEFINE_bool("search_threshold", False, "Whether to search for threshold")
flags.DEFINE_float("na_threshold", 0.0, "NA threshold for prediction")
flags.DEFINE_string("ensemble_output_file", default="",
                    help="Prediction output of the ensemble model.")

# squad specific
flags.DEFINE_bool("use_answer_class", True, "Use answer class")
flags.DEFINE_bool("use_masked_loss", True, "Use masked loss")
flags.DEFINE_bool("conditional_end", default=True,
                  help="Predict span end conditioned on span start.")
flags.DEFINE_float("cls_loss_weight", 0.5, "Cls loss weight")
flags.DEFINE_float("ans_loss_weight", 0.5, "Ans loss weight")

# debug
flags.DEFINE_bool("debug_mode", False, "Run the debug mode.")

FLAGS = flags.FLAGS


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               paragraph_text,
               orig_answer_text=None,
               start_position=None,
               is_impossible=False,
               regression_tgt=0.0):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.is_impossible = is_impossible
    self.regression_tgt = regression_tgt

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.convert_to_unicode(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.convert_to_unicode(self.question_text))
    s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tok_start_to_orig_index,
               tok_end_to_orig_index,
               token_is_max_context,
               input_ids,
               input_mask,
               p_mask,
               segment_ids,
               paragraph_len,
               cls_index,
               start_position=None,
               end_position=None,
               is_impossible=None,
               regression_tgt=0.0):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tok_start_to_orig_index = tok_start_to_orig_index
    self.tok_end_to_orig_index = tok_end_to_orig_index
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.p_mask = p_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.cls_index = cls_index
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible
    self.regression_tgt = regression_tgt


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.io.gfile.GFile(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        orig_answer_text = None
        is_impossible = False

        if is_training:
          if FLAGS.train_version == "v2":
            is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            start_position = answer["answer_start"]
          else:
            start_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            is_impossible=is_impossible)
        examples.append(example)

  return examples


def _convert_index(index, pos, M=None, is_start=True):
  if index[pos] is not None:
    return index[pos]
  N = len(index)
  rear = pos
  while rear < N - 1 and index[rear] is None:
    rear += 1
  front = pos
  while front > 0 and index[front] is None:
    front -= 1
  assert index[front] is not None or index[rear] is not None
  if index[front] is None:
    if index[rear] >= 1:
      if is_start:
        return 0
      else:
        return index[rear] - 1
    return index[rear]
  if index[rear] is None:
    if M is not None and index[front] < M - 1:
      if is_start:
        return index[front] + 1
      else:
        return M - 1
    return index[front]
  if is_start:
    if index[rear] > index[front] + 1:
      return index[front] + 1
    else:
      return index[rear]
  else:
    if index[rear] > index[front] + 1:
      return index[rear] - 1
    else:
      return index[front]


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  cnt_pos, cnt_neg = 0, 0
  unique_id = 1000000000
  max_N, max_M = 1024, 1024
  f = np.zeros((max_N, max_M), dtype=np.float32)

  for (example_index, example) in enumerate(examples):
    if example_index % 100 == 0:
      tf.logging.info("Converting {}/{} pos {} neg {}".format(
          example_index, len(examples), cnt_pos, cnt_neg))

    #### Query
    # Query text to ids
    query_ids = tokenizer.convert_text_to_ids(
        example.question_text)

    if len(query_ids) > max_query_length:
      query_ids = query_ids[0:max_query_length]
      tf.logging.info("Cut query of example %d", example_index)

    #### Paragraph
    paragraph_text = example.paragraph_text

    para_tokens = tokenizer.convert_text_to_tokens(
        example.paragraph_text)

    #### Mapping for `tok_cat_text`

    # list: char index to token index
    chartok_to_tok_index = []

    # list: token index to start char index
    tok_start_to_chartok_index = []

    # list: token index to end char index
    tok_end_to_chartok_index = []

    if FLAGS.tokenizer_type == "sent_piece":
      char_cnt = 0
      for i, token in enumerate(para_tokens):
        chartok_to_tok_index.extend([i] * len(token))
        tok_start_to_chartok_index.append(char_cnt)
        char_cnt += len(token)
        tok_end_to_chartok_index.append(char_cnt - 1)
      tok_cat_text = "".join(para_tokens).replace(SPIECE_UNDERLINE, " ")
    elif FLAGS.tokenizer_type == "word_piece":
      char_cnt = 0
      cat_tokens = []
      for i, token in enumerate(para_tokens):
        if token.startswith("##"):
          token = token[2:]
        elif (not (len(token) == 1 and tokenization._is_punctuation(token)) and
              not tokenizer.is_func_token(token)):
          token = " " + token
        cat_tokens.append(token)

        chartok_to_tok_index.extend([i] * len(token))
        tok_start_to_chartok_index.append(char_cnt)
        char_cnt += len(token)
        tok_end_to_chartok_index.append(char_cnt - 1)
      tok_cat_text = "".join(cat_tokens)
    else:
      raise NotImplementedError

    N, M = len(paragraph_text), len(tok_cat_text)

    if N > max_N or M > max_M:
      max_N = max(N, max_N)
      max_M = max(M, max_M)
      f = np.zeros((max_N, max_M), dtype=np.float32)
      gc.collect()

    # `g` tracks the type of match:
    # - 0: f[i - 1, j] -> f[i, j]
    # - 1: f[i, j - 1] -> f[i, j]
    # - 2: f[i - 1, j - 1] + match(i, j) -> f[i, j]
    g = {}

    def _lcs_match(max_dist):
      """Customized longest common subsequence."""
      f.fill(0)
      g.clear()

      ### f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
      for i in range(N):

        # unlike standard LCS, this is specifically optimized for the setting
        # assuming the mismatch between tokenized pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
          # out of range
          if j >= M or j < 0: continue

          #### f[i - 1, j] -> f[i, j]
          if i > 0:
            g[(i, j)] = 0
            f[i, j] = f[i - 1, j]

          #### f[i, j - 1] -> f[i, j]
          if j > 0 and f[i, j - 1] > f[i, j]:
            g[(i, j)] = 1
            f[i, j] = f[i, j - 1]

          #### f[i - 1, j - 1] + match(i, j) -> f[i, j]
          f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0

          para_text_i = tokenization.preprocess_text(
              paragraph_text[i], lower=FLAGS.uncased, remove_space=False,
              keep_accents=FLAGS.keep_accents)
          if (para_text_i == tok_cat_text[j] and f_prev + 1 > f[i, j]):
            g[(i, j)] = 2
            f[i, j] = f_prev + 1

    # maximum allowed char index difference
    max_dist = abs(N - M) + 5
    for _ in range(2):
      _lcs_match(max_dist)
      if f[N - 1, M - 1] > 0.8 * N: break
      max_dist *= 2

    # backtrace to fill the two mappings
    orig_to_chartok_index = [None] * N
    chartok_to_orig_index = [None] * M
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
      if (i, j) not in g: break
      if g[(i, j)] == 2:
        orig_to_chartok_index[i] = j
        chartok_to_orig_index[j] = i
        i, j = i - 1, j - 1
      elif g[(i, j)] == 1:
        j = j - 1
      else:
        i = i - 1

    if (all(v is None for v in orig_to_chartok_index) or
        f[N - 1, M - 1] < 0.8 * N):
      print("MISMATCH DETECTED!")
      continue

    # list: token start to original char index
    tok_start_to_orig_index = []
    # list: token end to original char index
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
      start_chartok_pos = tok_start_to_chartok_index[i]
      end_chartok_pos = tok_end_to_chartok_index[i]
      start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                      N, is_start=True)
      end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                    N, is_start=False)

      tok_start_to_orig_index.append(start_orig_pos)
      tok_end_to_orig_index.append(end_orig_pos)

    if not is_training:
      tok_start_position = tok_end_position = None

    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1

    if is_training and not example.is_impossible:
      start_position = example.start_position
      end_position = start_position + len(example.orig_answer_text) - 1

      start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                         is_start=True)
      tok_start_position = chartok_to_tok_index[start_chartok_pos]

      end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                       is_start=False)
      tok_end_position = chartok_to_tok_index[end_chartok_pos]
      assert tok_start_position <= tok_end_position

    all_doc_ids = tokenizer.convert_tokens_to_ids(para_tokens)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_ids) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_ids):
      length = len(all_doc_ids) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_ids):
        break
      start_offset += min(length, doc_stride)

    # For each span, doc_offset is the start index of `para_span`
    # [cls] query [sep] para_span [sep]
    doc_offset = len(query_ids) + 2

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      segment_ids = []
      p_mask = []

      ##### Paragraph specific
      token_is_max_context = {}

      # dict: with-span token start to original char index
      span_tok_start_to_orig_index = {}

      # dict: with-span token end to original char index
      span_tok_end_to_orig_index = {}

      para_span_tokens = []
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        span_token_index = doc_offset + i

        span_tok_start_to_orig_index[span_token_index] = \
            tok_start_to_orig_index[split_token_index]
        span_tok_end_to_orig_index[span_token_index] = \
            tok_end_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[span_token_index] = is_max_context

        para_span_tokens.append(all_doc_ids[split_token_index])

      paragraph_len = len(para_span_tokens)

      # [cls] query [sep] para_span [sep]
      input_ids = ([FLAGS.cls_id] + query_ids + [FLAGS.sep_id] +
                   para_span_tokens + [FLAGS.sep_id])
      segment_ids = ([FLAGS.seg_id_cls] +
                     [FLAGS.seg_id_b] * (len(query_ids) + 1) +
                     [FLAGS.seg_id_a] * (len(para_span_tokens) + 1))
      p_mask = ([0] + [1] * (len(query_ids) + 1) +
                [0] * len(para_span_tokens) + [1])
      cls_index = 0

      # The mask has 0 for real tokens and 1 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [0] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(FLAGS.pad_id)
        input_mask.append(1)
        segment_ids.append(FLAGS.seg_id_pad)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(p_mask) == max_seq_length

      span_is_impossible = example.is_impossible
      start_position = None
      end_position = None
      if is_training and not span_is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          # continue
          span_is_impossible = True
        else:
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and span_is_impossible:
        start_position = cls_index
        end_position = cls_index

      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s", unique_id)

        tf.logging.info(
            "input_ids: %s", " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s", " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s", " ".join([str(x) for x in segment_ids]))

        if is_training and span_is_impossible:
          tf.logging.info("impossible example span")

        if is_training and not span_is_impossible:
          answer_text = tokenizer.convert_ids_to_text(
              input_ids[start_position: (end_position + 1)])
          query_text = tokenizer.convert_ids_to_text(query_ids)
          tf.logging.info("query: %s",
                          tokenization.convert_to_unicode(query_text))
          tf.logging.info("answer: %s",
                          tokenization.convert_to_unicode(answer_text))
          tf.logging.info("truth: %s",
                          tokenization.convert_to_unicode(
                              example.orig_answer_text))

      # With multi processing, the example_index is actually the index within
      # the current process. Therefore we use example_index=None to avoid being
      # used in the future. The current code does not use example_index of
      # training data.
      if is_training:
        feat_example_index = None
      else:
        feat_example_index = example_index

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=feat_example_index,
          doc_span_index=doc_span_index,
          tok_start_to_orig_index=span_tok_start_to_orig_index,
          tok_end_to_orig_index=span_tok_end_to_orig_index,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          p_mask=p_mask,
          segment_ids=segment_ids,
          paragraph_len=paragraph_len,
          cls_index=cls_index,
          start_position=start_position,
          end_position=end_position,
          is_impossible=span_is_impossible,
          regression_tgt=example.regression_tgt)

      # Run callback
      output_fn(feature)

      unique_id += 1
      if span_is_impossible:
        cnt_neg += 1
      else:
        cnt_pos += 1

  tf.logging.info("Total number of instances: {} = pos {} neg {}".format(
      cnt_pos + cnt_neg, cnt_pos, cnt_neg))


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["p_mask"] = create_float_feature(feature.p_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    features["cls_index"] = create_int_feature([feature.cls_index])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_float_feature([impossible])
      features["regression_tgt"] = create_float_feature(
          [feature.regression_tgt])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob"])


_NbestPrediction = collections.namedtuple(
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])


def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length):
  tf.logging.info("Total number of examples: %d", len(all_examples))
  tf.logging.info("Total number of features: %d", len(all_features))
  tf.logging.info("Total number of results: %d", len(all_results))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = float("inf")

    # for each span in the current example
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]

      cur_null_score = result.cls_logits

      # if we could have irrelevant answers, get the min score of irrelevant
      score_null = min(score_null, cur_null_score)

      # beam search
      for i in range(FLAGS.start_n_top):
        for j in range(FLAGS.end_n_top):
          start_log_prob = result.start_top_log_probs[i]
          start_index = result.start_top_index[i]

          if FLAGS.conditional_end:
            j_index = i * FLAGS.end_n_top + j
          else:
            j_index = j

          end_log_prob = result.end_top_log_probs[j_index]
          end_index = result.end_top_index[j_index]

          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if end_index < start_index:
            continue

          length = end_index - start_index + 1
          if length > max_answer_length:
            continue

          if not feature.token_is_max_context.get(start_index, False):
            continue

          if isinstance(feature.tok_start_to_orig_index, list):
            min_idx, max_idx = 0, len(feature.tok_start_to_orig_index)
            if start_index < min_idx or start_index >= max_idx:
              continue
          elif isinstance(feature.tok_start_to_orig_index, dict):
            if start_index not in feature.tok_start_to_orig_index:
              start_keys = list(feature.tok_start_to_orig_index.keys())
              min_idx, max_idx = min(start_keys), max(start_keys)
              continue

          if isinstance(feature.tok_end_to_orig_index, list):
            min_idx, max_idx = 0, len(feature.tok_end_to_orig_index)
            if end_index < min_idx or end_index >= max_idx:
              continue
          elif isinstance(feature.tok_end_to_orig_index, dict):
            if end_index not in feature.tok_end_to_orig_index:
              end_keys = list(feature.tok_end_to_orig_index.keys())
              min_idx, max_idx = min(end_keys), max(end_keys)
              continue

          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_log_prob=start_log_prob,
                  end_log_prob=end_log_prob))

    # sort predictions among all spans
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_log_prob + x.end_log_prob),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]

      tok_start_to_orig_index = feature.tok_start_to_orig_index
      tok_end_to_orig_index = feature.tok_end_to_orig_index
      start_orig_pos = tok_start_to_orig_index[pred.start_index]
      end_orig_pos = tok_end_to_orig_index[pred.end_index]

      paragraph_text = example.paragraph_text
      final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_log_prob=pred.start_log_prob,
              end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="", start_log_prob=-1e6,
                           end_log_prob=-1e6))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None

    if FLAGS.use_answer_class:
      score_diff = score_null
    else:
      score_diff = score_null - best_non_null_entry.start_log_prob - (
          best_non_null_entry.end_log_prob)
    scores_diff_json[example.qas_id] = score_diff
    # Always predict `best_non_null_entry`
    # and the evaluation script will search for the best threshold
    all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  return all_nbest_json, scores_diff_json, all_predictions


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, orig_data):
  """Write final predictions to the json file and log-odds of null if needed."""

  #### Make predictions based on model results
  all_nbest_json, scores_diff_json, all_predictions = make_predictions(
      all_examples, all_features, all_results, n_best_size, max_answer_length)

  #### Write predictions to files
  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

  ### use squad utils to return eval results ###
  if FLAGS.eval_version == "v2":
    if FLAGS.debug_mode:
      tf.logging.info("[DEBUG MODE] write predictions.")
      new_orig_data = []
      for article in orig_data:
        for p in article["paragraphs"]:
          for qa in p["qas"]:
            if qa["id"] in all_predictions:
              new_para = {"qas": [qa]}
              new_article = {"paragraphs": [new_para]}
              new_orig_data.append(new_article)
      orig_data = new_orig_data

    qid_to_has_ans = squad_utils_v2.make_qid_to_has_ans(orig_data)
    exact_raw, f1_raw = squad_utils_v2.get_raw_scores(orig_data,
                                                      all_predictions)
    out_eval = {}

    squad_utils_v2.find_all_best_thresh_v2(
        out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json,
        qid_to_has_ans)
  else:
    out_eval = squad_utils_v1.evaluate(orig_data, all_predictions)

  return out_eval


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def input_fn_builder(input_glob, seq_length, is_training, drop_remainder,
                     num_hosts, num_threads=8):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "cls_index": tf.FixedLenFeature([], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  if FLAGS.use_masked_loss:
    name_to_features["p_mask"] = tf.FixedLenFeature([seq_length], tf.float32)

  if FLAGS.use_answer_class and is_training:
    name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.float32)

  tf.logging.info("Input tfrecord file glob {}".format(input_glob))
  global_input_paths = tf.io.gfile.glob(input_glob)
  tf.logging.info("Find {} input paths {}".format(
      len(global_input_paths), global_input_paths))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # type cast for example
    data_utils.convert_example(example, FLAGS.use_bfloat16)

    return example

  def get_options():
    options = tf.data.Options()
    # Forces map and interleave to be sloppy for enhance performance.
    options.experimental_deterministic = not is_training
    return options

  def input_fn(params):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    else:
      batch_size = FLAGS.predict_batch_size

    # Split tfrecords across hosts
    if num_hosts > 1:
      host_id = params["context"].current_host
      num_files = len(global_input_paths)
      if num_files >= num_hosts:
        num_files_per_host = (num_files + num_hosts - 1) // num_hosts
        my_start_file_id = host_id * num_files_per_host
        my_end_file_id = min((host_id + 1) * num_files_per_host, num_files)
        input_paths = global_input_paths[my_start_file_id: my_end_file_id]
      tf.logging.info("Host {} handles {} files".format(host_id,
                                                        len(input_paths)))
    else:
      input_paths = global_input_paths

    if len(input_paths) == 1:
      d = tf.data.TFRecordDataset(input_paths[0])

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
        d = d.repeat()
    else:
      d = tf.data.Dataset.from_tensor_slices(input_paths)

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn"t matter.
      if is_training:
        # file level shuffle
        d = d.shuffle(len(input_paths)).repeat()

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_threads, len(input_paths))

        d = d.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=cycle_length,
            cycle_length=cycle_length)
        d.with_options(get_options())

        # sample level shuffle
        d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)

    d = d.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder)
    d = d.prefetch(1024)

    return d

  return input_fn


def get_model_fn():
  """Create model function for TPU estimator."""
  def model_fn(features, labels, mode, params):
    """Model computational graph."""
    del labels
    del params

    #### Build model
    if FLAGS.model_config:
      net_config = modeling.ModelConfig.init_from_json(FLAGS.model_config)
    else:
      net_config = modeling.ModelConfig.init_from_flags()
    net_config.to_json(os.path.join(FLAGS.model_dir, "net_config.json"))
    model = modeling.FunnelTFM(net_config)

    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Get loss from inputs
    @model_utils.bf16_decorator
    def squad_func(features):
      """Get squad outputs."""
      inputs = features["input_ids"]
      seg_id = features["segment_ids"]
      input_mask = features["input_mask"]
      para_mask = features["p_mask"]
      cls_index = tf.reshape(features["cls_index"], [-1])

      if is_training and FLAGS.conditional_end:
        start_positions = tf.reshape(features["start_positions"], [-1])
      else:
        start_positions = None

      with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        outputs = model.get_squad_loss(
            inputs, cls_index, para_mask, is_training, seg_id=seg_id,
            input_mask=input_mask, start_positions=start_positions,
            use_tpu=FLAGS.use_tpu, use_bfloat16=FLAGS.use_bfloat16)

      return outputs

    outputs = squad_func(features)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: %d", num_params)
    if FLAGS.verbose:
      format_str = "{{:<{0}s}}\t{{}}".format(
          max([len(v.name) for v in tf.trainable_variables()]))
      for v in tf.trainable_variables():
        tf.logging.info(format_str.format(v.name, v.get_shape()))

    scaffold_fn = None

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      if FLAGS.init_checkpoint:
        tf.logging.info("init_checkpoint not being used in predict mode.")

      predictions = {
          "unique_ids": features["unique_ids"],
          "start_top_index": outputs["start_top_index"],
          "start_top_log_probs": outputs["start_top_log_probs"],
          "end_top_index": outputs["end_top_index"],
          "end_top_log_probs": outputs["end_top_log_probs"],
          "cls_logits": outputs["cls_logits"]
      }

      if FLAGS.use_tpu:
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
      return output_spec

    ### Compute loss
    seq_length = tf.shape(features["input_ids"])[1]
    def compute_loss(log_probs, positions):
      """Compute squad loss from model outputs."""
      one_hot_positions = tf.one_hot(
          positions, depth=seq_length, dtype=tf.float32)

      loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
      loss = tf.reduce_mean(loss)

      return loss

    start_loss = compute_loss(
        outputs["start_log_probs"], features["start_positions"])
    end_loss = compute_loss(
        outputs["end_log_probs"], features["end_positions"])

    total_loss = (start_loss + end_loss) * FLAGS.ans_loss_weight

    if FLAGS.use_answer_class:
      cls_logits = outputs["cls_logits"]
      is_impossible = tf.cast(
          tf.reshape(features["is_impossible"], [-1]), cls_logits.dtype)
      regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=is_impossible, logits=cls_logits)
      regression_loss = tf.reduce_mean(regression_loss)

      # Note: by default multiply the loss by 0.5 so that the scale is
      # comparable to start_loss and end_loss
      total_loss += regression_loss * FLAGS.cls_loss_weight

    # Get train op from loss
    train_op, monitor_dict = optimization.get_train_op(total_loss)

    #### load pretrained models
    scaffold_fn = model_utils.custom_initialization(FLAGS.init_global_vars)

    #### Constructing training TPUEstimatorSpec with new cache.
    if FLAGS.use_tpu:
      host_call = model_utils.construct_scalar_host_call(
          monitor_dict=monitor_dict,
          model_dir=FLAGS.model_dir,
          prefix="train/",
          reduce_fn=tf.reduce_mean)

      train_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      train_spec = tf.estimator.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op)

    return train_spec

  return model_fn


def _get_tok_identifier():
  tok_identifier = os.path.basename(FLAGS.tokenizer_path)
  return tok_identifier


def preprocess_train(tokenizer):
  """Preprocess training set."""
  tf.logging.info("Read examples from %s", FLAGS.train_file)
  train_examples = read_squad_examples(FLAGS.train_file, is_training=True)
  train_examples = train_examples[FLAGS.proc_id::FLAGS.num_proc]

  # Pre-shuffle the input to avoid having to make a very large shuffle
  # buffer in in the `input_fn`.
  np.random.shuffle(train_examples)

  if FLAGS.debug_mode:
    tf.logging.info("[DEBUG MODE] preprocess train.")
    train_examples = train_examples[:100]

  train_rec_file = format_train_record_name(FLAGS.proc_id)
  tf.logging.info("Write to %s", train_rec_file)

  train_writer = FeatureWriter(
      filename=train_rec_file,
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()


def format_train_record_name(proc_id):
  """Format processed training file names."""
  tok_identifier = _get_tok_identifier()
  version_str = "v1." if FLAGS.train_version == "v1" else ""
  rec_base = "{}.{}.slen-{}.qlen-{}.train.{}tfrecord".format(
      tok_identifier, proc_id, FLAGS.max_seq_length,
      FLAGS.max_query_length, version_str)

  return os.path.join(FLAGS.output_dir, rec_base)


def preprocess_eval(tokenizer):
  """Preprocess eval examples."""
  # preprocess eval examples
  eval_examples = read_squad_examples(FLAGS.predict_file, is_training=False)

  if FLAGS.debug_mode:
    tf.logging.info("[DEBUG MODE] preprocess eval.")
    eval_examples = eval_examples[: 100]

  # Default setting
  eval_rec_file, eval_feature_file = format_eval_names()

  eval_writer = FeatureWriter(filename=eval_rec_file, is_training=False)
  eval_features = []

  def append_feature(feature):
    eval_features.append(feature)
    eval_writer.process_feature(feature)

  convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=False,
      output_fn=append_feature)
  eval_writer.close()

  with tf.io.gfile.GFile(eval_feature_file, "wb") as fout:
    pickle.dump(eval_features, fout)


def format_eval_names():
  """Format processed eval file names."""
  tok_identifier = _get_tok_identifier()
  version_str = "v1." if FLAGS.eval_version == "v1" else ""
  eval_rec_base = "{}.slen-{}.qlen-{}.eval.{}tfrecord".format(
      tok_identifier, FLAGS.max_seq_length, FLAGS.max_query_length,
      version_str)
  eval_feature_base = "{}.slen-{}.qlen-{}.eval.features.{}pkl".format(
      tok_identifier, FLAGS.max_seq_length, FLAGS.max_query_length,
      version_str)

  eval_rec_file = os.path.join(FLAGS.output_dir, eval_rec_base)
  eval_feature_file = os.path.join(FLAGS.output_dir, eval_feature_base)

  return eval_rec_file, eval_feature_file


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Load tokenizer
  tokenizer = tokenization.get_tokenizer()
  data_utils.setup_special_ids(tokenizer)

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  #### Run preprocessing
  if FLAGS.prepro_split:
    if FLAGS.prepro_split == "train":
      preprocess_train(tokenizer)
    elif FLAGS.prepro_split == "eval":
      preprocess_eval(tokenizer)
    return

  #### Validate flags
  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_ensemble:
    raise ValueError(
        "One of `do_train`, `do_predict` and `do_ensemble` must be True.")

  if not tf.io.gfile.exists(FLAGS.predict_dir):
    tf.io.gfile.makedirs(FLAGS.predict_dir)

  # Model function
  model_fn = get_model_fn()

  # TPU Configuration
  run_config = model_utils.get_run_config()

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  if FLAGS.use_tpu:
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

  ###### Actual training and evaluation ######
  if FLAGS.do_train:
    train_input_fn = input_fn_builder(
        input_glob=format_train_record_name("*"),
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        num_hosts=FLAGS.num_hosts)

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    tf.io.gfile.makedirs(os.path.join(FLAGS.model_dir, "done"))

  if FLAGS.do_predict or FLAGS.do_ensemble:
    # load eval examples
    eval_examples = read_squad_examples(FLAGS.predict_file, is_training=False)

    if FLAGS.debug_mode:
      tf.logging.info("[DEBUG MODE] do predict.")
      eval_examples = eval_examples[: 100]

    # tfrecord and eval feature file path
    eval_rec_file, eval_feature_file = format_eval_names()

    assert (tf.io.gfile.exists(eval_rec_file) and
            tf.io.gfile.exists(eval_feature_file))
    tf.logging.info("Loading eval features from %s", eval_feature_file)
    with tf.io.gfile.GFile(eval_feature_file, "rb") as fin:
      eval_features = pickle.load(fin)

    with tf.io.gfile.GFile(FLAGS.predict_file) as f:
      orig_data = json.load(f)["data"]

    eval_input_fn = input_fn_builder(
        input_glob=eval_rec_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        num_hosts=1)

  # Predicting with a single model
  if FLAGS.do_predict:
    num_mins_waited = 0
    last_eval_step = FLAGS.eval_start_step

    tf.logging.info("Start evaluation")
    eval_results = []
    while True:
      if last_eval_step >= FLAGS.eval_end_step: break

      # Gather a list of checkpoints to evaluate
      steps_and_files = []
      try:
        filenames = sorted(tf.io.gfile.listdir(FLAGS.model_dir))
      except tf.errors.NotFoundError:
        filenames = []
        tf.logging.info("`model_dir` does not exist yet...")

      for filename in filenames:
        if filename.endswith(".index"):
          cur_filename = os.path.join(FLAGS.model_dir, filename[:-6])
          global_step = int(cur_filename.split("-")[-1])
          if (global_step <= last_eval_step or
              global_step > FLAGS.eval_end_step):
            continue
          tf.logging.info("Add %s to eval list.", cur_filename)
          steps_and_files.append([global_step, cur_filename])

      # Get empty list of checkpoints
      if not steps_and_files:
        # Training job is done: stop evaluation
        if tf.io.gfile.exists(os.path.join(FLAGS.model_dir, "done")):
          break
        # Wait for 60 seconds
        else:
          time.sleep(60)
          num_mins_waited += 1.0
          tf.logging.info("Waited {:.1f} mins".format(num_mins_waited))
      else:
        num_mins_waited = 0

      # Evaluate the current list of checkpoints
      for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
        cur_results = []
        for result in estimator.predict(
            input_fn=eval_input_fn,
            checkpoint_path=filename,
            yield_single_examples=True):

          if len(cur_results) % 1000 == 0:
            tf.logging.info("Processing example: %d", len(cur_results))

          unique_id = int(result["unique_ids"])
          start_top_log_probs = (
              [float(x) for x in result["start_top_log_probs"].flat])
          start_top_index = [int(x) for x in result["start_top_index"].flat]
          end_top_log_probs = (
              [float(x) for x in result["end_top_log_probs"].flat])
          end_top_index = [int(x) for x in result["end_top_index"].flat]

          cls_logits = float(result["cls_logits"].flat[0])

          cur_results.append(
              RawResult(
                  unique_id=unique_id,
                  start_top_log_probs=start_top_log_probs,
                  start_top_index=start_top_index,
                  end_top_log_probs=end_top_log_probs,
                  end_top_index=end_top_index,
                  cls_logits=cls_logits))

        output_prediction_file = os.path.join(
            FLAGS.predict_dir, "{}.predictions.json".format(global_step))
        output_nbest_file = os.path.join(
            FLAGS.predict_dir, "{}.nbest_predictions.json".format(global_step))
        output_null_log_odds_file = os.path.join(
            FLAGS.predict_dir, "{}.null_odds.json".format(global_step))

        ret = write_predictions(eval_examples, eval_features, cur_results,
                                FLAGS.n_best_size, FLAGS.max_answer_length,
                                output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file,
                                orig_data)

        ret["step"] = global_step
        ret["path"] = filename
        eval_results.append(ret)

        last_eval_step = max(last_eval_step, global_step)

        # Log current result
        tf.logging.info("=" * 80)
        log_str = "Result of step {} | ".format(global_step)
        for key, val in eval_results[-1].items():
          log_str += "{} {} | ".format(key, val)
        tf.logging.info(log_str)
        tf.logging.info("=" * 80)

    # Log the best result
    if FLAGS.eval_version == "v1":
      FLAGS.target_eval_key = "f1"
    eval_results.sort(key=lambda x: x[FLAGS.target_eval_key], reverse=True)
    tf.logging.info("=" * 80)
    log_str = "Best result | "
    for key, val in eval_results[0].items():
      log_str += "{} {} | ".format(key, val)
    tf.logging.info(log_str)
    tf.logging.info("=" * 80)

    # Save best eval result to model directory
    best_result = eval_results[0]

    save_path = os.path.join(FLAGS.model_dir, "best_result.json")
    tf.logging.info("Dump eval results to %s", save_path)
    tf.logging.info(best_result)

    with tf.io.gfile.GFile(save_path, "w") as fp:
      json.dump(best_result, fp, indent=4)

    # Clean all model ckpts
    if not FLAGS.retain_all:
      for idx, ret in enumerate(eval_results):
        if FLAGS.retain_best and idx == 0:
          continue
        for suffix in [".index", ".meta", ".data-00000-of-00001"]:
          tf.io.gfile.remove(ret["path"] + suffix)

    return

  # Predicting with an ensemble of models
  if FLAGS.do_ensemble:
    #### Used to store the ensemble scores
    all_nbest_json = collections.defaultdict(lambda:
                                             collections.defaultdict(float))
    all_na_json = collections.defaultdict(float)

    #### For each model we hope to ensemble
    ensemble_ckpts = FLAGS.ensemble_ckpts.split(",")
    eval_ckpt_num = len(ensemble_ckpts)
    for cur_ckpt_path in ensemble_ckpts:
      cur_results = []
      for result in estimator.predict(
          input_fn=eval_input_fn,
          checkpoint_path=cur_ckpt_path,
          yield_single_examples=True):
        if len(cur_results) % 100 == 0:
          tf.logging.info("Processing example: %d", len(cur_results))

        unique_id = int(result["unique_ids"])
        start_top_log_probs = (
            [float(x) for x in result["start_top_log_probs"].flat])
        start_top_index = [int(x) for x in result["start_top_index"].flat]
        end_top_log_probs = (
            [float(x) for x in result["end_top_log_probs"].flat])
        end_top_index = [int(x) for x in result["end_top_index"].flat]

        cls_logits = float(result["cls_logits"].flat[0])

        cur_results.append(
            RawResult(
                unique_id=unique_id,
                start_top_log_probs=start_top_log_probs,
                start_top_index=start_top_index,
                end_top_log_probs=end_top_log_probs,
                end_top_index=end_top_index,
                cls_logits=cls_logits))

      # Get the prediction of the current model
      cur_nbest_json, cur_na_json, _ = make_predictions(
          eval_examples, eval_features, cur_results,
          FLAGS.n_best_size, FLAGS.max_answer_length)

      # Accumulate current prediction scores into the ensemble scores
      for qid, nbest_entries in cur_nbest_json.items():
        for nbest_entry in nbest_entries:
          all_nbest_json[qid][nbest_entry["text"]] += nbest_entry["probability"]
      for qid, na_score in cur_na_json.items():
        all_na_json[qid] += na_score / eval_ckpt_num

    # Make prediction based on the accumulated ensemble scores
    all_predictions = {}
    for qid, nbest_entries in all_nbest_json.items():
      answers = sorted(nbest_entries.keys(),
                       key=lambda x: nbest_entries[x], reverse=True)
      all_predictions[qid] = answers[0]

    if FLAGS.search_threshold:
      qid_to_has_ans = squad_utils_v2.make_qid_to_has_ans(orig_data)
      exact_raw, f1_raw = squad_utils_v2.get_raw_scores(orig_data,
                                                        all_predictions)
      out_eval = {}

      squad_utils_v2.find_all_best_thresh_v2(
          out_eval, all_predictions, exact_raw, f1_raw, all_na_json,
          qid_to_has_ans)
      best_th = out_eval["best_f1_thresh"]
      tf.logging.info("=" * 80)
      tf.logging.info(out_eval)
      tf.logging.info("Find best NA threshold = %.3f", best_th)
    else:
      best_th = FLAGS.na_threshold
      tf.logging.info("=" * 80)
      tf.logging.info("Use provided NA threshold = %.3f", best_th)

    for qid in all_predictions:
      if all_na_json[qid] > best_th:
        all_predictions[qid] = ""

    if not tf.io.gfile.exists(os.path.dirname(FLAGS.ensemble_output_file)):
      tf.io.gfile.makedirs(os.path.dirname(FLAGS.ensemble_output_file))

    with tf.io.gfile.GFile(FLAGS.ensemble_output_file, "w") as f:
      json.dump(all_predictions, f)

    return


if __name__ == "__main__":
  tf.app.run()
