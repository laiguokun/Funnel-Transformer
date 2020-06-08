"""Finetune for classification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import math
import os
import time

from absl import flags
import absl.logging as _logging

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import tensorflow.compat.v1 as tf

import data_utils
import metric_ops
import modeling
import model_utils
import optimization
import classifier_utils
import tokenization

logger = tf.get_logger()
logger.propagate = False
tf.disable_v2_behavior()


##### I/O related
flags.DEFINE_bool("verbose", default=False,
                  help="Whether to print additional information.")
# Basic paths
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("eval_model_dir", "",
                    help="Evaluate the ckpts in the given directory.")
flags.DEFINE_string("data_dir", default="",
                    help="Directory for input data.")

# Init checkpoint
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("model_config", "",
                    help="FunnelTFM model configuration.")
flags.DEFINE_bool("init_global_vars", default=False,
                  help="If true, init all global vars. If false, init "
                  "trainable vars only.")

# I/O options
flags.DEFINE_bool("retain_all", default=False,
                  help="Retain the best checkpoint.")
flags.DEFINE_bool("retain_best", default=False,
                  help="Retain the best checkpoint.")
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")
flags.DEFINE_bool("overwrite", False,
                  help="Whether to overwrite exist model dir.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2, 16 for TPU v3.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# experiment options
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=40000,
                     help="Number of training steps")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of ckpts to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                     "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=128,
                     help="batch size for training")

# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_bool("do_submit", default=False, help="Produce submission files")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_start_step", default=-1,
                     help="Start step for evaluation")
flags.DEFINE_integer("eval_end_step", default=10000000,
                     help="End step for evaluation")
flags.DEFINE_integer("eval_batch_size", default=8,
                     help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=8,
                     help="batch size for prediction.")
flags.DEFINE_string("submit_dir", default=None,
                    help="Dir for saving submission files.")

# task specific
flags.DEFINE_string("task_name", default=None, help="Task name")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_bool("shuffle_examples", default=True,
                  help="Whether to shuffle examples.")
flags.DEFINE_integer("num_passes", default=1,
                     help="Num passes for processing training data. "
                     "This is use to batch data without loss for TPUs.")

# model
flags.DEFINE_string("cls_scope", default=None,
                    help="Classifier layer scope.")

# debug
flags.DEFINE_bool("debug_mode", False, "Run the debug mode.")

FLAGS = flags.FLAGS


def simple_accuracy(preds, labels):
  return (preds == labels).mean()


def accu_and_f1(preds, labels):
  accu = simple_accuracy(preds, labels)
  f1 = f1_score(y_true=labels, y_pred=preds)
  return {
      "accu": accu,
      "f1": f1,
      "accu_and_f1": (accu + f1) / 2,
  }


def pearson_and_spearman(preds, labels):
  pearson_corr = pearsonr(preds, labels)[0]
  spearman_corr = spearmanr(preds, labels)[0]
  return {
      "pearson": pearson_corr,
      "spearmanr": spearman_corr,
      "corr": (pearson_corr + spearman_corr) / 2,
  }


def compute_metrics(task_name, preds, labels):
  """doc."""
  task_name = task_name.lower()
  assert len(preds) == len(labels)
  if task_name == "cola":
    return {"corr": matthews_corrcoef(labels, preds),
            "accu": simple_accuracy(preds, labels)}
  elif task_name == "sst-2":
    return {"accu": simple_accuracy(preds, labels)}
  elif task_name == "mrpc":
    return accu_and_f1(preds, labels)
  elif task_name == "sts-b":
    return pearson_and_spearman(preds, labels)
  elif task_name == "qqp":
    return accu_and_f1(preds, labels)
  elif task_name == "mnli_matched":
    return {"accu": simple_accuracy(preds, labels)}
  elif task_name == "mnli_mismatched":
    return {"accu": simple_accuracy(preds, labels)}
  elif task_name == "qnli":
    return {"accu": simple_accuracy(preds, labels)}
  elif task_name == "rte":
    return {"accu": simple_accuracy(preds, labels)}
  elif task_name == "wnli":
    return {"accu": simple_accuracy(preds, labels)}
  else:
    raise KeyError(task_name)


def get_main_metric_key(task_name):
  task_name = task_name.lower()
  if task_name in ["cola", "sts-b"]:
    return "corr"
  elif task_name in ["mrpc", "qqp"]:
    return "accu_and_f1"
  else:
    return "accu"


def _compute_metric_based_on_keys(key, tp, fp, tn, fn):
  """Compute metrics."""

  if key == "corr":
    # because this is only used for threshold search,
    # we only care about matthews_corrcoef but not pearsons
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
      corr = 0.0
    else:
      corr = (tp * tn - fp * fn) / math.sqrt(
          (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return corr
  else:
    if tp + fp + tn + fn == 0:
      accu = 0
    else:
      accu = (tp + tn) / (tp + fp + tn + fn)

    if key == "accu":
      return accu

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    if key == "accu_and_f1":
      return (accu + f1) / 2
    elif key == "f1":
      return f1

    raise NotImplementedError


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


#########################################
########## Text classification ##########
#########################################
class Yelp5Processor(DataProcessor):
  """Yelp-5."""

  def get_train_examples(self, data_dir):
    """See base class."""
    examples = self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.csv"),
                       quotechar="\"",
                       delimiter=","), "train")
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.csv"),
                       quotechar="\"",
                       delimiter=","), "test")
    return examples

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%d" % (set_type, i)
      label = line[0]
      text_a = classifier_utils.clean_web_text(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["1", "2", "3", "4", "5"]


class Yelp2Processor(Yelp5Processor):
  """Yelp-2."""

  def get_labels(self):
    return ["1", "2"]


class DbpediaProcessor(Yelp5Processor):
  """DBPedia."""

  def get_labels(self):
    return list(map(str, range(1, 15)))

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%d" % (set_type, i)
      label = line[0]
      text_a = classifier_utils.clean_web_text(line[1])
      text_b = classifier_utils.clean_web_text(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class Amazon5Processor(DbpediaProcessor):
  """Amazon-5."""

  def get_labels(self):
    """See base class."""
    return ["1", "2", "3", "4", "5"]


class Amazon2Processor(DbpediaProcessor):
  """Amazon-2."""

  def get_labels(self):
    return ["1", "2"]


class AgProcessor(DbpediaProcessor):
  """AG."""

  def get_labels(self):
    return ["1", "2", "3", "4"]


class ImdbProcessor(DataProcessor):
  """IMDB."""

  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test"))

  def _create_examples(self, data_dir):
    """Create examples."""
    examples = []
    for label in ["neg", "pos"]:
      cur_dir = os.path.join(data_dir, label)
      for filename in tf.io.gfile.listdir(cur_dir):
        if not filename.endswith("txt"): continue
        path = os.path.join(cur_dir, filename)
        with tf.io.gfile.GFile(path) as f:
          text = classifier_utils.clean_web_text(f.read().strip())
        examples.append(InputExample(
            guid="unused_id", text_a=text, text_b=None, label=label))
    return examples


###################################
########## GLUE datasets ##########
###################################
class GLUEProcessor(DataProcessor):
  """GLUE."""

  def __init__(self):
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"
    self.label_column = None
    self.text_a_column = None
    self.text_b_column = None
    self.contains_header = True
    self.test_text_a_column = None
    self.test_text_b_column = None
    self.test_contains_header = True

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.dev_file)), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    if self.test_text_a_column is None:
      self.test_text_a_column = self.text_a_column
    if self.test_text_b_column is None:
      self.test_text_b_column = self.text_b_column

    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
                  self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
                  self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning("Incomplete line, ignored.")
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning("Incomplete line, ignored.")
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning("Incomplete line, ignored.")
          continue
        label = line[self.label_column]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class ColaProcessor(GLUEProcessor):
  """CoLA."""

  def __init__(self):
    super(ColaProcessor, self).__init__()
    self.label_column = 1
    self.text_a_column = 3
    self.contains_header = False
    self.test_text_a_column = 1
    self.test_contains_header = True


class MrpcProcessor(GLUEProcessor):
  """MRPC."""

  def __init__(self):
    super(MrpcProcessor, self).__init__()
    self.label_column = 0
    self.text_a_column = 3
    self.text_b_column = 4


class MnliMatchedProcessor(GLUEProcessor):
  """MNLI-matched."""

  def __init__(self):
    super(MnliMatchedProcessor, self).__init__()
    self.dev_file = "dev_matched.tsv"
    self.test_file = "test_matched.tsv"
    self.label_column = -1
    self.text_a_column = 8
    self.text_b_column = 9

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]


class MnliMismatchedProcessor(MnliMatchedProcessor):
  """MNLI-mismatched."""

  def __init__(self):
    super(MnliMismatchedProcessor, self).__init__()
    self.dev_file = "dev_mismatched.tsv"
    self.test_file = "test_mismatched.tsv"


class QnliProcessor(GLUEProcessor):
  """QNLI."""

  def __init__(self):
    super(QnliProcessor, self).__init__()
    self.label_column = 3
    self.text_a_column = 1
    self.text_b_column = 2

  def get_labels(self):
    return ["entailment", "not_entailment"]


class QqpProcessor(GLUEProcessor):
  """QQP."""

  def __init__(self):
    super(QqpProcessor, self).__init__()
    self.label_column = 5
    self.text_a_column = 3
    self.text_b_column = 4
    self.test_text_a_column = 1
    self.test_text_b_column = 2


class RteProcessor(GLUEProcessor):
  """RTE."""

  def __init__(self):
    super(RteProcessor, self).__init__()
    self.label_column = 3
    self.text_a_column = 1
    self.text_b_column = 2

  def get_labels(self):
    return ["entailment", "not_entailment"]


class SnliProcessor(MnliMatchedProcessor):
  """SNLI."""

  def __init__(self):
    super(SnliProcessor, self).__init__()
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"


class Sst2Processor(GLUEProcessor):
  """SST-2."""

  def __init__(self):
    super(Sst2Processor, self).__init__()
    self.label_column = 1
    self.text_a_column = 0
    self.test_text_a_column = 1
    self.train_file = "train.detok.tsv"
    self.dev_file = "dev.detok.tsv"
    self.test_file = "test.detok.tsv"


class WnliProcessor(GLUEProcessor):
  """WNLI."""

  def __init__(self):
    super(WnliProcessor, self).__init__()
    self.label_column = 3
    self.text_a_column = 1
    self.text_b_column = 2


class StsbProcessor(GLUEProcessor):
  """STS-B."""

  def __init__(self):
    super(StsbProcessor, self).__init__()
    self.label_column = 9
    self.text_a_column = 7
    self.text_b_column = 8

  def get_labels(self):
    return [0.0]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
                  self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
                  self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning("Incomplete line, ignored.")
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning("Incomplete line, ignored.")
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning("Incomplete line, ignored.")
          continue
        label = float(line[self.label_column])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class DiagnosticProcessor(GLUEProcessor):

  def __init__(self):
    super(DiagnosticProcessor, self).__init__()
    self.test_text_a_column = 1
    self.test_text_b_column = 2
    self.test_file = "diagnostic.tsv"

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]


####################################
########## Input function ##########
####################################
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenize_fn, output_file,
    num_passes=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  tf.logging.info("Create new tfrecord {}.".format(output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  examples *= num_passes

  stat = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature, example_len = classifier_utils.convert_single_example(
        ex_index, example, label_list, max_seq_length, tokenize_fn)
    stat.append(example_len)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if label_list is not None:
      features["label_ids"] = create_int_feature([feature.label_id])
    else:
      features["label_ids"] = create_float_feature([float(feature.label_id)])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

  hist, bins = np.histogram(stat,
                            bins=[0, 128, 256, 512, 1024, 102400])
  percent = hist / np.sum(hist)
  tf.logging.info("***** Example length histogram *****")
  for pct, l, r in zip(percent, bins[:-1], bins[1:]):
    tf.logging.info("  - [%d, %d]: %.4f", l, r, pct)


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, regression=False):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }
  if regression:
    name_to_features["label_ids"] = tf.io.FixedLenFeature([], tf.float32)

  tf.logging.info("Input tfrecord file {}".format(input_file))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    data_utils.convert_example(example, FLAGS.use_bfloat16)

    return example

  def input_fn(params):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    elif FLAGS.do_eval:
      batch_size = FLAGS.eval_batch_size
    else:
      batch_size = FLAGS.predict_batch_size

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
      d = d.repeat()

    d = d.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder)

    return d

  return input_fn


def get_model_fn(n_class):
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
    def cls_or_reg_loss_func(features, model):
      """Get classification loss."""
      inputs = features["input_ids"]
      seg_id = features["segment_ids"]
      input_mask = features["input_mask"]
      labels = tf.reshape(features["label_ids"], [-1])

      with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        scope = FLAGS.cls_scope if FLAGS.cls_scope else FLAGS.task_name.lower()
        if FLAGS.task_name.lower() == "sts-b":
          labels = tf.cast(labels, tf.float32)
          per_example_loss, logits = model.get_regression_loss(
              labels, inputs, is_training, scope, seg_id=seg_id,
              input_mask=input_mask, use_tpu=FLAGS.use_tpu,
              use_bfloat16=FLAGS.use_bfloat16)
        else:
          per_example_loss, logits = model.get_classification_loss(
              labels, inputs, n_class, is_training, scope,
              seg_id=seg_id, input_mask=input_mask, use_tpu=FLAGS.use_tpu,
              use_bfloat16=FLAGS.use_bfloat16)

        return per_example_loss, logits

    per_example_loss, logits = cls_or_reg_loss_func(features, model)
    total_loss = tf.reduce_mean(per_example_loss)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))
    if FLAGS.verbose:
      format_str = "{{:<{0}s}}\t{{}}".format(
          max([len(v.name) for v in tf.trainable_variables()]))
      for v in tf.trainable_variables():
        tf.logging.info(format_str.format(v.name, v.get_shape()))

    #### Load pretrained models
    scaffold_fn = model_utils.custom_initialization(FLAGS.init_global_vars)

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      assert FLAGS.num_hosts == 1

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        """Metrics to record during evaluation."""
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        eval_input_dict = {
            "labels": label_ids,
            "predictions": predictions,
            "weights": is_real_example
        }
        accuracy = tf.metrics.accuracy(**eval_input_dict)
        tp = tf.metrics.true_positives(**eval_input_dict)
        fp = tf.metrics.false_positives(**eval_input_dict)
        tn = tf.metrics.true_negatives(**eval_input_dict)
        fn = tf.metrics.false_negatives(**eval_input_dict)

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "eval_tp": tp,
            "eval_fp": fp,
            "eval_tn": tn,
            "eval_fn": fn
        }

      def regression_metric_fn(per_example_loss, label_ids, logits,
                               is_real_example):
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        eval_pearsonr = metric_ops.streaming_pearson_correlation(
            logits, label_ids, weights=is_real_example)
        return {"eval_loss": loss, "eval_pearsonr": eval_pearsonr}

      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

      #### Constructing evaluation TPUEstimatorSpec with new cache.
      label_ids = tf.cast(tf.reshape(features["label_ids"], [-1]), tf.float32)

      if FLAGS.task_name.lower() == "sts-b":
        metric_fn = regression_metric_fn
      metric_args = [per_example_loss, label_ids, logits, is_real_example]

      if FLAGS.use_tpu:
        eval_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=(metric_fn, metric_args),
            scaffold_fn=scaffold_fn)
      else:
        eval_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(*metric_args))

      return eval_spec

    elif mode == tf.estimator.ModeKeys.PREDICT:
      label_ids = tf.reshape(features["label_ids"], [-1])

      predictions = {
          "logits": logits,
          "labels": label_ids,
          "is_real": features["is_real_example"]
      }

      if FLAGS.use_tpu:
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
      return output_spec

    train_op, monitor_dict = optimization.get_train_op(total_loss)

    #### Constructing training TPUEstimatorSpec
    if FLAGS.use_tpu:
      #### Creating host calls
      if ("label_ids" in features and
          FLAGS.task_name.lower() not in ["sts-b"]):
        label_ids = tf.reshape(features["label_ids"], [-1])
        predictions = tf.argmax(logits, axis=-1, output_type=label_ids.dtype)
        is_correct = tf.equal(predictions, label_ids)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        monitor_dict["accuracy"] = accuracy

        host_call = model_utils.construct_scalar_host_call(
            monitor_dict=monitor_dict,
            model_dir=FLAGS.model_dir,
            prefix="train/",
            reduce_fn=tf.reduce_mean)
      else:
        host_call = None

      train_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      train_spec = tf.estimator.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op)

    return train_spec

  return model_fn


def create_tfrecord(task_name, split, processor, tokenizer, pad_for_eval=False):
  """Create tfrecord for a specific split of a task."""
  if task_name != FLAGS.task_name and task_name == "diagnostic":
    # a corner case
    data_dir = os.path.join(os.path.dirname(FLAGS.data_dir), task_name)
    output_dir = os.path.join(os.path.dirname(FLAGS.output_dir), task_name)
  else:
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)

  # Path to the tfrecord & meta data
  tok_basename = os.path.basename(FLAGS.tokenizer_path)
  file_base = "{}.len-{}.{}.tfrecord".format(
      tok_basename, FLAGS.max_seq_length, split)
  file_path = os.path.join(output_dir, file_base)
  meta_path = file_path.replace("tfrecord", "meta.json")

  if (FLAGS.overwrite_data or not tf.io.gfile.exists(file_path)
      or not tf.io.gfile.exists(meta_path)):
    # Load examples
    if split == "train":
      examples = processor.get_train_examples(data_dir)
    elif split == "dev":
      examples = processor.get_dev_examples(data_dir)
    elif split == "test":
      examples = processor.get_test_examples(data_dir)
    else:
      raise NotImplementedError

    num_real_examples = len(examples)
    if split == "train" and FLAGS.shuffle_examples:
      np.random.shuffle(examples)
    if pad_for_eval:
      while len(examples) % FLAGS.eval_batch_size != 0:
        examples.append(classifier_utils.PaddingInputExample())
    num_examples = len(examples)

    meta_dict = {"num_real_examples": num_real_examples,
                 "num_examples": num_examples}
    with tf.io.gfile.GFile(meta_path, "w") as fp:
      json.dump(meta_dict, fp, indent=4)
  else:
    with tf.io.gfile.GFile(meta_path, "r") as fp:
      meta_dict = json.load(fp)
    num_examples = meta_dict["num_examples"]
    num_real_examples = meta_dict["num_real_examples"]

  tf.logging.info("Num of %s samples:  %d real / %d total.", split,
                  num_real_examples, num_examples)

  if FLAGS.overwrite_data or not tf.io.gfile.exists(file_path):
    tokenize_fn = tokenizer.convert_text_to_ids
    label_list = processor.get_labels()
    if task_name == "sts-b":
      file_based_convert_examples_to_features(
          examples, None, FLAGS.max_seq_length, tokenize_fn,
          file_path, FLAGS.num_passes)
    else:
      file_based_convert_examples_to_features(
          examples, label_list, FLAGS.max_seq_length, tokenize_fn,
          file_path, FLAGS.num_passes)
  else:
    tf.logging.info("Do not overwrite existing tfrecord %s.", file_path)

  return num_examples, file_path


def main(_):
  tf.reset_default_graph()
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate flags
  if (not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_submit):
    raise ValueError(
        "At least one of `do_train`, `do_eval, or `do_submit` "
        "must be True.")

  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

  #### Load tokenizer
  tokenizer = tokenization.get_tokenizer()
  data_utils.setup_special_ids(tokenizer)

  #### Directories
  if FLAGS.do_eval and FLAGS.overwrite:
    eval_dir = os.path.join(FLAGS.model_dir, "eval")
    if tf.io.gfile.exists(eval_dir):
      tf.io.gfile.rmtree(eval_dir)

  if FLAGS.do_submit:
    if FLAGS.submit_dir:
      submit_dir = FLAGS.submit_dir
    else:
      submit_dir = os.path.join(FLAGS.model_dir, "submit")
    if not tf.io.gfile.exists(submit_dir):
      tf.io.gfile.makedirs(submit_dir)

  processors = {
      "cola": ColaProcessor,
      "mnli_matched": MnliMatchedProcessor,
      "mnli_mismatched": MnliMismatchedProcessor,
      "mrpc": MrpcProcessor,
      "qqp": QqpProcessor,
      "rte": RteProcessor,
      "snli": SnliProcessor,
      "sst-2": Sst2Processor,
      "wnli": WnliProcessor,
      "qnli": QnliProcessor,
      "sts-b": StsbProcessor,
      "diagnostic": DiagnosticProcessor,  # only used during test time
      "yelp5": Yelp5Processor,
      "yelp2": Yelp2Processor,
      "amazon5": Amazon5Processor,
      "amazon2": Amazon2Processor,
      "imdb": ImdbProcessor,
      "ag": AgProcessor,
      "dbpedia": DbpediaProcessor
  }
  submit_name_mapping = {
      "cola": "CoLA",
      "mnli_matched": "MNLI-m",
      "mnli_mismatched": "MNLI-mm",
      "mrpc": "MRPC",
      "qqp": "QQP",
      "rte": "RTE",
      "sst-2": "SST-2",
      "wnli": "WNLI",
      "qnli": "QNLI",
      "sts-b": "STS-B",
      "diagnostic": "AX"
  }

  task_name = FLAGS.task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  if FLAGS.do_submit and task_name == "wnli":
    submit_task_name = submit_name_mapping[task_name]
    with tf.io.gfile.GFile(os.path.join(submit_dir, "{}.tsv".format(
        submit_task_name)), "w") as fout:
      fout.write("index\tprediction\n")
      for i in range(146):
        fout.write("{}\t0\n".format(i))
    quit()

  # Task specific processor and label set
  processor = processors[task_name]()
  label_list = processor.get_labels()
  tf.logging.info("Label list for task %s: %s", FLAGS.task_name, label_list)

  # Model function
  model_fn = get_model_fn(len(label_list))

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
        predict_batch_size=FLAGS.predict_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

  ##### training
  if FLAGS.do_train:
    # Create train input function
    _, train_file = create_tfrecord(
        task_name, "train", processor, tokenizer)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        regression=task_name == "sts-b")
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    tf.io.gfile.makedirs(os.path.join(FLAGS.model_dir, "done"))

  ##### validation
  if FLAGS.do_eval:
    ##### Create eval tfrecords
    num_eval_examples, eval_file = create_tfrecord(
        task_name, FLAGS.eval_split, processor, tokenizer, pad_for_eval=True)

    ##### Validation input function
    assert num_eval_examples % FLAGS.eval_batch_size == 0
    eval_steps = int(num_eval_examples // FLAGS.eval_batch_size)
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True,
        regression=task_name == "sts-b")

    ###### Actual eval loop ######
    num_mins_waited = 0
    last_eval_step = FLAGS.eval_start_step
    eval_results = []
    pred_results = []
    while True:
      if last_eval_step >= FLAGS.eval_end_step: break

      # Evaluate all ckpts in the directory
      if FLAGS.eval_model_dir:
        eval_model_dir = FLAGS.eval_model_dir
      else:
        eval_model_dir = FLAGS.model_dir

      # Gather a list of checkpoints to evaluate
      steps_and_files = []
      try:
        filenames = sorted(tf.io.gfile.listdir(eval_model_dir))
      except tf.errors.NotFoundError:
        filenames = []
        tf.logging.info("`eval_model_dir` does not exist yet...")

      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          cur_filename = os.path.join(eval_model_dir, ckpt_name)
          global_step = int(cur_filename.split("-")[-1])
          if (global_step <= last_eval_step or
              global_step > FLAGS.eval_end_step):
            continue
          tf.logging.info("[{}] Add {} to eval list.".format(global_step,
                                                             cur_filename))
          steps_and_files.append([global_step, cur_filename])

      # Get empty list of checkpoints
      if not steps_and_files:
        # Training job is done: stop evaluation
        if tf.io.gfile.exists(os.path.join(eval_model_dir, "done")):
          break
        # Wait for 60 seconds
        else:
          time.sleep(60)
          num_mins_waited += 1.0
          tf.logging.info("Waited {:.1f} mins".format(num_mins_waited))
      else:
        num_mins_waited = 0

      # Evaluate / Predict / Submit the current list of checkpoints
      for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
        ##### Validation
        if FLAGS.do_eval:
          ret = estimator.evaluate(
              input_fn=eval_input_fn,
              steps=eval_steps,
              checkpoint_path=filename)

          ret["step"] = global_step
          ret["path"] = filename

          if task_name in ["cola", "mrpc", "qqp"]:
            tp, fp, tn, fn = (ret["eval_tp"], ret["eval_fp"], ret["eval_tn"],
                              ret["eval_fn"])
            ret["eval_f1"] = _compute_metric_based_on_keys(
                key="f1", tp=tp, fp=fp, tn=tn, fn=fn)
            ret["eval_corr"] = _compute_metric_based_on_keys(
                key="corr", tp=tp, fp=fp, tn=tn, fn=fn)

          eval_results.append(ret)

          # Log current result
          tf.logging.info("=" * 80)
          log_str = "Eval step {} | ".format(global_step)
          for key, val in eval_results[-1].items():
            log_str += "{} {} | ".format(key, val)
          tf.logging.info(log_str)
          tf.logging.info("=" * 80)

        # Update last eval step
        last_eval_step = max(last_eval_step, global_step)

    ##### Log the best validation result
    if FLAGS.do_eval:
      key_func = lambda x: x["eval_accuracy"]
      if task_name == "sts-b":
        key_func = lambda x: x["eval_pearsonr"]
      if task_name == "cola":
        key_func = lambda x: x["eval_corr"]
      if task_name in ["mrpc", "qqp"]:
        key_func = lambda x: x["eval_f1"] + x["eval_accuracy"]
      eval_results.sort(key=key_func, reverse=True)
      tf.logging.info("=" * 80)
      log_str = "Best eval result | "
      for key, val in eval_results[0].items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)

      # Save best eval result to model directory
      best_result = {}
      best_result["step"] = eval_results[0]["step"]
      best_result["path"] = eval_results[0]["path"]
      if task_name == "sts-b":
        best_result["pearsonr"] = float(eval_results[0]["eval_pearsonr"])
      else:
        best_result["accu"] = float(eval_results[0]["eval_accuracy"])
      if task_name in ["cola", "mrpc", "qqp"]:
        best_result["f1"] = float(eval_results[0]["eval_f1"])

        best_result["corr"] = float(eval_results[0]["eval_corr"])

      best_path = os.path.join(FLAGS.model_dir, "best_result.json")
      tf.logging.info("Dump eval results to {}".format(best_path))
      tf.logging.info(best_result)
      tf.logging.info("=" * 80)

      with tf.io.gfile.GFile(best_path, "w") as fp:
        json.dump(best_result, fp, indent=4)

  ##### Create submission
  if FLAGS.do_submit:
    ##### Retrieve the checkpoint with best validation performance
    if not FLAGS.do_eval:
      if FLAGS.eval_model_dir:
        eval_model_dir = FLAGS.eval_model_dir
      else:
        eval_model_dir = FLAGS.model_dir
      best_path = os.path.join(eval_model_dir, "best_result.json")
      if tf.io.gfile.exists(best_path):
        with tf.io.gfile.GFile(best_path, "r") as fp:
          best_result = json.load(fp)
        best_ckpt_path = best_result["path"]
      else:
        raise ValueError("Best result not found. Please do validation first.")
    else:
      best_ckpt_path = best_result["path"]

    ##### Create submission files for multiple tasks
    if task_name == "mnli_matched":
      task_names = ["mnli_matched", "diagnostic"]
    else:
      task_names = [task_name]
    for cur_task_name in task_names:
      tf.logging.info("********** Create submission **********")
      tf.logging.info("  - task name: %s", cur_task_name)
      tf.logging.info("  - best ckpt: %s", best_ckpt_path)

      #### Create test tfrecords and input function
      cur_processor = processors[cur_task_name]()
      _, test_file = create_tfrecord(
          cur_task_name, "test", cur_processor, tokenizer)
      submit_input_fn = file_based_input_fn_builder(
          input_file=test_file,
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=False,
          regression=cur_task_name == "sts-b")

      ##### Make prediction
      logits_list, predict_list = [], []
      for idx, result in enumerate(estimator.predict(
          input_fn=submit_input_fn,
          checkpoint_path=best_ckpt_path,
          yield_single_examples=True)):
        if idx % 1000 == 0:
          tf.logging.info("Predicting submission for example: {}".format(
              len(logits_list)))

        logits = [float(x) for x in result["logits"].flat]
        if len(logits) == 1:
          label_out = logits[0]
        elif len(logits) == 2:
          if logits[1] - logits[0] > 0.0:
            label_out = label_list[1]
          else:
            label_out = label_list[0]
        elif len(logits) == 3:
          max_index = np.argmax(np.array(logits, dtype=np.float32))
          label_out = label_list[max_index]
        else:
          raise NotImplementedError

        predict_list.append(label_out)
        logits_list.append(logits)

      ##### Perform normalization if needed
      if cur_task_name == "sts-b":
        max_pred = np.max(predict_list)
        predict_list = [pred / max_pred * 5.0 for pred in predict_list]

      ##### Write the predictions to the tsv file
      submit_task_name = submit_name_mapping[cur_task_name]
      submit_file_path = os.path.join(
          submit_dir, "{}.tsv".format(submit_task_name))
      with tf.io.gfile.GFile(submit_file_path, "w") as fout:
        fout.write("index\tprediction\n")
        for submit_cnt, pred in enumerate(predict_list):
          fout.write("{}\t{}\n".format(submit_cnt, pred))

      ##### Dump logits to the json file
      submit_json_path = os.path.join(
          submit_dir, "{}.logits.json".format(submit_task_name))
      with tf.io.gfile.GFile(submit_json_path, "w") as fp:
        json.dump(logits_list, fp, indent=4)

  ##### Clean all model ckpts
  if FLAGS.do_eval or FLAGS.do_predict:
    if not FLAGS.retain_all:
      retain_ckpt = set()

      # retain best
      if FLAGS.retain_best:
        if FLAGS.do_eval:
          retain_ckpt.add(eval_results[0]["path"])
        if FLAGS.do_predict and len(label_list) == 2:
          retain_ckpt.add(pred_results[0]["path"])

      for ret in eval_results:
        if ret["path"] in retain_ckpt:
          continue
        for suffix in [".index", ".meta", ".data-00000-of-00001"]:
          tf.io.gfile.remove(ret["path"] + suffix)

    ##### Remove all event files
    for event_file in tf.io.gfile.glob(os.path.join(FLAGS.model_dir,
                                                    "events.out.*")):
      tf.io.gfile.remove(event_file)

    ##### Remove all tfrecords files
    for record_file in tf.io.gfile.glob(os.path.join(FLAGS.model_dir,
                                                     "*.tfrecord")):
      tf.io.gfile.remove(record_file)


if __name__ == "__main__":
  tf.app.run()
