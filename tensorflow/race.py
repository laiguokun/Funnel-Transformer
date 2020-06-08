"""Finetune for RACE."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import random
import time

from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow.compat.v1 as tf

import classifier_utils
import data_utils
import model_utils
import optimization
import tokenization
import modeling


##### I/O related
# Basic paths
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
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

# training
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
flags.DEFINE_string("eval_model_dir", default="", help="Model dir to evaluate")
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_start_step", default=-1,
                     help="Start step for evaluation")
flags.DEFINE_integer("eval_end_step", default=10000000,
                     help="End step for evaluation")
flags.DEFINE_integer("eval_batch_size", default=128,
                     help="batch size for evaluation")

# task specific
flags.DEFINE_integer("max_seq_length", default=512, help="Max sequence length")
flags.DEFINE_integer("max_qa_length", default=128, help="Max QA length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")

# debug
flags.DEFINE_bool("verbose", default=False,
                  help="Whether to print additional information.")

FLAGS = flags.FLAGS


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True,
               is_high_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
    self.is_high_example = is_high_example


def convert_single_example(example, tokenize_fn):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, classifier_utils.PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * FLAGS.max_seq_length * 4,
        input_mask=[1] * FLAGS.max_seq_length * 4,
        segment_ids=[0] * FLAGS.max_seq_length * 4,
        label_id=0,
        is_real_example=False,
        is_high_example=False)

  input_ids, input_mask, segment_ids = [], [], []

  tokens_context = tokenize_fn(example.context)
  for i in range(len(example.qa_list)):
    tokens_qa = tokenize_fn(example.qa_list[i])
    if len(tokens_qa) > FLAGS.max_qa_length:
      tokens_qa = tokens_qa[- FLAGS.max_qa_length:]

    if len(tokens_context) + len(tokens_qa) > FLAGS.max_seq_length - 3:
      tokens_p = tokens_context[: FLAGS.max_seq_length - 3 - len(tokens_qa)]
    else:
      tokens_p = tokens_context

    # [CLS QA SEP P SEP]
    cur_inp_ids = ([FLAGS.cls_id] +
                   tokens_qa + [FLAGS.sep_id] +
                   tokens_p + [FLAGS.sep_id])
    cur_seg_ids = ([FLAGS.seg_id_cls] +
                   [FLAGS.seg_id_a] * (len(tokens_qa) + 1) +
                   [FLAGS.seg_id_b] * (len(tokens_p) + 1))
    cur_inp_mask = [0] * len(cur_inp_ids)

    if len(cur_inp_ids) < FLAGS.max_seq_length:
      delta_len = FLAGS.max_seq_length - len(cur_inp_ids)
      cur_inp_ids = cur_inp_ids + [0] * delta_len
      cur_inp_mask = cur_inp_mask + [1] * delta_len
      cur_seg_ids = cur_seg_ids + [FLAGS.seg_id_pad] * delta_len

    assert len(cur_inp_ids) == FLAGS.max_seq_length
    assert len(cur_inp_mask) == FLAGS.max_seq_length
    assert len(cur_seg_ids) == FLAGS.max_seq_length

    input_ids.extend(cur_inp_ids)
    input_mask.extend(cur_inp_mask)
    segment_ids.extend(cur_seg_ids)

  label_id = example.label
  level = example.level

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_high_example=level == "high")

  return feature


class InputExample(object):

  def __init__(self, context, qa_list, label, level):
    self.context = context
    self.qa_list = qa_list
    self.label = label
    self.level = level


def get_examples(data_dir, set_type):
  """Get examples from raw data."""
  examples = []

  for level in ["middle", "high"]:
    cur_dir = os.path.join(data_dir, set_type, level)
    for filename in tf.io.gfile.listdir(cur_dir):
      cur_path = os.path.join(cur_dir, filename)
      with tf.io.gfile.GFile(cur_path) as f:
        cur_data = json.load(f)

        answers = cur_data["answers"]
        options = cur_data["options"]
        questions = cur_data["questions"]
        context = cur_data["article"]

        for i in range(len(answers)):
          label = ord(answers[i]) - ord("A")
          qa_list = []

          question = questions[i]
          for j in range(4):
            option = options[i][j]

            if "_" in question:
              qa_cat = question.replace("_", option)
            else:
              qa_cat = " ".join([question, option])

            qa_list.append(qa_cat)

          examples.append(InputExample(context, qa_list, label, level))

          if len(examples) % 10000 == 0:
            tf.logging.info("Reading example %d", len(examples))

  return examples


def file_based_convert_examples_to_features(examples, tokenize_fn, output_file):
  """Convert examples to tfrecords."""
  if tf.io.gfile.exists(output_file) and not FLAGS.overwrite_data:
    return

  writer = tf.python_io.TFRecordWriter(output_file)

  for ex_index, example in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d", ex_index, len(examples))

    feature = convert_single_example(example, tokenize_fn)

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
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    features["is_high_example"] = create_int_feature(
        [int(feature.is_high_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * 4], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * 4], tf.float32),
      "segment_ids": tf.FixedLenFeature([seq_length * 4], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
      "is_high_example": tf.FixedLenFeature([], tf.int64),
  }

  tf.logging.info("Input tfrecord file %s", input_file)

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

    @model_utils.bf16_decorator
    def race_loss_func(features, model):
      """Get race loss."""
      #### Get loss from inputs
      inputs = features["input_ids"]
      seg_id = features["segment_ids"]
      input_mask = features["input_mask"]
      labels = features["label_ids"]

      with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        per_example_loss, logits = model.get_race_loss(
            labels, inputs, is_training, seg_id=seg_id, input_mask=input_mask,
            use_tpu=FLAGS.use_tpu, use_bfloat16=FLAGS.use_bfloat16)

      return per_example_loss, logits

    per_example_loss, logits = race_loss_func(features, model)
    total_loss = tf.reduce_mean(per_example_loss)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: %d", num_params)
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

      def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                    is_high_example):
        """Metric function used for evaluation."""
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        eval_input_dict = {
            "labels": label_ids,
            "predictions": predictions,
            "weights": is_real_example
        }
        accuracy = tf.metrics.accuracy(**eval_input_dict)

        high_eval_input_dict = {
            "labels": label_ids,
            "predictions": predictions,
            "weights": is_real_example * is_high_example
        }
        accuracy_high = tf.metrics.accuracy(**high_eval_input_dict)

        mid_eval_input_dict = {
            "labels": label_ids,
            "predictions": predictions,
            "weights": is_real_example * (1.0 - is_high_example)
        }
        accuracy_mid = tf.metrics.accuracy(**mid_eval_input_dict)

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_accuracy_high": accuracy_high,
            "eval_accuracy_mid": accuracy_mid,
            "eval_loss": loss}

      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
      is_high_example = tf.cast(features["is_high_example"], dtype=tf.float32)

      #### Constructing evaluation TPUEstimatorSpec with new cache.
      label_ids = tf.reshape(features["label_ids"], [-1])
      metric_args = [per_example_loss, label_ids, logits, is_real_example,
                     is_high_example]

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

    #### Get train op
    train_op, _ = optimization.get_train_op(total_loss)

    #### Constructing training TPUEstimatorSpec
    if FLAGS.use_tpu:
      #### Creating host calls
      host_call = None

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


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate flags
  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

  #### Load tokenizer
  tokenizer = tokenization.get_tokenizer()
  data_utils.setup_special_ids(tokenizer)

  tokenize_fn = tokenizer.convert_text_to_ids
  tok_identifier = _get_tok_identifier()

  #### Overwrite existing `eval_dir`
  if FLAGS.do_eval and FLAGS.overwrite:
    eval_dir = os.path.join(FLAGS.model_dir, "eval")
    if tf.io.gfile.exists(eval_dir):
      tf.io.gfile.rmtree(eval_dir)

  if (not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict
      and not FLAGS.do_submit):
    raise ValueError(
        "At least one of `do_train`, `do_eval, `do_predict` or "
        "`do_submit` must be True.")

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

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
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

  if FLAGS.do_train:
    train_file_base = "{}.len-{}.train.tfrecord".format(
        tok_identifier, FLAGS.max_seq_length)
    train_file = os.path.join(FLAGS.output_dir, train_file_base)

    if not tf.io.gfile.exists(train_file) or FLAGS.overwrite_data:
      train_examples = get_examples(FLAGS.data_dir, "train")
      tf.logging.info("Num of train samples: %d", len(train_examples))
      random.shuffle(train_examples)
      file_based_convert_examples_to_features(
          train_examples, tokenize_fn, train_file)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    tf.io.gfile.makedirs(os.path.join(FLAGS.model_dir, "done"))

  if FLAGS.do_eval:
    ##### Load raw eval examples: we always do this to get the example number
    eval_examples = get_examples(FLAGS.data_dir, FLAGS.eval_split)
    tf.logging.info("Num of eval samples: %d", len(eval_examples))

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
    #
    # Modified in XL: We also adopt the same mechanism for GPUs.

    while len(eval_examples) % FLAGS.eval_batch_size != 0:
      eval_examples.append(classifier_utils.PaddingInputExample())

    assert len(eval_examples) % FLAGS.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    ##### Convert raw example into tfrecords (if needed)
    eval_file_base = "{}.len-{}.{}.tfrecord".format(
        tok_identifier, FLAGS.max_seq_length, FLAGS.eval_split)
    eval_file = os.path.join(FLAGS.output_dir, eval_file_base)
    if not tf.io.gfile.exists(eval_file) or FLAGS.overwrite_data:
      file_based_convert_examples_to_features(
          eval_examples, tokenize_fn, eval_file)

    ##### Construct eval input function
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True)

    ##### Evaluation loop
    num_mins_waited = 0
    last_eval_step = FLAGS.eval_start_step
    eval_results = []
    while True:
      if last_eval_step >= FLAGS.eval_end_step: break

      if FLAGS.eval_model_dir is not None and FLAGS.eval_model_dir:
        eval_model_dir = FLAGS.eval_model_dir
      else:
        eval_model_dir = FLAGS.model_dir

      # Gather a list of checkpoints to evaluate
      steps_and_files = []
      try:
        filenames = sorted(tf.io.gfile.listdir(eval_model_dir))
        # tf.logging.info("Filenames in eval_model_dir: {}".format(filenames))
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
          tf.logging.info("Add %s to eval list.", cur_filename)
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

      # Evaluate / Predict the current list of checkpoints
      for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
        # Evaluate
        ret = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=eval_steps,
            checkpoint_path=filename)

        ret["step"] = global_step
        ret["path"] = filename

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

    # Log the best result
    key_func = lambda x: x["eval_accuracy"]
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
    best_result["accu"] = float(eval_results[0]["eval_accuracy"])

    save_path = os.path.join(FLAGS.model_dir, "best_result.json")
    tf.logging.info("Dump eval results to %s", save_path)
    tf.logging.info(best_result)
    tf.logging.info("=" * 80)

    with tf.io.gfile.GFile(save_path, "w") as fp:
      json.dump(best_result, fp, indent=4)

    # Clean all model ckpts
    if not FLAGS.retain_all:
      for idx, ret in enumerate(eval_results):
        if FLAGS.retain_best and idx == 0:
          continue
        for suffix in [".index", ".meta", ".data-00000-of-00001"]:
          tf.io.gfile.remove(ret["path"] + suffix)

    # Remove all event files
    for event_file in tf.io.gfile.glob(os.path.join(FLAGS.model_dir,
                                                    "events.out.*")):
      tf.io.gfile.remove(event_file)


if __name__ == "__main__":
  tf.app.run()
