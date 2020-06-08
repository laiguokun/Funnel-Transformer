"""Perform pretraining."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow.compat.v1 as tf

import data_utils
import input_func_builder
import model_utils
import modeling
import optimization
import tokenization

logger = tf.get_logger()
logger.propagate = False
tf.disable_v2_behavior()


# TPU parameters
flags.DEFINE_string("master", default=None,
                    help="master")
flags.DEFINE_string("tpu", default=None,
                    help="The Cloud TPU to use for training. This should be "
                    "either the name used when creating the Cloud TPU, or a "
                    "grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("gcp_project", default=None,
                    help="Project name for the Cloud TPU-enabled project. If "
                    "not specified, we will attempt to automatically detect "
                    "the GCE project from metadata.")
flags.DEFINE_string("tpu_zone", default=None,
                    help="GCE zone where the Cloud TPU is located in. If not "
                    "specified, we will attempt to automatically detect the "
                    "GCE project from metadata.")
flags.DEFINE_bool("use_tpu", default=True,
                  help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
                     help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=16,
                     help="number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("record_dir", default=None,
                    help="Path to local directory containing tfrecord.")
flags.DEFINE_string("model_dir", default=None,
                    help="Estimator model_dir.")
flags.DEFINE_bool("overwrite", default=False,
                  help="Whether to overwrite exist model dir.")

# Init checkpoint
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model.")
flags.DEFINE_string("model_config", "",
                    help="FunnelTFM model configuration.")
flags.DEFINE_bool("init_global_vars", default=True,
                  help="If true, init all global vars. If false, init "
                  "trainable vars only.")

# Training config
flags.DEFINE_bool("do_train", default=True,
                  help="Whether to run training.")
flags.DEFINE_integer("train_batch_size", default=60,
                     help="Size of train batch.")
flags.DEFINE_integer("train_steps", default=100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
                     help="Number of steps for model checkpointing.")
flags.DEFINE_integer("max_save", default=100000,
                     help="Maximum number of checkpoints to save.")

# Evaluation config
flags.DEFINE_bool("do_eval", default=False,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path", default=None,
                    help="Checkpoint path for do_test evaluation."
                    "If set, model_dir will be ignored."
                    "If unset, will use the latest ckpt in `model_dir`.")
flags.DEFINE_integer("eval_batch_size", default=60,
                     help="Size of evalation batch.")
flags.DEFINE_integer("max_eval_batch", default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_integer("start_eval_steps", default=200000,
                     help="Which checkpoint to start with in `do_eval_only` "
                     "mode.")
flags.DEFINE_string("eval_split", default="dev",
                    help="Which data split to evaluate.")

##### Data config
flags.DEFINE_integer("num_passes", default=1,
                     help="Number of passed used for training.")
flags.DEFINE_integer("seq_len", default=0,
                     help="tgt len for objective; 0 for not using it")
flags.DEFINE_integer("num_predict", default=None,
                     help="Number of masked tokens.")

##### Loss related
flags.DEFINE_string("loss_type", default="mlm",
                    help="Which pretraining loss to use.")
# electra specific
flags.DEFINE_float("width_shrink", default=0.25,
                   help="Generator width / discriminator width.")
flags.DEFINE_float("disc_coeff", default=50,
                   help="Coefficient for discriminator loss.")

##### Precision
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")

##### Debug
flags.DEFINE_bool("verbose", default=False,
                  help="Whether to print additional information.")

FLAGS = flags.FLAGS


def metric_fn(loss):
  """Evaluation metric Fn which runs on CPU."""
  return {
      "eval/loss": tf.metrics.mean(loss),
  }


#### Wrap the loss function with the bf16 decorator
# Note: The bfloat_scope needs to be the `out most` scope. Otherwise, it
#       adds an unnecessary empty "" scope in between, leading to double
#       slashes (i.e., "...//...") in the scope name.
@model_utils.bf16_decorator
def mlm_loss_func(
    model, features, is_training, target=None, **kwargs):
  """Get mlm loss."""
  #### Unpack features
  seg_id = features["seg_id"]
  inputs = features["masked_input"]
  input_mask = features["input_mask"]
  target_mapping = features["target_mapping"]

  if target is None:
    target = features["target"]

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    total_loss, _ = model.get_mlm_loss(
        target,
        inputs,
        is_training,
        seg_id=seg_id,
        input_mask=input_mask,
        mapping=target_mapping,
        use_tpu=FLAGS.use_tpu,
        use_bfloat16=FLAGS.use_bfloat16,
        **kwargs)
    total_loss = tf.reduce_mean(total_loss)

  monitor_dict = {}
  monitor_dict["lm_loss"] = total_loss

  return total_loss, monitor_dict


def mlm(features, mode):
  """MLM pretraining."""
  #### Build Model
  if FLAGS.model_config:
    net_config = modeling.ModelConfig.init_from_json(FLAGS.model_config)
  else:
    net_config = modeling.ModelConfig.init_from_flags()
  net_config.to_json(os.path.join(FLAGS.model_dir, "net_config.json"))
  model = modeling.FunnelTFM(net_config)

  #### Training or Evaluation
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  total_loss, monitor_dict = mlm_loss_func(
      model, features, is_training)

  return total_loss, monitor_dict


@model_utils.bf16_decorator
def electra_loss_func(generator, model, features, is_training):
  """Get electra loss."""

  #### Unpack features
  masked_input = features["masked_input"]
  origin_input = features["origin_input"]
  pad_mask = features["pad_mask"]
  seg_id = features["seg_id"]
  target = features["target"]
  target_mapping = features["target_mapping"]
  is_target = features["is_target"]

  dtype = tf.float32 if not FLAGS.use_bfloat16 else tf.bfloat16

  #### Shared embedding layer
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    masked_embed, word_embed_table, _ = model.input_embedding(
        masked_input, is_training, dtype=dtype)

  #### Generator
  with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
    masked_hidden, masked_hiddens, _ = generator.encoder(
        masked_embed,
        is_training,
        seg_id=seg_id,
        input_mask=pad_mask)
    if generator.net_config.n_block > 1:
      masked_hidden, _ = generator.decoder(
          masked_hiddens,
          is_training,
          seg_id=seg_id,
          input_mask=pad_mask)
    gen_loss, gen_logits = generator.lm_loss(
        masked_hidden, target, lookup_table=word_embed_table,
        mapping=target_mapping, return_logits=True, use_tpu=FLAGS.use_tpu)
    avg_gen_loss = tf.reduce_mean(gen_loss)

  #### Sample from generator
  uniform = tf.random.uniform(minval=0, maxval=1, shape=gen_logits.shape,
                              dtype=tf.float32)
  gumbel = -tf.log(-tf.log(uniform + 1e-9) + 1e-9)
  samples = tf.argmax(gen_logits + gumbel, -1)

  # map `num_predict` samples to full length
  samples = tf.einsum("...m,...ml->...l",
                      tf.cast(samples, tf.float32),
                      tf.cast(target_mapping, tf.float32))
  samples = tf.cast(samples, tf.int32)

  sample_input = tf.where(is_target, samples, origin_input)
  binary_target = tf.equal(origin_input, sample_input)

  #### Discriminator
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    sample_embed, _, _ = model.input_embedding(
        sample_input, is_training, word_embed_table=word_embed_table,
        dtype=dtype)
    sample_hidden, sample_hiddens, _ = model.encoder(
        sample_embed,
        is_training,
        seg_id=seg_id,
        input_mask=pad_mask)
    if model.net_config.n_block > 1:
      sample_hidden, _ = model.decoder(
          sample_hiddens,
          is_training,
          seg_id=seg_id,
          input_mask=pad_mask)
    disc_loss, _ = model.binary_loss(
        sample_hidden,
        binary_target)
    non_pad_mask = tf.cast(1.0 - pad_mask, disc_loss.dtype)
    avg_disc_loss = (tf.reduce_sum(disc_loss * non_pad_mask) /
                     tf.reduce_sum(non_pad_mask))
    avg_disc_loss *= FLAGS.disc_coeff

  total_loss = avg_gen_loss + avg_disc_loss
  real_ratio = tf.reduce_mean(tf.cast(binary_target, tf.float32))
  monitor_dict = {
      "gen_loss": avg_gen_loss,
      "disc_loss": avg_disc_loss,
      "real_ratio": real_ratio,
  }

  return total_loss, monitor_dict


def get_generator_config(net_config):
  """Get generator net config."""
  args = {}
  for key in modeling.ModelConfig.keys:
    val = getattr(net_config, key)
    args[key] = val

  # shrink size of the network
  for key in ["d_model", "d_inner", "n_head"]:
    args[key] = int(args[key] * FLAGS.width_shrink)

  gen_config = modeling.ModelConfig(**args)

  return gen_config


def electra(features, mode):
  """ELECTRA pretraining."""
  #### Build Model
  if FLAGS.model_config:
    net_config = modeling.ModelConfig.init_from_json(FLAGS.model_config)
  else:
    net_config = modeling.ModelConfig.init_from_flags()
  model = modeling.FunnelTFM(net_config)
  net_config.to_json(os.path.join(FLAGS.model_dir, "net_config.json"))

  gen_config = get_generator_config(net_config)
  generator = modeling.FunnelTFM(gen_config)

  #### Training or Evaluation
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  #### Get loss
  total_loss, monitor_dict = electra_loss_func(
      generator, model, features, is_training)

  return total_loss, monitor_dict


def get_model_fn():
  """Create model function for TPU estimator."""
  def model_fn(features, labels, mode, params):
    """Model computational graph."""
    del labels
    del params

    total_loss, monitor_dict = eval(FLAGS.loss_type)(features, mode)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: %d", num_params)

    if FLAGS.verbose:
      format_str = "{{:<{0}s}}\t{{}}".format(
          max([len(v.name) for v in tf.trainable_variables()]))
      for v in tf.trainable_variables():
        tf.logging.info(format_str.format(v.name, v.get_shape()))

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      #### Reduce sum losses from all TPU cores
      with tf.colocate_with(total_loss):
        total_loss = tf.tpu.cross_replica_sum(total_loss)
        total_loss = total_loss / FLAGS.num_hosts / FLAGS.num_core_per_host
      metric_loss = tf.reshape(total_loss, [1])

      #### Constructing evaluation TPUEstimatorSpec with new cache.
      eval_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [metric_loss]))

      return eval_spec

    #### Get the train op
    train_op, optim_dict = optimization.get_train_op(total_loss)
    monitor_dict.update(optim_dict)

    #### Customized initial checkpoint
    scaffold_fn = model_utils.custom_initialization(FLAGS.init_global_vars)

    #### Creating host calls
    host_call = model_utils.construct_scalar_host_call(
        monitor_dict=monitor_dict,
        model_dir=FLAGS.model_dir,
        prefix="train/",
        reduce_fn=tf.reduce_mean)

    #### Constructing training TPUEstimatorSpec with new cache.
    train_spec = tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
        scaffold_fn=scaffold_fn)

    return train_spec

  return model_fn


def get_input_fn(split):
  """Get input function for TPU Estimator."""
  batch_size = (FLAGS.train_batch_size if split == "train" else
                FLAGS.eval_batch_size)

  if FLAGS.model_config:
    net_config = modeling.ModelConfig.init_from_json(FLAGS.model_config)
  else:
    net_config = modeling.ModelConfig.init_from_flags()

  kwargs = dict(
      tfrecord_dir=FLAGS.record_dir,
      split=split,
      bsz_per_host=batch_size // FLAGS.num_hosts,
      seq_len=FLAGS.seq_len,
      num_predict=FLAGS.num_predict,
      num_hosts=FLAGS.num_hosts,
      num_core_per_host=FLAGS.num_core_per_host,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16,
      num_pool=net_config.n_block - 1,
      truncate_seq=FLAGS.truncate_seq and net_config.separate_cls,
  )

  input_fn, record_info_dict = input_func_builder.get_input_fn(**kwargs)

  return input_fn, record_info_dict


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)
  if not tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.makedirs(FLAGS.model_dir)

  #### Validate FLAGS
  if FLAGS.save_steps == 0:
    FLAGS.save_steps = None

  assert FLAGS.seq_len > 0

  #### Tokenizer
  tokenizer = tokenization.get_tokenizer()
  data_utils.setup_special_ids(tokenizer)

  if FLAGS.do_train:
    # Get train input function
    train_input_fn, train_record_info_dict = get_input_fn("train")

    tf.logging.info("num of batches %d", train_record_info_dict["num_batch"])

  if FLAGS.do_eval:
    assert FLAGS.num_hosts == 1
    # Get eval input function
    eval_input_fn, eval_record_info_dict = get_input_fn(FLAGS.eval_split)

    if FLAGS.max_eval_batch > 0:
      num_eval_batch = FLAGS.max_eval_batch
    else:
      num_eval_batch = 1e30

    num_eval_batch = eval_record_info_dict["num_batch"]
    tf.logging.info("num of eval batches %d", num_eval_batch)

  ##### Get model function
  model_fn = get_model_fn()

  ##### Create TPUEstimator
  # TPU Configuration
  run_config = model_utils.get_run_config()

  # Custom TPU Estimator
  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params={},
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_on_tpu=FLAGS.use_tpu)

  #### Training
  if FLAGS.do_train:
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  #### Evaluation
  if FLAGS.do_eval:
    if FLAGS.eval_ckpt_path is not None:
      if FLAGS.eval_ckpt_path.endswith("latest"):
        ckpt_dir = os.path.dirname(FLAGS.eval_ckpt_path)
        FLAGS.eval_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

      ret = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch,
                               checkpoint_path=FLAGS.eval_ckpt_path)
      tf.logging.info("=" * 200)
      log_str = "Eval results | "
      for key, val in ret.items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)
    else:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
      eval_results = []
      for eval_checkpoint in ckpt_state.all_model_checkpoint_paths:
        if not tf.io.gfile.exists(eval_checkpoint + ".index"):
          continue
        global_step = int(eval_checkpoint.split("-")[-1])
        if (global_step < FLAGS.start_eval_steps or global_step >
            FLAGS.train_steps):
          continue
        tf.logging.info("Evaluate ckpt %d", global_step)
        ret = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch,
                                 checkpoint_path=eval_checkpoint)
        eval_results.append(ret)

      eval_results.sort(key=lambda x: x["loss"])

      tf.logging.info("=" * 200)
      log_str = "Best results | "
      for key, val in eval_results[0].items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)


if __name__ == "__main__":
  app.run(main)
