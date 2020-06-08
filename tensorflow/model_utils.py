"""Common model utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import re

from absl import flags
import absl.logging as _logging

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2  # used for summaries only.


flags.DEFINE_integer("log_freq", default=100, help="log frequence.")
FLAGS = flags.FLAGS


def construct_scalar_host_call(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None):
  """Construct host call for scalar."""

  # Only consider scalar
  metric_names = []
  for k, v in sorted(monitor_dict.items(), key=lambda x: x[0]):
    if v.shape.ndims == 0:
      metric_names.append(k)
      tf.logging.info("Host call receives %s: %s", k, v.shape)
      monitor_dict[k] = tf.reshape(v, [1])
    else:
      tf.logging.info("Host call ignores %s: %s", k, v.shape)

  def host_call_fn(global_step, *args):
    """Actual host call function."""
    step = global_step[0]
    with tf2.summary.create_file_writer(
        model_dir, filename_suffix=".host_call").as_default():
      with tf2.summary.record_if(lambda: tf.equal(step % FLAGS.log_freq, 0)):
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          tf2.summary.scalar(prefix + name, scalar, step=step)

        return tf.summary.all_v2_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [monitor_dict[key] for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def get_run_config():
  """Get TPU run config."""
  if FLAGS.use_tpu and FLAGS.tpu:
    tf.logging.info("TPU name: %s", FLAGS.tpu)
    tpu_cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster = None

  per_host_input = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations,
          num_shards=FLAGS.num_core_per_host * FLAGS.num_hosts,
          per_host_input_for_training=per_host_input,
          tpu_job_name=FLAGS.tpu_job_name),
      keep_checkpoint_max=FLAGS.max_save,
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_steps,
      train_distribute=(None if FLAGS.use_tpu or FLAGS.num_core_per_host == 1
                        else tf.distribute.MirroredStrategy())
  )

  return run_config


def custom_initialization(init_global_vars=False, map_func=None):
  """Config customized initialization."""
  if init_global_vars:
    tvars = tf.global_variables()
  else:
    tvars = tf.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  if FLAGS.init_checkpoint:
    if FLAGS.init_checkpoint.endswith("latest"):
      ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
      init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    else:
      init_checkpoint = FLAGS.init_checkpoint

    tf.logging.info("Initialize from the ckpt %s", init_checkpoint)

    (assignment_map, initialized_variable_names
    ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint, map_func)
    if FLAGS.use_tpu:
      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Log customized initialization
    tf.logging.info("**** Global Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

  return scaffold_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, map_func=None):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  # Dict(ckpt_var_name = curr_graph_var)
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)

    name_to_variable[name] = var

  assignment_map = collections.OrderedDict()
  init_vars = tf.train.list_variables(init_checkpoint)
  for x in init_vars:
    (src_name, var) = (x[0], x[1])

    # map the `src_name` in the ckpt to the `tgt_name` in the current graph
    if map_func is not None:
      tgt_name = map_func(src_name)
    else:
      tgt_name = src_name

    if tgt_name not in name_to_variable:
      if "adam" not in src_name:
        tf.logging.warning("Ignore variable %s in ckpt.", src_name)
      continue
    assignment_map[src_name] = name_to_variable[tgt_name]

    # record tgt_name is initialized
    initialized_variable_names[tgt_name] = 1
    initialized_variable_names[tgt_name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def bf16_decorator(func):
  """A wrapper function for bfloat16 scope."""
  @functools.wraps(func)
  def wrapped_func(*args, **kwargs):
    if FLAGS.use_bfloat16:
      with tf.tpu.bfloat16_scope():
        return func(*args, **kwargs)
    else:
      with tf.variable_scope(""):
        return func(*args, **kwargs)

  return wrapped_func

