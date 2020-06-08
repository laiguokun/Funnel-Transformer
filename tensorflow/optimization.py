"""Optimization related functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import flags
import absl.logging as _logging

import tensorflow.compat.v1 as tf

import tpu_optimizer


##### Optimization related flags #####
# learning rate schedule
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")

# weight decay
flags.DEFINE_float("weight_decay", default=0.00, help="weight decay rate")

# gradient clip
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_bool("per_core_clip", True,
                  help="Perform gradient clip on each TPU core.")
flags.DEFINE_bool("skip_nan_grad", False,
                  help="Whether to use skip NaN or Inf gradient.")

# used during finetune
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")

# adam specific hparams
flags.DEFINE_float("adam_beta1", default=0.9,
                   help="The exponential decay rate for the 1st moment.")
flags.DEFINE_float("adam_beta2", default=0.99,
                   help="The exponential decay rate for the 2nd moment.")
flags.DEFINE_bool("adam_correction", default=True,
                  help="Use the adam bias correction.")
flags.DEFINE_bool("use_wd_exclusion", default=False,
                  help="Exclude certain params from weight decay as in BERT.")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="adam epsilon")


FLAGS = flags.FLAGS


def _get_variable_name(param_name):
  """Get the variable name from the tensor name."""
  m = re.match("^(.*):\\d+$", param_name)
  if m is not None:
    param_name = m.group(1)
    return param_name


def compute_gradients(total_loss):
  """Separate the function of gradient computation."""
  monitor_dict = {}

  ##### Configure optimizer
  global_step = tf.train.get_or_create_global_step()

  # Warmup the learning rate linearly
  if FLAGS.warmup_steps > 0:
    progress = (tf.cast(global_step, tf.float32) /
                tf.cast(FLAGS.warmup_steps, tf.float32))
  else:
    progress = 1.0
  curr_ratio = progress + (1.0 - progress) * FLAGS.min_lr_ratio
  warmup_lr = curr_ratio * FLAGS.learning_rate

  # Decay the learning rate
  if FLAGS.decay_method == "poly":
    decay_lr = tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)
  elif FLAGS.decay_method == "cos":
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)
  else:
    raise ValueError(FLAGS.decay_method)

  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  if (FLAGS.weight_decay > 0 and not FLAGS.use_tpu and
      FLAGS.num_core_per_host > 1):
    raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                     "training so far.")

  if FLAGS.use_wd_exclusion:
    exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"]
  else:
    exclude_from_weight_decay = []

  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      beta_1=FLAGS.adam_beta1,
      beta_2=FLAGS.adam_beta2,
      epsilon=FLAGS.adam_epsilon,
      bias_correction=FLAGS.adam_correction,
      exclude_from_weight_decay=exclude_from_weight_decay,
      weight_decay_rate=FLAGS.weight_decay)

  if FLAGS.use_tpu:
    if FLAGS.per_core_clip:
      optimizer = tpu_optimizer.CrossShardOptimizer(
          optimizer, skip_nan_grad=FLAGS.skip_nan_grad)
    else:
      optimizer = tpu_optimizer.CrossShardOptimizer(
          optimizer, skip_nan_grad=FLAGS.skip_nan_grad, clip=FLAGS.clip)

  ##### Compute gradient
  variables = tf.trainable_variables()
  gradients = tf.gradients(total_loss, variables)

  if FLAGS.clip > 0 and FLAGS.per_core_clip:
    tf.logging.info("Clip local gradient with norm %.3f.", FLAGS.clip)
    clipped, local_gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)
  else:
    tf.logging.info("Do not clip local gradient.")
    clipped = list(gradients)
    local_gnorm = tf.linalg.global_norm(gradients)

  # layer-wise learning rate decay
  if FLAGS.lr_layer_decay_rate != 1.0:
    def _get_layer_id(name):
      if "model/input" in name:
        return 0
      m = re.search(r"model/(encoder|decoder)/layer_(\d+?)/", name)
      if not m: return None
      return int(m.group(2)) + 1

    n_layer = 0
    for i in range(len(clipped)):
      layer_id = _get_layer_id(variables[i].name)
      if layer_id is None: continue
      n_layer = max(n_layer, layer_id + 1)

    for i in range(len(clipped)):
      layer_id = _get_layer_id(variables[i].name)
      if layer_id is not None:
        abs_rate = FLAGS.lr_layer_decay_rate ** (n_layer - 1 - layer_id)
        tf.logging.info("Apply mult %.4f to the grad of %s",
                        abs_rate, variables[i].name)
        if isinstance(clipped[i], tf.IndexedSlices):
          clipped[i] = tf.IndexedSlices(clipped[i].values * abs_rate,
                                        clipped[i].indices,
                                        clipped[i].dense_shape)
        else:
          clipped[i] *= abs_rate
      else:
        tf.logging.info("Grad of %s is not decayed.", variables[i].name)

  grad_and_vars = list(zip(clipped, variables))

  monitor_dict["local_gnorm"] = local_gnorm
  monitor_dict["learning_rate"] = learning_rate

  return optimizer, grad_and_vars, global_step, monitor_dict


def get_train_op(total_loss):
  """Get the train op from training loss."""
  ##### Compute gradients
  optimizer, grad_and_vars, global_step, monitor_dict = compute_gradients(
      total_loss)

  ##### Construct train op
  train_op = optimizer.apply_gradients(
      grad_and_vars, global_step=global_step)

  # Manually increment `global_step` for AdamW and LAMB
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  return train_op, monitor_dict


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               bias_correction=False,
               exclude_from_weight_decay=None,
               include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.bias_correction = bias_correction
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.include_in_weight_decay = include_in_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []

    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        m = tf.get_variable(
            name=param_name + "/adam_m",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())
        v = tf.get_variable(
            name=param_name + "/adam_v",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn"t interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name) and self.weight_decay_rate > 0:
        update += self.weight_decay_rate * param

      # Adam bias correction
      if self.bias_correction:
        global_step_float = tf.cast(global_step, update.dtype)
        bias_correction1 = 1.0 - self.beta_1 ** (global_step_float + 1)
        bias_correction2 = 1.0 - self.beta_2 ** (global_step_float + 1)
        learning_rate = (self.learning_rate * tf.sqrt(bias_correction2)
                         / bias_correction1)
      else:
        learning_rate = self.learning_rate

      update_with_lr = learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False

    for r in self.include_in_weight_decay:
      if re.search(r, param_name) is not None:
        tf.logging.info("Include %s in weight decay", param_name)
        return True

    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          tf.logging.info("Adam WD excludes %s", param_name)
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


