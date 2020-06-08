"""Modify ckpt to desired format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("src_ckpt", "",
                    "Path to the checkpoint.")
flags.DEFINE_string("tgt_ckpt", "",
                    "Path to the checkpoint.")
flags.DEFINE_bool("finetune_only", default=True,
                  help="Only retain params/states related to finetuning.")


def get_tensor(reader, src_name):
  """Get the correctly mapped tensor."""
  if "transformer_xl" in src_name:
    tgt_name = src_name.replace("transformer_xl", "encoder")
    tf.logging.info("encoder: %s -> %s", src_name, tgt_name)
  elif "model/layer" in src_name:
    tgt_name = src_name.replace("model/layer", "model/decoder/layer")
    tf.logging.info("decoder: %s -> %s", src_name, tgt_name)
  elif "generator/layer" in src_name:
    tgt_name = src_name.replace("generator/layer", "generator/decoder/layer")
    tf.logging.info("decoder: %s -> %s", src_name, tgt_name)
  else:
    tgt_name = src_name
  tensor = reader.get_tensor(src_name)

  return tensor, tgt_name


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  reader = tf.train.load_checkpoint(FLAGS.src_ckpt)
  var_values, var_dtypes = {}, {}
  for (name, _) in tf.train.list_variables(FLAGS.src_ckpt):
    # skip global_step and optimizer states in src ckpt if not FLAGS.retain_all
    if FLAGS.finetune_only and (name.startswith("global_step") or
                                "adam" in name.lower() or
                                "generator" in name.lower()):
      continue

    tensor, tgt_name = get_tensor(reader, name)
    var_values[tgt_name] = tensor
    var_dtypes[tgt_name] = tensor.dtype

  if FLAGS.tgt_ckpt:
    if not tf.io.gfile.exists(os.path.dirname(FLAGS.tgt_ckpt)):
      tf.io.gfile.makedirs(os.path.dirname(FLAGS.tgt_ckpt))

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      tf_vars = [
          tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
          for v in var_values
      ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    group_assign_op = tf.group(assign_ops)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      # Run grouped assign op
      feed_dict = {}
      for p, (name, value) in zip(placeholders, var_values.items()):
        feed_dict[p] = value
      sess.run(group_assign_op, feed_dict)

      # Use the built saver to save the averaged checkpoint.
      saver.save(sess, FLAGS.tgt_ckpt)

    tf.logging.info("Modified checkpoint saved to %s", FLAGS.tgt_ckpt)

if __name__ == "__main__":
  app.run(main)
  tf.logging.set_verbosity(tf.logging.INFO)
