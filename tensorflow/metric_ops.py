# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains metric-computing operations on streamed tensors.

Module documentation, including "@@" callouts, should be put in
third_party/tensorflow/contrib/metrics/__init__.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops

# Epsilon constant used to represent extremely small quantity.
_EPSILON = 1e-7


def streaming_covariance(predictions,
                         labels,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  """Computes the unbiased sample covariance between `predictions` and `labels`.

  The `streaming_covariance` function creates four local variables,
  `comoment`, `mean_prediction`, `mean_label`, and `count`, which are used to
  compute the sample covariance between predictions and labels across multiple
  batches of data. The covariance is ultimately returned as an idempotent
  operation that simply divides `comoment` by `count` - 1. We use `count` - 1
  in order to get an unbiased estimate.

  The algorithm used for this online computation is described in
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
  Specifically, the formula used to combine two sample comoments is
  `C_AB = C_A + C_B + (E[x_A] - E[x_B]) * (E[y_A] - E[y_B]) * n_A * n_B / n_AB`
  The comoment for a single batch of data is simply
  `sum((x - E[x]) * (y - E[y]))`, optionally weighted.

  If `weights` is not None, then it is used to compute weighted comoments,
  means, and count. NOTE: these weights are treated as "frequency weights", as
  opposed to "reliability weights". See discussion of the difference on
  https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

  To facilitate the computation of covariance across multiple batches of data,
  the function creates an `update_op` operation, which updates underlying
  variables and returns the updated covariance.

  Args:
    predictions: A `Tensor` of arbitrary size.
    labels: A `Tensor` of the same size as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or the
      same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric value
      variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    covariance: A `Tensor` representing the current unbiased sample covariance,
      `comoment` / (`count` - 1).
    update_op: An operation that updates the local variables appropriately.

  Raises:
    ValueError: If labels and predictions are of different sizes or if either
      `metrics_collections` or `updates_collections` are not a list or tuple.
  """
  with variable_scope.variable_scope(name, 'covariance',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    count_ = metrics_impl.metric_variable([], dtypes.float32, name='count')
    mean_prediction = metrics_impl.metric_variable([],
                                                   dtypes.float32,
                                                   name='mean_prediction')
    mean_label = metrics_impl.metric_variable([],
                                              dtypes.float32,
                                              name='mean_label')
    comoment = metrics_impl.metric_variable(  # C_A in update equation
        [], dtypes.float32, name='comoment')

    if weights is None:
      batch_count = math_ops.cast(array_ops.size(labels),
                                  dtypes.float32)  # n_B in eqn
      weighted_predictions = predictions
      weighted_labels = labels
    else:
      weights = weights_broadcast_ops.broadcast_weights(weights, labels)
      batch_count = math_ops.reduce_sum(weights)  # n_B in eqn
      weighted_predictions = math_ops.multiply(predictions, weights)
      weighted_labels = math_ops.multiply(labels, weights)

    update_count = state_ops.assign_add(count_, batch_count)  # n_AB in eqn
    prev_count = update_count - batch_count  # n_A in update equation

    # We update the means by Delta=Error*BatchCount/(BatchCount+PrevCount)
    # batch_mean_prediction is E[x_B] in the update equation
    batch_mean_prediction = math_ops.div_no_nan(
        math_ops.reduce_sum(weighted_predictions), batch_count)
    delta_mean_prediction = math_ops.div_no_nan(
        (batch_mean_prediction - mean_prediction) * batch_count, update_count)
    update_mean_prediction = state_ops.assign_add(mean_prediction,
                                                  delta_mean_prediction)
    # prev_mean_prediction is E[x_A] in the update equation
    prev_mean_prediction = update_mean_prediction - delta_mean_prediction

    # batch_mean_label is E[y_B] in the update equation
    batch_mean_label = math_ops.div_no_nan(
        math_ops.reduce_sum(weighted_labels), batch_count)
    delta_mean_label = math_ops.div_no_nan(
        (batch_mean_label - mean_label) * batch_count, update_count)
    update_mean_label = state_ops.assign_add(mean_label, delta_mean_label)
    # prev_mean_label is E[y_A] in the update equation
    prev_mean_label = update_mean_label - delta_mean_label

    unweighted_batch_coresiduals = ((predictions - batch_mean_prediction) *
                                    (labels - batch_mean_label))
    # batch_comoment is C_B in the update equation
    if weights is None:
      batch_comoment = math_ops.reduce_sum(unweighted_batch_coresiduals)
    else:
      batch_comoment = math_ops.reduce_sum(unweighted_batch_coresiduals *
                                           weights)

    # View delta_comoment as = C_AB - C_A in the update equation above.
    # Since C_A is stored in a var, by how much do we need to increment that var
    # to make the var = C_AB?
    delta_comoment = (
        batch_comoment + (prev_mean_prediction - batch_mean_prediction) *
        (prev_mean_label - batch_mean_label) *
        (prev_count * batch_count / update_count))
    update_comoment = state_ops.assign_add(comoment, delta_comoment)

    covariance = array_ops.where(
        math_ops.less_equal(count_, 1.),
        float('nan'),
        math_ops.truediv(comoment, count_ - 1),
        name='covariance')
    with ops.control_dependencies([update_comoment]):
      update_op = array_ops.where(
          math_ops.less_equal(count_, 1.),
          float('nan'),
          math_ops.truediv(comoment, count_ - 1),
          name='update_op')

  if metrics_collections:
    ops.add_to_collections(metrics_collections, covariance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return covariance, update_op


def streaming_pearson_correlation(predictions,
                                  labels,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes Pearson correlation coefficient between `predictions`, `labels`.

  The `streaming_pearson_correlation` function delegates to
  `streaming_covariance` the tracking of three [co]variances:

  - `streaming_covariance(predictions, labels)`, i.e. covariance
  - `streaming_covariance(predictions, predictions)`, i.e. variance
  - `streaming_covariance(labels, labels)`, i.e. variance

  The product-moment correlation ultimately returned is an idempotent operation
  `cov(predictions, labels) / sqrt(var(predictions) * var(labels))`. To
  facilitate correlation computation across multiple batches, the function
  groups the `update_op`s of the underlying streaming_covariance and returns an
  `update_op`.

  If `weights` is not None, then it is used to compute a weighted correlation.
  NOTE: these weights are treated as "frequency weights", as opposed to
  "reliability weights". See discussion of the difference on
  https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

  Args:
    predictions: A `Tensor` of arbitrary size.
    labels: A `Tensor` of the same size as predictions.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or the
      same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric value
      variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    pearson_r: A `Tensor` representing the current Pearson product-moment
      correlation coefficient, the value of
      `cov(predictions, labels) / sqrt(var(predictions) * var(labels))`.
    update_op: An operation that updates the underlying variables appropriately.

  Raises:
    ValueError: If `labels` and `predictions` are of different sizes, or if
      `weights` is the wrong size, or if either `metrics_collections` or
      `updates_collections` are not a `list` or `tuple`.
  """
  with variable_scope.variable_scope(name, 'pearson_r',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    # Broadcast weights here to avoid duplicate broadcasting in each call to
    # `streaming_covariance`.
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(weights, labels)
    cov, update_cov = streaming_covariance(
        predictions, labels, weights=weights, name='covariance')
    var_predictions, update_var_predictions = streaming_covariance(
        predictions, predictions, weights=weights, name='variance_predictions')
    var_labels, update_var_labels = streaming_covariance(
        labels, labels, weights=weights, name='variance_labels')

    pearson_r = math_ops.truediv(
        cov,
        math_ops.multiply(
            math_ops.sqrt(var_predictions), math_ops.sqrt(var_labels)),
        name='pearson_r')
    update_op = math_ops.truediv(
        update_cov,
        math_ops.multiply(
            math_ops.sqrt(update_var_predictions),
            math_ops.sqrt(update_var_labels)),
        name='update_op')

  if metrics_collections:
    ops.add_to_collections(metrics_collections, pearson_r)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return pearson_r, update_op

