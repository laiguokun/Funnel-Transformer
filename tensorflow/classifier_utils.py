"""Utils to create classification data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenize_fn, truncate_len=0):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    pad_feature = InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[1] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)
    return pad_feature, max_seq_length

  if label_list is not None:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  example_len = 0
  tokens_a = tokenize_fn(example.text_a)
  example_len += len(tokens_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenize_fn(example.text_b)
    example_len += len(tokens_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for two [SEP] & one [CLS] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3 - truncate_len)
  else:
    # Account for one [SEP] & one [CLS] with "- 2"
    if len(tokens_a) > max_seq_length - 2 - truncate_len:
      tokens_a = tokens_a[:max_seq_length - 2 - truncate_len]

  input_ids = []
  segment_ids = []
  if tokens_b is not None:
    input_ids = ([FLAGS.cls_id] +
                 tokens_a + [FLAGS.sep_id] + tokens_b + [FLAGS.sep_id])
    segment_ids = ([FLAGS.seg_id_cls] +
                   [FLAGS.seg_id_a] * (len(tokens_a) + 1) +
                   [FLAGS.seg_id_b] * (len(tokens_b) + 1))
  else:
    input_ids = [FLAGS.cls_id] + tokens_a + [FLAGS.sep_id]
    segment_ids = ([FLAGS.seg_id_cls] +
                   [FLAGS.seg_id_a] * (len(tokens_a) + 1))

  # The mask has 0 for real tokens and 1 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [0] * len(input_ids)

  # Zero-pad up to the sequence length.
  if len(input_ids) < max_seq_length:
    delta_len = max_seq_length - len(input_ids)
    input_ids = input_ids + [FLAGS.pad_id] * delta_len
    input_mask = input_mask + [1] * delta_len
    segment_ids = segment_ids + [FLAGS.seg_id_pad] * delta_len

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if label_list is not None:
    label_id = label_map[example.label]
  else:
    label_id = example.label
  if ex_index < 10:
    tf.logging.info("********** Example **********")
    tf.logging.info("guid: %s", example.guid)
    tf.logging.info("label: %s (id = %s)", example.label, label_id)
    tf.logging.info("length: %s", example_len)
    tf.logging.info("text_a: %s", example.text_a)
    if example.text_b:
      tf.logging.info("text_b: %s", example.text_b)
    tf.logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature, example_len


def clean_web_text(text):
  """Some rules used to clean web text."""
  text = text.replace("<br />", " ")
  text = text.replace("&quot;", "\"")
  text = text.replace("<p>", " ")
  if "<a href=" in text:
    while "<a href=" in text:
      start_pos = text.find("<a href=")
      end_pos = text.find(">", start_pos)
      if end_pos != -1:
        text = text[:start_pos] + text[end_pos + 1:]
      else:
        text = text[:start_pos] + text[start_pos + len("<a href=")]

    text = text.replace("</a>", "")
  text = text.replace("\\n", " ")
  text = text.replace("\\", " ")
  text = " ".join(text.split())
  return text

