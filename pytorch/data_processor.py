"""Data preprocessors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import math
import os
import time


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


#######################################
########## Generl structures ##########
#######################################
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
    with open(input_file, "r") as f:
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
      text_a = clean_web_text(line[1])
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
      text_a = clean_web_text(line[1])
      text_b = clean_web_text(line[2])
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
      for filename in os.listdir(cur_dir):
        if not filename.endswith("txt"): continue
        path = os.path.join(cur_dir, filename)
        with open(path) as f:
          text = clean_web_text(f.read().strip())
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
        print("Incomplete line, ignored.")
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          print("Incomplete line, ignored.")
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          print("Incomplete line, ignored.")
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


class Sst2Processor(GLUEProcessor):
  """SST-2."""

  def __init__(self):
    super(Sst2Processor, self).__init__()
    self.label_column = 1
    self.text_a_column = 0
    self.test_text_a_column = 1
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"


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
        print("Incomplete line, ignored.")
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          print("Incomplete line, ignored.")
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          print("Incomplete line, ignored.")
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


PROCESSORS = {
    "cola": ColaProcessor,
    "mnli_matched": MnliMatchedProcessor,
    "mnli_mismatched": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "qqp": QqpProcessor,
    "rte": RteProcessor,
    "sst-2": Sst2Processor,
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


