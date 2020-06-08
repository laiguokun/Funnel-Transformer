"""Data loader."""
import collections
import json
import os
import numpy as np
import torch

from distributed_utils import is_master
import tokenization
import data_processor


class BucketIterator(object):
  """Bucket Iterator."""

  def __init__(self, data, batch_size, pad_id, pad_seg_id, device="cpu",
               max_length=-1, trunc_length=0):
    self.data = data

    self.num_data = len(self.data)
    self.batch_size = batch_size
    self.pad_id = pad_id
    self.pad_seg_id = pad_seg_id
    self.device = device
    self.max_length = max_length
    self.trunc_length = trunc_length

    self.cache_size = self.batch_size * 1

  def _batchify(self, data):
    """Batch sequences with padding."""
    sents, seg_id, labels = zip(*data)
    max_len = max(x.size(0) for x in sents)
    if self.max_length is not None and self.max_length > 0:
      max_len = min(max_len, self.max_length)

    # deal with sequence truncation
    batch_len = max_len + self.trunc_length
    batch_tok_id = sents[0].new(len(sents), batch_len).fill_(self.pad_id)
    batch_seg_id = sents[0].new(len(sents), batch_len).fill_(self.pad_seg_id)
    for i in range(len(sents)):
      sent_len = min(sents[i].size(0), max_len)
      batch_seg_id[i, :sent_len].copy_(seg_id[i][:sent_len])
      # deal with the trailing [sep] token
      batch_tok_id[i, :sent_len-1].copy_(sents[i][:sent_len-1])
      batch_tok_id[i, sent_len-1].copy_(sents[i][-1])

    batch_tok_id = batch_tok_id.to(self.device)
    batch_seg_id = batch_seg_id.to(self.device)
    labels = torch.LongTensor(labels).to(self.device)

    return (batch_tok_id, batch_seg_id, labels)

  def yield_chunk(self, data_list, chunk_size, shuffle=False):
    offsets = [i for i in range(0, len(data_list), chunk_size)]

    # shuffle chunks insteads of samples in the chunk
    if shuffle:
      np.random.shuffle(offsets)

    for offset in offsets:
      yield data_list[offset:offset+chunk_size]

  def __getitem__(self, key):
    return self.data[key]

  def _process_batch(self, batch):
    return self._batchify(batch)

  def build_iterator(self, indices, shuffle):
    """Data iterator that yield batches."""
    if shuffle:
      for cache_indices in self.yield_chunk(indices, self.cache_size, True):
        cache = [self.__getitem__(idx) for idx in cache_indices]
        # sort samples in the cache
        cache = sorted(cache, key=lambda x: len(x[0]), reverse=True)
        for batch in self.yield_chunk(cache, self.batch_size, True):
          yield self._process_batch(batch)
    else:
      data = [self.data[i] for i in indices]
      for batch in self.yield_chunk(data, self.batch_size, False):
        yield self._process_batch(batch)

  def __len__(self):
    return self.num_data

  def get_iter(self, epoch=0, shuffle=True, distributed=False):
    """Get data iterator."""
    # make sure all processes share the same rng state
    np.random.seed(epoch)

    # shufle
    if shuffle:
      indices = np.random.permutation(len(self.data))
    else:
      indices = list(range(len(self.data)))

    # distributed
    if distributed:
      world_size = torch.distributed.get_world_size()
      local_rank = torch.distributed.get_rank()
      indices = indices[local_rank::world_size]

    return self.build_iterator(indices, shuffle=shuffle)

def setup_special_ids(args, tokenizer):
  """Set up the id of special tokens."""
  special_symbols_mapping = collections.OrderedDict([
      ("<unk>", "unk_id"),
      ("<s>", "bos_id"),
      ("</s>", "eos_id"),
      ("<cls>", "cls_id"),
      ("<sep>", "sep_id"),
      ("<pad>", "pad_id"),
      ("<mask>", "mask_id"),
      ("<eod>", "eod_id"),
      ("<eop>", "eop_id")
  ])
  args.vocab_size = tokenizer.get_vocab_size()
  if is_master(args):
    print("Set vocab_size: {}.".format(args.vocab_size))
  for sym, sym_id_str in special_symbols_mapping.items():
    try:
      sym_id = tokenizer.get_token_id(sym)
      setattr(args, sym_id_str, sym_id)
      if is_master(args):
        print("Set {} to {}.".format(sym_id_str, sym_id))
    except KeyError:
      if is_master(args):
        print("Skip {}: not found in tokenizer's vocab.".format(sym))


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

def create_pytorch_data(args, split, tokenizer, processor, label_dict):
  tok_basename = os.path.basename(args.tokenizer_path)
  file_base = "{}.len-{}.{}.pt".format(
      tok_basename, args.max_length, split)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  file_path = os.path.join(args.output_dir, file_base)

  # Load examples
  if split == "train":
    examples = processor.get_train_examples(args.data_dir)
  elif split == "dev":
    examples = processor.get_dev_examples(args.data_dir)
  elif split == "test":
    examples = processor.get_test_examples(args.data_dir)
  else:
    raise NotImplementedError
  print("Num of {} samples: {}".format(split, len(examples)))

  return convert_examples_to_tensors(args,
                                     examples,
                                     label_dict,
                                     tokenizer,
                                     file_path)


def convert_examples_to_tensors(args, examples, label_dict, tokenizer,
                                output_file):
  """Encode and cache raw data into pytorch format."""
  if not is_master(args) and args.distributed:
    torch.distributed.barrier()

  if not os.path.exists(output_file) or args.overwrite_data:
    sents, labels, seg_ids = [], [], []
    for (ex_index, example) in enumerate(examples):
      example_len = 0
      tokens_a = tokenizer.convert_text_to_ids(example.text_a)
      example_len += len(tokens_a)
      tokens_b = None
      if example.text_b:
        tokens_b = tokenizer.convert_text_to_ids(example.text_b)
        example_len += len(tokens_b)

      if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for two [SEP] & one [CLS] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, args.max_length - 3)
      else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > args.max_length - 2:
          tokens_a = tokens_a[:args.max_length - 2]

      input_ids = []
      segment_ids = []
      if tokens_b is not None:
        input_ids = ([args.cls_id] +
                     tokens_a + [args.sep_id] + tokens_b + [args.sep_id])
        segment_ids = ([args.seg_id_cls] +
                       [args.seg_id_a] * (len(tokens_a) + 1) +
                       [args.seg_id_b] * (len(tokens_b) + 1))
      else:
        input_ids = [args.cls_id] + tokens_a + [args.sep_id]
        segment_ids = ([args.seg_id_cls] +
                       [args.seg_id_a] * (len(tokens_a) + 1))

      # Label
      if label_dict is not None:
        label_id = label_dict[example.label]
      else:
        label_id = example.label

      input_ids = torch.LongTensor(input_ids)
      segment_ids = torch.LongTensor(segment_ids)

      sents.append(input_ids)
      seg_ids.append(segment_ids)
      labels.append(label_id)

    data = list(zip(sents, seg_ids, labels))

    torch.save(data, output_file)
  else:
    data = torch.load(output_file)

  if is_master(args) and args.distributed:
      torch.distributed.barrier()

  stat(args, data)

  return data

def stat(args, data):
  lengths = [len(x[0]) for x in data]
  if is_master(args):
    print("Number of sent: {}".format(len(data)))
    print("Sent length: mean {}, std {}, max {}".format(
        np.mean(lengths), np.std(lengths), np.max(lengths)))


def setup_tokenizer(args):
  tokenizer = tokenization.get_tokenizer(args)
  setup_special_ids(args, tokenizer)
  return tokenizer

def load_data(args):
  """Load train/valid/test data."""
  tokenizer = setup_tokenizer(args)
  processor = data_processor.PROCESSORS[args.dataset.lower()]()
  if args.dataset == "sts-b":
    label_dict = None
  else:
    label_dict = {}
    for (i, label) in enumerate(processor.get_labels()):
      label_dict[label] = i

  tr_data = create_pytorch_data(args, "train", tokenizer, processor, label_dict)
  va_data = create_pytorch_data(args, "dev", tokenizer, processor, label_dict)
  te_data = create_pytorch_data(args, "test", tokenizer, processor, label_dict)

  return (tr_data, va_data, te_data), label_dict
