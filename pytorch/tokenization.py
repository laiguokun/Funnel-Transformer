#! coding=utf-8
"""Tokenization related."""

import collections
import unicodedata

import six

SPIECE_UNDERLINE = "▁"


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  """Common text preprocessing."""
  if remove_space:
    outputs = " ".join(inputs.strip().split())
  else:
    outputs = inputs

  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode("utf-8", "ignore")

  outputs = outputs.replace(u"``", u'"').replace(u"''", u'"')   # pylint: disable=g-inconsistent-quotes
  outputs = outputs.replace(u'”', u'"').replace(u'“', u'"')   # pylint: disable=g-inconsistent-quotes
  outputs = outputs.replace(u"‘", u"'").replace(u"’", u"'")   # pylint: disable=g-inconsistent-quotes

  if not keep_accents:
    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  """Encode text into sentence pieces. return_unicode is used only for py2."""

  # note(zhiliny): in some systems, sentencepiece only accepts str for py2
  if six.PY2 and isinstance(text, unicode):
    text = text.encode("utf-8")

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          piece[:-1].replace(SPIECE_UNDERLINE, ""))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = piece.decode("utf-8", "ignore")
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


def encode_ids(sp_model, text, sample=False):
  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids


class SPMTokenizer(object):
  """Sentence piece tokenizer."""

  def __init__(self, spm_file, do_lower_case=True):
    import sentencepiece as spm

    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(spm_file)
    self.do_lower_case = do_lower_case

    self.punct_tokens = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))

  def convert_text_to_tokens(self, text):
    """Converts a raw text into a list of sentence pieces."""
    text = preprocess_text(text, lower=self.do_lower_case)
    return encode_pieces(self.sp_model, text)

  def convert_tokens_to_text(self, tokens):
    return self.sp_model.DecodePieces(tokens)

  def convert_tokens_to_ids(self, tokens):
    """Converts a list of sentence pieces into a list of int ids."""
    def to_unicode(x):
      if six.PY2 and isinstance(x, unicode):
        x = x.encode("utf-8")
      return x

    ids = [self.sp_model.PieceToId(to_unicode(token)) for token in tokens]
    return ids

  def convert_ids_to_tokens(self, ids):
    """Converts a list of int ids into a list of sentence pieces."""
    tokens = [self.sp_model.IdToPiece(token_id) for token_id in ids]
    return tokens

  def get_token_id(self, token):
    return self.sp_model.PieceToId(token)

  def convert_text_to_ids(self, text):
    text = preprocess_text(text, lower=self.do_lower_case)
    tokens = encode_pieces(self.sp_model, text, return_unicode=False)
    return self.convert_tokens_to_ids(tokens)

  def convert_ids_to_text(self, ids):
    return self.sp_model.DecodeIds(ids)

  def is_start_id(self, token_id):
    return self.is_start_token(self.sp_model.IdToPiece(token_id))

  def is_start_token(self, token):
    if (token.startswith(SPIECE_UNDERLINE) or token in self.punct_tokens):
      return True
    else:
      return False

  def is_func_id(self, token_id):
    return self.is_func_token(self.sp_model.IdToPiece(token_id))

  def is_func_token(self, token):
    return token != "<unk>" and token.startswith("<") and token.endswith(">")

  def get_vocab_size(self):
    return self.sp_model.GetPieceSize()


######## BERT Word Piece ########
def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  with open(vocab_file) as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      if token not in vocab:
        vocab[token] = len(vocab)
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def convert_text_to_tokens(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.convert_text_to_tokens(text):
      for sub_token in self.wordpiece_tokenizer.convert_text_to_tokens(token):
        split_tokens.append(sub_token)

    return split_tokens

  def get_token_id(self, token):
    return self.vocab[token]

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)

  def convert_text_to_ids(self, text):
    tokens = self.convert_text_to_tokens(text)
    return self.convert_tokens_to_ids(tokens)

  def convert_ids_to_text(self, ids):
    tokens = self.convert_ids_to_tokens(ids)
    return " ".join(tokens)

  def is_start_id(self, token_id):
    token = self.inv_vocab[token_id]
    return not token.startswith("##")

  def is_func_id(self, token_id):
    token = self.inv_vocab[token_id]
    return self.is_func_token(token)

  def is_func_token(self, token):
    return token != "<unk>" and token.startswith("<") and token.endswith(">")

  def get_vocab_size(self):
    return len(self.vocab)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def convert_text_to_tokens(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def convert_text_to_tokens(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def get_tokenizer(args):
  if args.tokenizer_type == "sent_piece":
    return SPMTokenizer(args.tokenizer_path, not args.cased)
  elif args.tokenizer_type == "word_piece":
    return FullTokenizer(args.tokenizer_path, not args.cased)
  else:
    raise NotImplementedError
