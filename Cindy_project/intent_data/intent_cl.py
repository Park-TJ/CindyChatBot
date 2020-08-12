
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
import numpy as np
import pandas as pd

import json
import collections
import re
import unicodedata
import six


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


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output

def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)

def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)

def convert_ids_to_tokens2(inv_vocab, ids):
  output = []
  for idss in ids:
    for key, valus in inv_vocab.items():
      if valus == idss:
        output.append(key)
  return output

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
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
          ### joonho.lim @ 2019-03-15
          # if start > 0:
            # substr = "##" + substr
          # print ( '[substr]\t%s\t%s\t%d\t%d' % ( substr, substr in self.vocab, start, end))
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
    output_tokens.insert(0, '[CLS]')
    output_tokens.append('[SEP]')
    return output_tokens

class Intent_model():
  def __init__(self):
    
    self.max_len = 29
    self.config_path = './intent_data/Bert_model/bert_config.json'
    self.data = pd.read_csv('./intent_data/Bert_model/category_data')
    with open('./intent_data/Bert_model/vocab.json', 'r') as read_file:
        self.vocab = json.loads(read_file.read())

    with tf.io.gfile.GFile(self.config_path, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      self.bert_params = map_stock_config_to_params(bc)
      self.bert_params.adapter_size = None


    self.intent_model = keras.models.load_model('./intent_data/Bert_model/nomal_news_weather_etc_kobert_model_category.h5',
                                        custom_objects={"BertModelLayer":BertModelLayer.from_params(self.bert_params, name="bert")} )
    self.classes = self.data.intent.unique().tolist()
    token_ids = []

    # for i in token_sentence:
    #   token_ids.append(convert_by_vocab(self.vocab, i))
    # self.max_len = max([len(i) for i in token_ids])
    

  def intent_classification(self, text):
    padding_text = WordpieceTokenizer(self.vocab)
    a= padding_text.tokenize(text)
    token_ids = convert_by_vocab(self.vocab, a)
    for x in range(self.max_len - len(token_ids)):
        token_ids.append(0)
    token_ids = np.array(token_ids)
    token_ids = token_ids.reshape(1,self.max_len, order= 'F')
    predictions = self.intent_model.predict(token_ids).argmax(axis=-1)
    return self.classes[int(predictions)]