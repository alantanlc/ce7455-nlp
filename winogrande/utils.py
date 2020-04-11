# coding=utf-8

""" Utility for finetuning BERT/RoBERTa models on WinoGrande. """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score
import json

logger = logging.getLogger(__name__)

class InputExample(object):
  """ A single training/test example for simple sequence classification. """

  def __init__(self, guid, text_a, text_b=None, label=None):
    """ Constructs an InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence. Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class MCInputExample(object):
  def __init__(self, guid, options, label):
    self.guid = guid
    self.options = options
    self.label = label

class InputFeatures(object):
  """ A single set of features of data. """
  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

class MultipleChoiceFeatures(object):
  def __init__(self, example_id, option_features, label=None):
    self.example_id = example_id
    self.option_features = self.choices_features = [
      {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
      }
      for _, input_ids, input_mask, segment_ids in option_features
    ]
    self.label = int(label)

class DataProcessor(object):
  """ Base class for data converters for sequence classification data sets. """

  def get_train_examples(self, data_dir):
    """ Gets a collection of `InputExample`s for the train set. """
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """ Gets a collection of `InputExample`s for the dev set. """
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """ Gets a collection of `InputExample`s for the test set. """
    raise NotImplementedError()

  def get_labels(self):
    """ Gets the list of labels for this data set. """
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """ Reads a tab separated value file. """
    with open(input_file, "r", encoding="utf-8-sig") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        if sys.version_info[0] == 2:
          line = list(unicode(cell, 'utf-8') for cell in line)
        lines.append(line)
      return lines

  @classmethod
  def _read_jsonl(cls, input_file):
    """ Reads a tab separated value file. """
    records = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
      for line in f:
        records.append(json.loads(line))
      return records

class WinograndeProcessor(DataProcessor):

  def get_train_examples(self, data_dir, data_size='xs'):
    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, "train.jsonl")

  def get_dev_examples(self, data_dir):
    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))

  def get_test_examples(self, data_dir):
    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, "test.jsonl")))

  def get_labels(self):
    return ["1", "2"]

  def _create_examples(self, records):
    examples = []
    for (i, record) in enumerate(records):
      guid = record['qID']
      sentence = record['sentence']

      name1 = record['option1']
      name2 = record['option2']
      if not 'answer' in record:
        # This is a dummy label for test prediction.
        # test.jsonl doesn't include the `answer`.
        label = "1"
      else:
        label = record['answer']

      conj = "_"
      idx = sentence.index(conj)
      context = sentence[:idx]
      option_str = "_ " + sentence[idx + len(conj):].strip()

      option1 = option_str.replace("_", name1)
      option2 = option_str.replace("_", name2)

      mc_example = MCInputExample(
        guid=guid,
        options=[
          {
            'segment1': context,
            'segment2': option1
          },
          {
            'segment1': context,
            'segment2': option2
          }
        ],
        label=label
      )
      examples.append(mc_example)

    return examples

class SuperGlueWscProcessor(DataProcessor):
  """ Processor for the SuperGLUE-WSC """
  def get_train_examples(self, data_dir):
    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, "train.jsonl")))

  def get_dev_examples(self, data_dir):
    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))

  def get_test_examples(self, data_dir):
    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, "test.jsonl")))

  def get_labels(self):
    return ["0", "1"]

  def _create_examples(self, records):
    """ Creates examples for the training and dev sets. """
    examples = []
    for (i, record) in enumerate(records):
      guid = record['idx']
      text_a = record["sentence1"]
      text_b = record["sentence2"]
      label = record["label"]
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode, cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]', sep_token_extra=False, pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True):
  """ Loads a data file into a list of `InputBatch`s 
  `cls_token_at_end` define the location of the CLS token:
    - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
    - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
  `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
  """

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d of %d" % (ex_index, len(examples)))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    special_tokens_count = 3 if sep_token_extra else 2
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"

      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count - 1)
    else:
      # Account for [CLS] and [SEP] with "- 2" or "-3" for RoBERTa
      if len(tokens_a) > max_seq_length - special_tokens_count:
        tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:  [CLS] is this jack ##son ##vile ? [SEP] no it is not . [SEP]
    #  type_ids:  0   0  0    0    0     0      0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:  [CLS] the dog is hairy . [SEP]
    #  type_ids:  0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence of the second sequence.
    tokens = tokens_a + [sep_token]
    if sep_token_extra:
      # roberta uses an extra separator b/w pairs of sentences
      tokens += [sep_token[
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
      tokens += tokens_b + [sep_token]
      segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
      tokens = tokens + [cls_token]
      segment_ids = segment_ids + [cls_token_segment_id]
    else:
      tokens = [cls_token] + tokens
      segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids

