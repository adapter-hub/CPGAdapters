# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import glob
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock

import numpy as np
import pandas as pd
import pyconll

from transformers import (
        is_tf_available,
        is_torch_available,
        language_aware_data_collator,
        PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset, IterableDataset

    class TokenClassificationDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            task: str,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Union[Split, str] = Split.train,
        ):
            if isinstance(mode, Split):
                mode = mode.value
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir, "cached_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    examples = read_examples_from_file(task, data_dir, mode)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class _MultiSourceTokenClassificationDatasetIterator:

    def __init__(self, tc_datasets):
        datasets = sorted(list(tc_datasets.datasets.items()))
        self.generators = [dataset.generator() for _, dataset in datasets]
        self.n_steps = tc_datasets.n_steps
        self.step_count = 0 

    def __next__(self):
        if self.step_count >= self.n_steps:
            raise StopIteration()

        dataset = np.random.choice(self.generators)
        batch = next(dataset)
        self.step_count += 1
        return batch


class SingleSourceTokenClassificationDataset(TokenClassificationDataset):

    def __init__(
            self,
            language: str,
            batch_size: int,
            task: str,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Union[Split, str] = Split.train,
    ):
        super().__init__(
            task, data_dir, tokenizer, labels, model_type,
            max_seq_length=max_seq_length,
            overwrite_cache=overwrite_cache,
            mode=mode,
        )
        logging.info('Initialised %s dataset with %d examples' % (language, len(self)))
        self.language = language
        self.batch_size = batch_size
        self.data_collator = language_aware_data_collator(language)
        self.n_examples = len(self)

    def set_max_examples(self, n_examples):
        assert n_examples <= len(self)
        self.n_examples = n_examples

    def generator(self):
        indices = list(range(self.n_examples))
        while True:
            random.shuffle(indices)
            for batch_begin in range(0, self.n_examples - self.batch_size, self.batch_size):
                batch_end = batch_begin + self.batch_size
                batch = [self.features[indices[i]]
                         for i in range(batch_begin, batch_end)]
                batch = self.data_collator(batch)
                yield batch


class MultiSourceTokenClassificationDataset(IterableDataset):

    def __init__(
        self,
        task: str,
        data_dir: str,
        languages_file: str,
        batch_size: int,
        n_steps: int,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Union[Split, str] = Split.train,
        max_examples: int = None,
    ):
        language_df = pd.read_csv(languages_file, na_filter=False)
        self.languages = []
        self.datasets = {}
        for i in range(language_df.shape[0]):
            language = language_df['iso_code'][i]
            self.languages.append(language)
            path = os.path.join(data_dir, language_df['path'][i])
            dataset = SingleSourceTokenClassificationDataset(
                language=language,
                batch_size=batch_size,
                task=task,
                data_dir=path,
                tokenizer=tokenizer,
                labels=labels,
                model_type=model_type,
                max_seq_length=max_seq_length,
                overwrite_cache=overwrite_cache,
                mode=mode
            )
            self.datasets[language] = dataset

        if max_examples:
            n_examples = [(len(self.datasets[lang]), lang) for lang in self.languages]
            n_examples.sort()
            examples_left = max_examples
            for i, (dataset_size, language) in enumerate(n_examples):
                examples_per_language = examples_left // (len(n_examples) - i)
                examples_for_this_language = min(dataset_size, examples_per_language)
                logging.info(f'Dataset contains {examples_for_this_language} {language} examples')
                self.datasets[language].set_max_examples(examples_for_this_language)
                examples_left -= examples_for_this_language

        self.n_steps = n_steps

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        return _MultiSourceTokenClassificationDatasetIterator(self)


def read_examples_from_file(task, data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    if task == 'ner':
        return read_ner_examples_from_file(data_dir, mode)
    elif task == 'udpos':
        return read_udpos_examples_from_file(data_dir, mode)
    else:
        raise ValueError('Unrecognised task name: "%s"' % task)


def read_ner_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                if '\t' in line:
                    splits = line.split("\t")
                else:
                    splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples


def read_udpos_examples_from_file(data_dir, mode):
    pattern = f'{data_dir}/*-ud-{mode}.conllu'
    matching_files = list(glob.glob(pattern))
    if len(matching_files) != 1:
        raise Exception(f'There should be exactly one file matching pattern '
                        f'{pattern}, instead the following files match: {matching_files}')
    file_path = matching_files[0]
    corpus = pyconll.load_from_file(file_path)
    upos_labels = set(get_labels('udpos'))

    examples = []
    for i, sentence in enumerate(corpus):
        words = []
        labels = []
        corrupt = False
        for token in sentence:
            if token.upos is None:
                continue
            if token.upos not in upos_labels or token.form is None:
                logging.warn('upos = %s, xpos = %s' % (token.upos, token.xpos))
                logging.warn(sentence.conll())
                corrupt = True
                break
            words.append(token.form)
            labels.append(token.upos)
        if not corrupt:
            guid = f'{mode}-{i}'
            examples.append(InputExample(guid=guid, words=words, labels=labels))

    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
            )
        )
    return features


def get_labels(task: str, path: str = None) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if task == "ner" and "O" not in labels:
            labels = ["O"] + labels
        return labels
    elif task == "ner":
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    elif task == "udpos":
        return ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART',
                'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        #return ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN',
        #        'PRON', 'X', '_', 'ADV', 'INTJ', 'VERB', 'AUX']

