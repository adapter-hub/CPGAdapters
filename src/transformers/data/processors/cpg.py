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
""" GLUE processors and helpers """

import logging
import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
import glob
from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample, InputFeatures
import json
import copy
from dataclasses import dataclass
from datasets import load_dataset
logger = logging.getLogger(__name__)
from tqdm import *
import csv

def cpg_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    max_label_length: Optional[int] = 10,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _cpg_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode,
        max_label_length=max_label_length
    )


def _cpg_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    max_label_length=None,
):
    # if max_length is None:
    #     max_length = tokenizer.max_len

    if task is not None:
        processor = CPGProcessor()
        # if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))
        # if output_mode is None:
        output_mode = "classification"
        logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if isinstance(example.label, list):
            return [label_map[el] for el in example.label]
        return label_map[example.label]
    # print("TRAIN:")
    # print(max([len(ex) for ex in tokenizer([(example.text_a) for example in examples], truncation=True, )['input_ids']]))
    # print(max([len(ex) for i in range(len(examples[0].text_b)) for ex in tokenizer([(tokenizer.sep_token + example.text_b[i] + tokenizer.sep_token) for example in examples],   truncation=True, add_special_tokens=True )['input_ids'] ]))

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    batch_encoding_labels = []

    # max([len(ex) for ex in tokenizer([(example.text_a) for example in examples], truncation=True, )['input_ids']])
    # max([len(ex) for i in range(len(examples[0].text_b)) for ex in tokenizer([(tokenizer.sep_token + example.text_b[i] + tokenizer.sep_token) for example in examples],   truncation=True, add_special_tokens=True )['input_ids'] ])
    for i in range(len(examples[0].text_b)):
        batch_encoding_labels.append(tokenizer(
            [(example.text_b[i] + tokenizer.sep_token) for example in examples],
            max_length=max_label_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False
        ))

    for i in range(len(batch_encoding.data['input_ids'])):
        for j in range(len(examples[0].text_b)):
            for key in batch_encoding.data.keys():
                batch_encoding.data[key][i] += batch_encoding_labels[j].data[key][i]

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i], max_label_length=max_label_length, max_length=max_length)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features




def mc_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    max_label_length: Optional[int] = 10,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _mc_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode,
        max_label_length=max_label_length
    )

@dataclass(frozen=True)
class MCInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
    max_length: Optional[int]
    max_label_length: Optional[int]


def _mc_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    max_label_length=None,
):
    # if max_length is None:
    #     max_length = tokenizer.max_len

    if task is not None:
        processor = CPGProcessor()
        # if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))
        # if output_mode is None:
        output_mode = "classification"
        logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if isinstance(example.label, list):
            return [label_map[el] for el in example.label]
        return label_map[example.label]

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []

        text_a = tokenizer(
            "Q: " + example.text_a,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )


        for ending_idx, ending in enumerate(example.text_b):
            text_b = tokenizer(
                tokenizer.sep_token + "A: " + ending + tokenizer.sep_token,
                max_length=max_label_length +2,
                padding="max_length",
                truncation=True,
                add_special_tokens=False
            )

            text_a_copy = copy.deepcopy(text_a)

            for key in text_a_copy.data.keys():
                text_a_copy.data[key] += text_b.data[key]

            choices_inputs.append(text_a_copy)


        label = label_from_example(example)

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            MCInputFeatures(
                example_id=example.guid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                max_label_length=max_label_length,
                max_length=max_length
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class CPGProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def load_file_data(self, file_name):

        'test.en.jsonl'

        data = []

        with open(file_name, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def load_line_data(self, file_name):

        'test.en.jsonl'

        data = []

        with open(file_name, 'r') as f:
            for line in f:
                data.append(line.split('\t'))
        return data

    def load_csv_data(self, file_name):
        data = []
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0: continue
                data.append(row)
        return data

    def load_pair_line_data(self, file_name, label_file_name):

        'test.en.jsonl'

        data = []

        with open(file_name, 'r') as f:
            with open(label_file_name, 'r') as l:
                for line, label in zip(f, l):
                    data.append([line.strip(), int(label)])
        return data

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def get_train_examples(self,task_name):
        """See base class."""
        self.task_name = task_name
        if task_name == 'copa':
            return self._create_examples(load_dataset('super_glue', name = 'copa', split='train'), 'train')
        if task_name == 'cb':
            return self._create_examples(load_dataset('super_glue', name = 'cb', split='train'), 'train')
        if task_name == 'sst-2':
            return self._create_examples(self._read_tsv(os.path.join("glue_data/SST-2", "train.tsv")), "train")
        if task_name.startswith('bzs_emotion'):
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/emotion/", "train_pu_half_v0.txt")), "train")
        if task_name == 'bzs_situation':
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/situation/", "train_pu_half_v0.txt")), "train")
        if task_name == 'bzs_topic':
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/topic/", "train_pu_half_v0.txt")), "train")
        if task_name == 'tweeteval_emotion':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/emotion/", "train_text.txt"), os.path.join("data/tweeteval/emotion/", "train_labels.txt")), "train")
        if task_name == 'tweeteval_hate':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/hate/", "train_text.txt"), os.path.join("data/tweeteval/hate/", "train_labels.txt")), "train")
        if task_name == 'tweeteval_irony':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/irony/", "train_text.txt"), os.path.join("data/tweeteval/irony/", "train_labels.txt")), "train")
        if task_name == 'tweeteval_offensive':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/offensive/", "train_text.txt"), os.path.join("data/tweeteval/offensive/", "train_labels.txt")), "train")
        if task_name == 'tweeteval_sentiment':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/sentiment/", "train_text.txt"), os.path.join("data/tweeteval/sentiment/", "train_labels.txt")), "train")
        if task_name == 'tweeteval_stance_abortion':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/abortion", "train_text.txt"), os.path.join("data/tweeteval/stance/abortion", "train_labels.txt")), "train")
        if task_name == 'tweeteval_stance_atheism':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/atheism", "train_text.txt"), os.path.join("data/tweeteval/stance/atheism", "train_labels.txt")), "train")
        if task_name == 'tweeteval_stance_climate':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/climate", "train_text.txt"), os.path.join("data/tweeteval/stance/climate", "train_labels.txt")), "train")
        if task_name == 'tweeteval_stance_feminist':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/feminist", "train_text.txt"), os.path.join("data/tweeteval/stance/feminist", "train_labels.txt")), "train")
        if task_name == 'tweeteval_stance_hillary':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/hillary", "train_text.txt"), os.path.join("data/tweeteval/stance/hillary", "train_labels.txt")), "train")
        if task_name == 'mnli':
            return self._create_examples(self._read_tsv(os.path.join('glue_data/MNLI', "train.tsv")), "train")
        if task_name.startswith('ukp'):
            data = []
            sub_data = self._read_tsv(os.path.join('data/UKP/', task_name + ".tsv"))
            for s in sub_data:
                if s[-1] == 'train':
                    data.append(s)
            return self._create_examples(data, "train")
        if task_name == 'cosmos_qa':
            return self._create_examples(self.load_csv_data(os.path.join('data/COSMOSQA', "train.csv")), "train")
        if task_name == 'race':
            data_dir = 'data/RACE/'
            logger.info("LOOKING AT {} train".format(data_dir))
            high = os.path.join(data_dir, "train/high")
            middle = os.path.join(data_dir, "train/middle")
            high = self._read_txt(high)
            middle = self._read_txt(middle)
            return self._create_examples(high + middle, "train")
        if task_name == 'banking':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/banking/', "train.csv")), "train")
        if task_name == 'clinic':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/clinc/', "train.csv")), "train")
        if task_name == 'hwu':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/hwu/', "train.csv")), "train")

        return self._create_examples(load_dataset(task_name, split='train'), 'train')

    def get_dev_examples(self,task_name):
        """See base class."""
        self.task_name = task_name
        if task_name == 'copa':
            return self._create_examples(load_dataset('super_glue', name = 'copa', split='validation'), 'dev')
        if task_name == 'cb':
            return self._create_examples(load_dataset('super_glue', name = 'cb', split='validation'), 'dev')
        if task_name == 'sst-2':
            return self._create_examples(self._read_tsv(os.path.join("glue_data/SST-2", "dev.tsv")), "dev")
        if task_name.startswith('bzs_emotion'):
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/emotion/", "dev.txt")), "dev")
        if task_name == 'bzs_situation':
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/situation/", "dev.txt")), "dev")
        if task_name == 'bzs_topic':
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/topic/", "dev.txt")), "dev")
        if task_name == 'tweeteval_emotion':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/emotion/", "val_text.txt"), os.path.join("data/tweeteval/emotion/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_hate':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/hate/", "val_text.txt"), os.path.join("data/tweeteval/hate/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_irony':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/irony/", "val_text.txt"), os.path.join("data/tweeteval/irony/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_offensive':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/offensive/", "val_text.txt"), os.path.join("data/tweeteval/offensive/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_sentiment':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/sentiment/", "val_text.txt"), os.path.join("data/tweeteval/sentiment/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_stance_abortion':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/abortion/", "val_text.txt"), os.path.join("data/tweeteval/stance/abortion/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_stance_atheism':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/atheism/", "val_text.txt"), os.path.join("data/tweeteval/stance/atheism/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_stance_climate':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/climate/", "val_text.txt"), os.path.join("data/tweeteval/stance/climate/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_stance_feminist':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/feminist/", "val_text.txt"), os.path.join("data/tweeteval/stance/feminist/", "val_labels.txt")), "dev")
        if task_name == 'tweeteval_stance_hillary':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/hillary/", "val_text.txt"), os.path.join("data/tweeteval/stance/hillary/", "val_labels.txt")), "dev")
        if task_name == 'mnli':
            return self._create_examples(self._read_tsv(os.path.join('glue_data/MNLI', "dev_matched.tsv")), "dev")
        if task_name.startswith('ukp'):
            data = []
            sub_data = self._read_tsv(os.path.join('data/UKP/', task_name + ".tsv"))
            for s in sub_data:
                if s[-1] == 'val':
                    data.append(s)
            return self._create_examples(data, "dev")
        if task_name == 'cosmos_qa':
            return self._create_examples(self.load_csv_data(os.path.join('data/COSMOSQA', "valid.csv")), "dev")
        if task_name == 'race':
            data_dir = 'data/RACE/'
            logger.info("LOOKING AT {} dev".format(data_dir))
            high = os.path.join(data_dir, "dev/high")
            middle = os.path.join(data_dir, "dev/middle")
            high = self._read_txt(high)
            middle = self._read_txt(middle)
            return self._create_examples(high + middle, "dev")
        if task_name == 'banking':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/banking/', "val.csv")), "dev")
        if task_name == 'clinic':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/clinc/', "val.csv")), "dev")
        if task_name == 'hwu':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/hwu/', "val.csv")), "dev")
        return self._create_examples(load_dataset(task_name, split='validation'), 'dev')

    def get_test_examples(self,task_name):
        """See base class."""
        self.task_name = task_name
        # if task_name == 'copa':
        #     return self._create_examples(load_dataset('super_glue', name = 'copa', split='test'), 'test')
        if task_name == 'cb':
            return self._create_examples(load_dataset('super_glue', name = 'cb', split='test'), 'test')
        if task_name == 'sst-2':
            return self._create_examples(self._read_tsv(os.path.join("glue_data/SST-2", "test.tsv")), "test")
        if task_name == 'copa':
            return self._create_examples(self.load_file_data('data/copa/test.en.jsonl'), "test")
        if task_name.startswith('bzs_emotion'):
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/emotion/", "test.txt")), "test")
        if task_name == 'bzs_situation':
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/situation/", "test.txt")), "test")
        if task_name == 'bzs_topic':
            return self._create_examples(self.load_line_data(os.path.join("data/BenchmarkingZeroShot/topic/", "test.txt")), "test")
        if task_name == 'tweeteval_emotion':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/emotion/", "test_text.txt"), os.path.join("data/tweeteval/emotion/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_sentiment':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/sentiment/", "test_text.txt"), os.path.join("data/tweeteval/sentiment/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_hate':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/hate/", "test_text.txt"), os.path.join("data/tweeteval/hate/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_irony':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/irony/", "test_text.txt"), os.path.join("data/tweeteval/irony/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_offensive':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/offensive/", "test_text.txt"), os.path.join("data/tweeteval/offensive/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_stance_abortion':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/abortion/", "test_text.txt"), os.path.join("data/tweeteval/stance/abortion/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_stance_atheism':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/atheism/", "test_text.txt"), os.path.join("data/tweeteval/stance/atheism/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_stance_climate':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/climate/", "test_text.txt"), os.path.join("data/tweeteval/stance/climate/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_stance_feminist':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/feminist/", "test_text.txt"), os.path.join("data/tweeteval/stance/feminist/", "test_labels.txt")), "test")
        if task_name == 'tweeteval_stance_hillary':
            return self._create_examples(self.load_pair_line_data(os.path.join("data/tweeteval/stance/hillary/", "test_text.txt"), os.path.join("data/tweeteval/stance/hillary/", "test_labels.txt")), "test")
        if task_name == 'mnli':
            return self._create_examples(self._read_tsv(os.path.join('glue_data/MNLI', "test_matched.tsv")), "test")
        if task_name.startswith('ukp'):
            data = []
            sub_data = self._read_tsv(os.path.join('data/UKP/', task_name + ".tsv"))
            for s in sub_data:
                if s[-1] == 'test':
                    data.append(s)
            return self._create_examples(data, "test")
        if task_name == 'race':
            data_dir = 'data/RACE/'
            logger.info("LOOKING AT {} test".format(data_dir))
            high = os.path.join(data_dir, "test/high")
            middle = os.path.join(data_dir, "test/middle")
            high = self._read_txt(high)
            middle = self._read_txt(middle)
            return self._create_examples(high + middle, "test")
        if task_name == 'banking':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/banking/', "test.csv")), "test")
        if task_name == 'clinic':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/clinc/', "test.csv")), "test")
        if task_name == 'hwu':
            return self._create_examples(self.load_csv_data(os.path.join('data/dialoglue/hwu/', "test.csv")), "test")
        return self._create_examples(load_dataset(task_name, split='test'), 'test')

    def get_labels(self, task_name):
        """See base class."""
        if task_name == "commonsense_qa":
            return ["A", "B", "C", "D", "E"]
        if task_name == "social_i_qa":
            return ["answerA", "answerB", "answerC",]
        if task_name == "copa":
            return [0, 1]
        if task_name == "boolq":
            return [0, 1]
        if task_name == "cb":
            return [0, 1, 2]
        if task_name == "sst-2":
            return [0, 1]
        if task_name.startswith("bzs_emotion"):
            return ["joy", "sadness", "guilt", "disgust", "shame", "fear", "anger", "surprise", "love", "noemo"]
        if task_name == "bzs_situation":
            return ["food",
                    "infra",
                    "med",
                    "search",
                    "shelter",
                    "utils",
                    "water",
                    "evac",
                    "regimechange",
                    "terrorism",
                    "crimeviolence",
                    "out-of-domain"]
        if task_name == 'bzs_topic':
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if task_name == 'tweeteval_emotion':
            return [0,1,2,3]
        if task_name == 'tweeteval_hate':
            return [0,1]
        if task_name == 'tweeteval_irony':
            return [0,1]
        if task_name == 'tweeteval_offensive':
            return [0,1]
        if task_name == 'tweeteval_sentiment':
            return [0,1,2]
        if task_name.startswith( 'tweeteval_stance'):
            return [0,1,2]
        if task_name == 'mnli':
            return ["contradiction", "entailment", "neutral"]
        if task_name == 'race':
            return ["0", "1", "2", "3"]
        if task_name == 'cosmos_qa':
            return ["0", "1", "2", "3"]
        if task_name == 'piqa':
            return [0,1]
        if task_name.startswith('ukp'):
            return ['Argument_for', "Argument_against", 'NoArgument']

        if task_name == 'banking':
            return [
                "card_arrival",
                "card_linking",
                "exchange_rate",
                "card_payment_wrong_exchange_rate",
                "extra_charge_on_statement",
                "pending_cash_withdrawal",
                "fiat_currency_support",
                "card_delivery_estimate",
                "automatic_top_up",
                "card_not_working",
                "exchange_via_app",
                "lost_or_stolen_card",
                "age_limit",
                "pin_blocked",
                "contactless_not_working",
                "top_up_by_bank_transfer_charge",
                "pending_top_up",
                "cancel_transfer",
                "top_up_limits",
                "wrong_amount_of_cash_received",
                "card_payment_fee_charged",
                "transfer_not_received_by_recipient",
                "supported_cards_and_currencies",
                "getting_virtual_card",
                "card_acceptance",
                "top_up_reverted",
                "balance_not_updated_after_cheque_or_cash_deposit",
                "card_payment_not_recognised",
                "edit_personal_details",
                "why_verify_identity",
                "unable_to_verify_identity",
                "get_physical_card",
                "visa_or_mastercard",
                "topping_up_by_card",
                "disposable_card_limits",
                "compromised_card",
                "atm_support",
                "direct_debit_payment_not_recognised",
                "passcode_forgotten",
                "declined_cash_withdrawal",
                "pending_card_payment",
                "lost_or_stolen_phone",
                "request_refund",
                "declined_transfer",
                "Refund_not_showing_up",
                "declined_card_payment",
                "pending_transfer",
                "terminate_account",
                "card_swallowed",
                "transaction_charged_twice",
                "verify_source_of_funds",
                "transfer_timing",
                "reverted_card_payment?",
                "change_pin",
                "beneficiary_not_allowed",
                "transfer_fee_charged",
                "receiving_money",
                "failed_transfer",
                "transfer_into_account",
                "verify_top_up",
                "getting_spare_card",
                "top_up_by_cash_or_cheque",
                "order_physical_card",
                "virtual_card_not_working",
                "wrong_exchange_rate_for_cash_withdrawal",
                "get_disposable_virtual_card",
                "top_up_failed",
                "balance_not_updated_after_bank_transfer",
                "cash_withdrawal_not_recognised",
                "exchange_charge",
                "top_up_by_card_charge",
                "activate_my_card",
                "cash_withdrawal_charge",
                "card_about_to_expire",
                "apple_pay_or_google_pay",
                "verify_my_identity",
                "country_support"
            ]

        if task_name == 'clinic':
            return ["accept_reservations", "account_blocked", "alarm", "application_status", "apr", "are_you_a_bot", "balance", "bill_balance", "bill_due", "book_flight", "book_hotel", "calculator", "calendar", "calendar_update", "calories", "cancel", "cancel_reservation", "car_rental", "card_declined", "carry_on", "change_accent", "change_ai_name", "change_language", "change_speed", "change_user_name", "change_volume", "confirm_reservation", "cook_time", "credit_limit", "credit_limit_change", "credit_score", "current_location", "damaged_card", "date", "definition", "direct_deposit", "directions", "distance", "do_you_have_pets", "exchange_rate", "expiration_date", "find_phone", "flight_status", "flip_coin", "food_last", "freeze_account", "fun_fact", "gas", "gas_type", "goodbye", "greeting", "how_busy", "how_old_are_you", "improve_credit_score", "income", "ingredient_substitution", "ingredients_list", "insurance", "insurance_change", "interest_rate", "international_fees", "international_visa", "jump_start", "last_maintenance", "lost_luggage", "make_call", "maybe", "meal_suggestion", "meaning_of_life", "measurement_conversion", "meeting_schedule", "min_payment", "mpg", "new_card", "next_holiday", "next_song", "no", "nutrition_info", "oil_change_how", "oil_change_when", "order", "order_checks", "order_status", "pay_bill", "payday", "pin_change", "play_music", "plug_type", "pto_balance", "pto_request", "pto_request_status", "pto_used", "recipe", "redeem_rewards", "reminder", "reminder_update", "repeat", "replacement_card_duration", "report_fraud", "report_lost_card", "reset_settings", "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion", "rewards_balance", "roll_dice", "rollover_401k", "routing", "schedule_maintenance", "schedule_meeting", "share_location", "shopping_list", "shopping_list_update", "smart_home", "spelling", "spending_history", "sync_device", "taxes", "tell_joke", "text", "thank_you", "time", "timer", "timezone", "tire_change", "tire_pressure", "todo_list", "todo_list_update", "traffic", "transactions", "transfer", "translate", "travel_alert", "travel_notification", "travel_suggestion", "uber", "update_playlist", "user_name", "vaccines", "w2", "weather", "what_are_your_hobbies", "what_can_i_ask_you", "what_is_your_name", "what_song", "where_are_you_from", "whisper_mode", "who_do_you_work_for", "who_made_you", "yes"]

        if task_name == 'hwu':
            return ["alarm_query", "alarm_remove", "alarm_set", "audio_volume_down", "audio_volume_mute", "audio_volume_up", "calendar_query", "calendar_remove", "calendar_set", "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact", "email_query", "email_querycontact", "email_sendemail", "general_affirm", "general_commandstop", "general_confirm", "general_dontcare", "general_explain", "general_joke", "general_negate", "general_praise", "general_quirky", "general_repeat", "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on", "lists_createoradd", "lists_query", "lists_remove", "music_likeness", "music_query", "music_settings", "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio", "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock", "recommendation_events", "recommendation_locations", "recommendation_movies", "social_post", "social_query", "takeaway_order", "takeaway_query", "transport_query", "transport_taxi", "transport_ticket", "transport_traffic", "weather_query"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, str(i))

            if self.task_name == "commonsense_qa":
                text_a = line['question']
                text_b = line['choices']['text']
                label = None if set_type.startswith("test") else line['answerKey']

            elif self.task_name == "cosmos_qa":
                text_a = line[1] + ' ' + line[2]
                text_b = [line[3], line[4], line[5], line[6], ]
                label = line[7].strip()

            elif self.task_name == "social_i_qa":
                text_a = line['context'] + ' ' + line['question']
                text_b = [line['answerA'], line['answerB'], line['answerC'], ]
                label = self.get_labels(self.task_name)[int(line['label']) - 1]

            elif self.task_name == "copa":
                text_a = line['premise'] + ' What was the ' + line['question'] + '?'
                text_b = [line['choice1'], line['choice2'],  ]
                label = line['label']

            elif self.task_name == "boolq":
                text_a = line['passage']
                text_b = ["No.", "Yes." ]
                label = int(line['answer'])

            elif self.task_name == "piqa":
                text_a = line['goal']
                text_b = [line['sol1'], line['sol1']]
                label = int(line['label'])

            elif self.task_name == "cb":
                text_a = line['premise'] + line['hypothesis']
                text_b = ["The sentences entail each other (mean the same).", "The sentences contradict each other.", "The sentences are neutral (talk about different settings)." ]
                label = int(line['label'])

            elif self.task_name == "sst-2":
                text_index = 1 if set_type == "test" else 0
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[text_index]
                text_b = ["Negative.", "Positive"]
                label = None if set_type == "test" else int(line[1])
            elif self.task_name.startswith('bzs_emotion'):

                if line[1] in self.task_name:
                    guid = "%s-%s" % (set_type, i)
                    text_a = line[2]
                    text_b = self.get_labels(self.task_name)
                    text_b[-1] = "None of the above."
                    label = line[0]
                else:
                    continue

            elif self.task_name.startswith('bzs_situation'):

                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                text_b = ["food supply",
                    "infastructure",
                    "medical assistance",
                    "search/rescue",
                    "shelter",
                    "utilities, energy, or sanitation",
                    "water supply",
                    "evacuation",
                    "regime change",
                    "terrorism",
                    "crime violence",
                    "None of the above."]
                label = line[0].split()

            elif  self.task_name == 'bzs_topic':
                text_b = [
                    "Society & Culture",
                    "Science & Mathematics",
                    "Health",
                    "Education & Reference",
                    "Computers & Internet",
                    "Sports",
                    "Business & Finance",
                    "Entertainment & Music",
                    "Family & Relationships",
                    "Politics & Government",
                    ]
                label = line[0]
                text_a = line[1]

            elif self.task_name == 'tweeteval_emotion':
                text_a = line[0]
                label = line[1]
                text_b = ['anger', 'joy', 'optimism', 'sadness']

            elif self.task_name == 'tweeteval_hate':
                text_a = line[0]
                label = line[1]
                text_b = ['not hateful', 'hateful',]

            elif self.task_name == 'tweeteval_irony':
                text_a = line[0]
                label = line[1]
                text_b = ['not irony', 'iron',]

            elif self.task_name == 'tweeteval_offensive':
                text_a = line[0]
                label = line[1]
                text_b = ['not offensive', 'offensive',]

            elif self.task_name == 'tweeteval_sentiment':
                text_a = line[0]
                label = line[1]
                text_b = ['negative', 'neutral', 'positive',]

            elif self.task_name == 'tweeteval_stance_abortion':
                text_a = line[0]
                label = line[1]
                text_b = ['neutral', 'against legalizing abortion', 'in favor of legalizing abortion',]

            elif self.task_name == 'tweeteval_stance_atheism':
                text_a = line[0]
                label = line[1]
                text_b = ['neutral', 'religious', 'atheist',]

            elif self.task_name == 'tweeteval_stance_climate':
                text_a = line[0]
                label = line[1]
                text_b = ['neutral', 'climate change is not a concern', 'climate change is a concern',]

            elif self.task_name == 'tweeteval_stance_feminist':
                text_a = line[0]
                label = line[1]
                text_b = ['neutral', 'against feminist movement', 'in favor of feminist movement',]

            elif self.task_name == 'tweeteval_stance_hillary':
                text_a = line[0]
                label = line[1]
                text_b = ['neutral', 'against Hillary Clinton', 'in favor of Hillary Clinton',]

            elif self.task_name == 'mnli':
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[8] + ' <SEP> ' + line[9]
                text_b = ["contradiction", "entailment", "neutral"]
                label = None if set_type.startswith("test") else line[-1]
            elif self.task_name == 'race':

                race_id = "%s-%s" % (set_type, line["race_id"])
                article = line["article"]
                for i in range(len(line["answers"])):
                    truth = str(ord(line["answers"][i]) - ord("A"))
                    question = line["questions"][i]
                    options = line["options"][i]

                    examples.append(
                        InputExample(
                            guid=race_id,
                            text_a= question + ' ' + article ,
                            text_b=[options[0], options[1], options[2], options[3]],
                            label=truth,
                        ))

            elif self.task_name.startswith('ukp'):
                concept = self.task_name[4:].replace('_', ' ') + '.'
                text_a = line[4]
                label = line[5]
                text_b = ['Argument for ' + concept, "Argument against " + concept, 'None of the above.']

            if self.task_name in ['clinic', 'banking', 'hwu']:
                if i == 0: continue
                text_a = line[0]
                label = line[1]
                text_b = [label.replace('_', ' ') for label in self.get_labels(self.task_name)]

            if self.task_name != 'race':
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def return_mc(task_name, ds):
    return ds

def return_class(task_name, ds):
    return cpg_tasks_num_labels[task_name]

cpg_tasks_num_labels = {
    'banking': 77,
    'clinic': 150,
    'hwu': 64,
    "ukp_abortion": 3,
    "ukp_cloning": 3,
    "ukp_death_penalty": 3,
    "ukp_gun_control": 3,
    "ukp_marijuana_legalization": 3,
    "ukp_minimum_wage": 3,
    "ukp_nuclear_energy": 3,
    "ukp_school_uniforms": 3,
    "piqa": 2,
    "cosmos_qa": 4,
    "race": 4,
    "commonsense_qa": 5,
    "social_i_qa": 3,
    "copa": 2,
    # "boolq": 2,
    # "cb": 3,
    "sst-2": 2,
    # "bzs_situation": 12,
    "bzs_emotion_fairytale_sentences": 10,
    "bzs_emotion_artificial_sentences": 10,
    "bzs_emotion_tweets": 10,
    "bzs_emotion_emotional_events": 10,
    "tweeteval_emotion": 4,
    "tweeteval_hate": 2,
    "tweeteval_irony": 2,
    "tweeteval_offensive": 2,
    "tweeteval_sentiment": 3,
    "tweeteval_stance_abortion": 3,
    "tweeteval_stance_atheism": 3,
    "tweeteval_stance_climate": 3,
    "tweeteval_stance_feminist": 3,
    "tweeteval_stance_hillary": 3,
    # "bzs_topic": 10,
    # 'mnli':3
}


cpg_diversity = {
    "banking": return_class,
    "clinic": return_class,
    "hwu": return_class,
    "ukp_abortion": return_class,
    "ukp_cloning": return_class,
    "ukp_death_penalty": return_class,
    "ukp_gun_control": return_class,
    "ukp_marijuana_legalization": return_class,
    "ukp_minimum_wage": return_class,
    "ukp_nuclear_energy": return_class,
    "ukp_school_uniforms": return_class,
    "piqa": return_mc,
    "cosmos_qa": return_mc,
    "race": return_mc,
    "commonsense_qa": return_mc,
    "social_i_qa": return_mc,
    "copa": return_class,
    "boolq": return_class,
    "cb": return_class,
    "sst-2": return_class,
    "bzs_situation": return_class,
    "bzs_emotion_fairytale_sentences": return_class,
    "bzs_emotion_artificial_sentences": return_class,
    "bzs_emotion_tweets": return_class,
    "bzs_emotion_emotional_events": return_class,
    "tweeteval_emotion": return_class,
    "tweeteval_hate": return_class,
    "tweeteval_irony": return_class,
    "tweeteval_offensive": return_class,
    "tweeteval_sentiment": return_class,
    "tweeteval_stance_abortion": return_class,
    "tweeteval_stance_atheism": return_class,
    "tweeteval_stance_climate": return_class,
    "tweeteval_stance_feminist": return_class,
    "tweeteval_stance_hillary": return_class,
}


cpg_processors = {
    "banking": CPGProcessor,
    "clinic": CPGProcessor,
    "hwu": CPGProcessor,
    "ukp_abortion": CPGProcessor,
    "ukp_cloning": CPGProcessor,
    "ukp_death_penalty": CPGProcessor,
    "ukp_gun_control": CPGProcessor,
    "ukp_marijuana_legalization": CPGProcessor,
    "ukp_minimum_wage": CPGProcessor,
    "ukp_nuclear_energy": CPGProcessor,
    "ukp_school_uniforms": CPGProcessor,
    "piqa": CPGProcessor,
    "cosmos_qa": CPGProcessor,
    "race": CPGProcessor,
    "commonsense_qa": CPGProcessor,
    "social_i_qa": CPGProcessor,
    "copa": CPGProcessor,
    "boolq": CPGProcessor,
    "cb": CPGProcessor,
    "sst-2": CPGProcessor,
    "bzs_situation": CPGProcessor,
    "bzs_emotion_fairytale_sentences": CPGProcessor,
    "bzs_emotion_artificial_sentences": CPGProcessor,
    "bzs_emotion_tweets": CPGProcessor,
    "bzs_emotion_emotional_events": CPGProcessor,
    "bzs_topic": CPGProcessor,
    "tweeteval_emotion": CPGProcessor,
    "tweeteval_hate": CPGProcessor,
    "tweeteval_irony": CPGProcessor,
    "tweeteval_offensive": CPGProcessor,
    "tweeteval_sentiment": CPGProcessor,
    "tweeteval_stance_abortion": CPGProcessor,
    "tweeteval_stance_atheism": CPGProcessor,
    "tweeteval_stance_climate": CPGProcessor,
    "tweeteval_stance_feminist": CPGProcessor,
    "tweeteval_stance_hillary": CPGProcessor,
    "mnli": CPGProcessor,
}

cpg_output_modes = {
    "banking": "classification",
    "clinic": "classification",
    "hwu": "classification",
    "ukp_abortion": "classification",
    "ukp_cloning": "classification",
    "ukp_death_penalty": "classification",
    "ukp_gun_control": "classification",
    "ukp_marijuana_legalization": "classification",
    "ukp_minimum_wage": "classification",
    "ukp_nuclear_energy": "classification",
    "ukp_school_uniforms": "classification",
    "piqa": "classification",
    "cosmos_qa": "classification",
    "race": "classification",
    "commonsense_qa": "classification",
    "social_i_qa": "classification",
    "copa": "classification",
    "boolq": "classification",
    "cb": "classification",
    "sst-2": "classification",
    "bzs_situation": "classification",
    "bzs_emotion_fairytale_sentences": "classification",
    "bzs_emotion_artificial_sentences": "classification",
    "bzs_emotion_tweets": "classification",
    "bzs_emotion_emotional_events": "classification",
    "bzs_topic": "classification",
    "tweeteval_emotion": "classification",
    "tweeteval_hate": "classification",
    "tweeteval_irony": "classification",
    "tweeteval_offensive": "classification",
    "tweeteval_sentiment": "classification",
    "tweeteval_stance_abortion": "classification",
    "tweeteval_stance_atheism": "classification",
    "tweeteval_stance_climate": "classification",
    "tweeteval_stance_feminist": "classification",
    "tweeteval_stance_hillary": "classification",
    "mnli": "classification",
}

cpg_seq_lengths = {

    "ukp_abortion": {
        'test': [134, 9],
        'validation': [106, 9],
        'train': [153, 9],
        'batch_size': 8,
    },
    "ukp_cloning": {
        'test': [142, 9],
        'validation': [95, 9],
        'train': [192, 9],
        'batch_size': 8,
    },
    "ukp_death_penalty": {
        'test': [158, 10],
        'validation': [261, 10],
        'train': [292, 10],
        'batch_size': 8,
    },
    "ukp_gun_control": {
        'test': [121, 10],
        'validation': [93, 10],
        'train': [164, 10],
        'batch_size': 8,
    },
    "ukp_marijuana_legalization": {
        'test': [116, 10],
        'validation': [63, 10],
        'train': [157, 10],
        'batch_size': 8,
    },
    "ukp_minimum_wage": {
        'test': [140, 10],
        'validation': [81, 10],
        'train': [97, 10],
        'batch_size': 8,
    },
    "ukp_nuclear_energy": {
        'test': [175, 10],
        'validation': [82, 10],
        'train': [191, 10],
        'batch_size': 8,
    },
    "ukp_school_uniforms": {
        'test': [193, 10],
        'validation': [110, 10],
        'train': [208, 10],
        'batch_size': 8,
    },

    "piqa": {
        'validation': [29, 241],
        'train': [38, 237],
        'batch_size': 1,
    },
    "cosmos_qa": {
        'validation': [178, 46],
        'train': [201, 47],
        'batch_size': 1,
    },
    "race": {
        'test': [392, 30],
        'validation': [392, 30],
        'train': [392, 30],
        'batch_size': 1,
    },
    "mnli": {
        'test': [427, 7],
        'validation': [427, 7],
        'train': [427, 7],
        'batch_size': 1,
    },
    "bzs_situation": {
        'test': [380,11],
        'validation': [380,11],
        'batch_size':1,
                      },
    "bzs_emotion_fairytale_sentences": {
        'test': [122,9],
        'validation': [137,9],
        'batch_size':8,
                      },
    "bzs_emotion_artificial_sentences": {
        'test': [67,9],
        'validation': [55,9],
        'batch_size':16,
                      },
    "bzs_emotion_tweets": {
        'test': [94,9],
        'validation': [63,9],
        'batch_size':16,
                      },
    "bzs_topic": {
        'test': [432,8],
        'validation': [432,8],
        'batch_size':8,
                      },
    "bzs_emotion_emotional_events":  {
        'test': [124,9],
        'validation': [163,9],
        'batch_size':8,
                      },
    "commonsense_qa":  {
        'train': [73,36],
        'validation': [67,12],
        'batch_size':16,
                      },
    "social_i_qa":  {
        'train': [134, 44],
        'validation': [58,38],
        'batch_size':16,
                      },
    "copa":  {
        'test': [21,16],
        'validation': [21,16],
        'batch_size':16,
                      },
    "boolq":  {
        'validation': [500,6],
        'batch_size':1,
                      },
    "cb":  {
        'validation': [268,14],
        'batch_size':8,
                      },
    "sst-2":  {
        'validation': [63,7],
        'batch_size':16,
                      },
    "tweeteval_emotion":  {
        'test': [80,7],
        'validation': [71,7],
        'batch_size':16,
                      },
    "tweeteval_hate": {
        'test': [142,5],
        'validation': [142,5],
        'batch_size':16,
                      },
    "tweeteval_irony": {
        'test': [81,6],
        'validation': [56,6],
        'batch_size':16,
                      },
    "tweeteval_offensive": {
        'test': [132,5],
        'validation': [147,5],
        'batch_size':16,
                      },
    "tweeteval_sentiment": {
        'test': [74,5],
        'validation': [59,5],
        'batch_size':16,
                      },
    "tweeteval_stance_abortion": {
        'test': [61,9],
        'validation': [47,9],
        'batch_size':16,
                      },
    "tweeteval_stance_atheism": {
        'test': [52,6],
        'validation': [45,6],
        'batch_size':16,
                      },
    "tweeteval_stance_climate": {
        'test': [41,10],
        'validation': [45,10],
        'batch_size':16,
                      },
    "tweeteval_stance_feminist": {
        'test': [53,9],
        'validation': [57,9],
        'batch_size':16,
                      },
    "tweeteval_stance_hillary": {
        'test': [65,9],
        'validation': [47,9],
        'batch_size':16,
                      },
}

