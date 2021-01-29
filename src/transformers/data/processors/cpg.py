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

from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample, InputFeatures
from datasets import load_dataset
logger = logging.getLogger(__name__)


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
    max_label_length=10,
):
    if max_length is None:
        max_length = tokenizer.max_len

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
        return label_map[example.label]


    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    batch_encoding_labels = []

    for i in range(len(examples[0].text_b)):
        batch_encoding_labels.append(tokenizer(
            [(tokenizer.sep_token + example.text_b[i] + tokenizer.sep_token) for example in examples],
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

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

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

    def get_train_examples(self,task_name):
        """See base class."""
        self.task_name = task_name
        return self._create_examples(load_dataset(task_name, split='train'), 'train')

    def get_dev_examples(self,task_name):
        """See base class."""
        self.task_name = task_name
        return self._create_examples(load_dataset(task_name, split='validation'), 'dev')

    def get_test_examples(self,task_name):
        """See base class."""
        self.task_name = task_name
        return self._create_examples(load_dataset(task_name, split='test'), 'test')

    def get_labels(self, task_name):
        """See base class."""
        if task_name == "commonsense_qa":
            return ["A", "B", "C", "D", "E"]
        if task_name == "social_i_qa":
            return ["answerA", "answerB", "answerC",]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, str(i))

            if self.task_name == "commonsense_qa":
                text_a = line['question']
                text_b = line['choices']['text']
                label = None if set_type.startswith("test") else line['answerKey']

            if self.task_name == "social_i_qa":
                text_a = line['context'] + ' ' + line['question']
                text_b = [line['answerA'], line['answerB'], line['answerC'], ]
                label = self.get_labels(self.task_name)[int(line['label']) - 1]

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


cpg_tasks_num_labels = {
    "commonsense_qa": 5,
    "social_i_qa": 5,
}

cpg_processors = {
    "commonsense_qa": CPGProcessor,
    "social_i_qa": CPGProcessor,
}

cpg_output_modes = {
    "commonsense_qa": "classification",
    "social_i_qa": "classification",
}
