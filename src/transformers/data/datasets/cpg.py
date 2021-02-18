import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_bart import BartTokenizer, BartTokenizerFast
from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.cpg import cpg_convert_examples_to_features, cpg_output_modes, cpg_processors, cpg_seq_lengths, mc_convert_examples_to_features
from ..processors.utils import InputFeatures
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class CPGDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", "})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_label_length: int = field(
        default=10,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )



class Split(Enum):
    train = "train"
    dev = "validation"
    test = "test"


class CPGDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: CPGDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: CPGDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        task_name = None,
        mc = False
    ):
        self.args = args
        self.task_name = task_name
        self.processor = cpg_processors[task_name]()
        self.output_mode = cpg_output_modes[task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file

        if task_name in cpg_seq_lengths and mode.value in cpg_seq_lengths[task_name]:
            max_seq_length = cpg_seq_lengths[task_name][mode.value][0]
            max_label_length = cpg_seq_lengths[task_name][mode.value][1]
        else:
            max_seq_length = args.max_seq_length
            max_label_length = args.max_label_length

        group_max_seq_length = max_seq_length + max_label_length
        if not mc:
            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else args.data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value, tokenizer.__class__.__name__, str(group_max_seq_length), task_name,
                ),
            )
        else:
            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else args.data_dir,
                "cached_mcqa2_{}_{}_{}_{}".format(
                    mode.value, tokenizer.__class__.__name__, str(group_max_seq_length), task_name,
                ),
            )
        self.label_list = self.processor.get_labels(task_name)

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(task_name)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(task_name)
                else:
                    examples = self.processor.get_train_examples(task_name)
                if limit_length is not None:
                    examples = examples[:limit_length]
                if not mc :
                    self.features = cpg_convert_examples_to_features(
                        examples,
                        tokenizer,
                        max_length=max_seq_length,
                        label_list=self.label_list,
                        output_mode=self.output_mode,
                        max_label_length=max_label_length,
                    )
                else:
                    self.features = mc_convert_examples_to_features(
                        examples,
                        tokenizer,
                        max_length=max_seq_length,
                        label_list=self.label_list,
                        output_mode=self.output_mode,
                        max_label_length=max_label_length,
                    )

                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
