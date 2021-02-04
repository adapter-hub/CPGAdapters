import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset, IterableDataset

from ..data_collator import DataCollatorForLanguageModeling
from ...tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                lines = []
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        tokens = line.strip().split()
                        if len(tokens) < 10:
                            continue
                        lines.append(line.strip())
                text = '\n'.join(lines)

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class MonolingualDataset(TextDataset):

    def __init__(
        self, tokenizer, file_path, block_size, language, batch_size, overwrite_cache=False,
    ):
        super().__init__(tokenizer, file_path, block_size, overwrite_cache=overwrite_cache)
        logging.info('Initialised %s dataset with %d examples' % (language, len(self)))
        self.language = language
        self.batch_size = batch_size
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    def generator(self, repeat=False, shuffle=False):
        n_examples = len(self)
        while True:
            indices = list(range(n_examples))
            if shuffle:
                random.shuffle(indices)
            for batch_begin in range(0, n_examples - self.batch_size, self.batch_size):
                batch_end = batch_begin + self.batch_size
                batch = [torch.tensor(self.examples[indices[i]], dtype=torch.long)
                         for i in range(batch_begin, batch_end)]
                batch = self.data_collator(batch)
                batch['language'] = self.language
                yield batch
            if not repeat:
                break


class _MultilingualDatasetIterator:

    def __init__(self, ml_dataset):
        datasets = sorted(list(ml_dataset.datasets.items()))
        self.monolingual_generators = [
                dataset.generator(repeat=ml_dataset.training,
                                  shuffle=ml_dataset.training)
                for _, dataset in datasets
        ]

        if ml_dataset.weighted_sampling:
            lengths = np.array([len(dataset) for _, dataset in datasets], dtype=np.float32)
            smoothed_lengths = lengths ** ml_dataset.smoothing
            self.sample_prob = smoothed_lengths / np.sum(smoothed_lengths)
        else:
            self.sample_prob = np.ones([len(datasets)], dtype=np.float32) / len(datasets)
        logging.info('Languages will be sampled in the following proportions:')
        for i, (language, _) in enumerate(datasets):
            logging.info('%s: %.7f' % (language, self.sample_prob[i]))
        
        self.training = ml_dataset.training
        self.n_steps = ml_dataset.n_steps
        self.step_count = 0 
        self.current = 0

    def __next__(self):
        if self.training:
            if self.step_count >= self.n_steps:
                raise StopIteration()

            dataset = np.random.choice(self.monolingual_generators, p=self.sample_prob)
            batch = next(dataset)
            self.step_count += 1
            return batch

        else:
            while True:
                if self.current >= len(self.monolingual_generators):
                    raise StopIteration()

                try:
                    batch = next(self.monolingual_generators[self.current])
                except StopIteration:
                    self.current += 1
                    continue
                
                return batch


class MultilingualDataset(IterableDataset):

    def __init__(
        self, tokenizer: PreTrainedTokenizer, files_by_language, block_size, batch_size,
        overwrite_cache=False, training=False, weighted_sampling=True, smoothing=0.7, n_steps=250000
    ):
        self.training = training
        self.weighted_sampling = weighted_sampling
        self.smoothing = smoothing
        self.n_steps = n_steps
        languages = ', '.join(sorted(list(files_by_language.keys())))
        logging.info('Initialising multilingual dataset with languages ' + languages)
        self.datasets = {
                language: MonolingualDataset(tokenizer, file_path, block_size, language,
                                             batch_size, overwrite_cache=overwrite_cache)
                for language, file_path in files_by_language.items()
        }

        self.length = sum(
                len(dataset) // batch_size for dataset in self.datasets.values())

    def __len__(self):
        if self.training:
            return self.n_steps
        else:
            return self.length

    def __iter__(self):
        return _MultilingualDatasetIterator(self)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
