import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

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
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

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

    def __iter__(self):
        def iterator():
            n_examples = len(self)
            indices = list(range(n_examples))
            random.shuffle(indices)
            for batch_begin in range(0, n_examples - self.batch_size, self.batch_size):
                batch_end = batch_begin + self.batch_size
                batch = [torch.tensor(self.examples[indices[i]], dtype=torch.long)
                         for i in range(batch_begin, batch_end)]
                batch = self.data_collator(batch)
                batch['language'] = self.language
                yield batch
        
        return iterator


class MultilingualDataset(Dataset):

    def __init__(
        self, tokenizer: PreTrainedTokenizer, files_by_language, block_size, batch_size, overwrite_cache=False,
    ):
        self.datasets = {
                language: MonolingualDataset(tokenizer, file_path, block_size, language,
                                             batch_size, overwrite_cache=overwrite_cache)
                for language, file_path in files_by_language.items()
        }

        self.length = min(len(dataset) for dataset in self.datasets.values()) // batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        def iterator():
            monolingual_iterators = [iter(dataset) for dataset in self.datasets.values()]
            while True:
                for dataset in monolingual_iterators:
                    try:
                        yield next(dataset)
                    except StopIteration:
                        return

        return iterator


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
