import os
import pandas as pd
import utils

from transformers import AutoTokenizer


def unk_proportion(examples):
    unk_count = 0
    token_count = 0
    for example in examples:
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            token_count += 1
            for token in word_tokens:
                if token == '[UNK]':
                    unk_count += 1
                    break
    return unk_count / token_count

def mfs_baseline(examples, lang):
    label_count = {}
    for example in examples:
        for label in example.labels:
            label_count[label] = label_count.get(label, 0) + 1

    majority_label = None
    majority_count = 0
    total = 0
    for label, count in label_count.items():
        if count > majority_count:
            majority_count = count
            majority_label = label
        total += count

    majority_proportion = majority_count / total
    print('Majority class for %s is %s (%.4f)' % (lang, majority_label, majority_proportion))
    return majority_proportion

tokenizer = AutoTokenizer.from_pretrained(
    'bert-base-multilingual-cased',
)

df = pd.read_csv('/home/aja63/projects/CPGAdapters/ud.csv')
mfs = []
unks = []
for lang, treebank in zip(df['iso_code'], df['treebank']):
    path = os.path.join('/mnt/hdd/aja63/datasets/ud-treebanks-v2.7', treebank)
    examples = utils.read_udpos_examples_from_file(path, 'test')
    unks.append(unk_proportion(examples))
    mfs.append(mfs_baseline(examples, lang))

df['udpos_mfs_baseline'] = mfs
df['unk_proportion'] = unks
df = df[['iso_code', 'udpos_mfs_baseline', 'unk_proportion']]
print(df)
df.to_csv('/home/aja63/projects/CPGAdapters/ud-with-mfs.csv', index=False)

