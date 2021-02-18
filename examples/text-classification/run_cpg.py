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
""" Finetuning the library models for sequence classification on GLUE
 (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    # GlueDataset,
    AutoModelWithHeads,
)

import random
from transformers.data.datasets.cpg import CPGDataset, CPGDataTrainingArguments as DataTrainingArguments
from transformers.data.processors.cpg import cpg_tasks_num_labels, cpg_processors, cpg_output_modes, cpg_seq_lengths

from transformers.adapter_config import  CpgAdapterConfig

from transformers.data.data_collator import CPGCollator

from transformers import (
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    # glue_compute_metrics,
    # glue_output_modes,
    # glue_tasks_num_labels,
    set_seed,
)
from datasets import list_datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    score_file: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


    seed = random.choice(list(range(9999)))
    # Set seed
    set_seed(seed)

    # dataset = random.choice(['commonsense_qa', 'social_i_qa'])
    dataset = 'social_i_qa'
    batch_size = random.choice([16, 32, 64])
    batch_size = 8
    # lr = random.choice([1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4])
    epochs = random.choice([2,3,4,6,7,8,9,5, 10, 15, 20])
    epochs = 3
    # only_head = random.choices(population=[True, False], weights=[0.10, 0.90])[0]
    only_head = False
    if not only_head:
        lr = random.choice([5e-5, 1e-4, 2e-4, 5e-4])
        lr =  2e-4
        label_adapter = random.choice([True, False])
        add_label_noise = random.choice([True, False])

        if add_label_noise:

            # adapter_label_noise = random.choice([True, False])
            adapter_label_noise = False
            if adapter_label_noise:
                position_label_noise = random.choice([True, False])
            else:
                position_label_noise = True
        else:
            adapter_label_noise = False
            position_label_noise = False

        # pass_label_into_classifier = random.choice([True, False])
        pass_label_into_classifier = False

        if pass_label_into_classifier and add_label_noise:
            reuse_label_noise = random.choice([True, False])
        else:
            reuse_label_noise = False

        # adapter_layer_context = random.choice([False, True])
        adapter_layer_context = True

    else:
        lr = random.choice([1e-5, 2e-5, 5e-5, 1e-4])
        label_adapter = False
        add_label_noise = False
        adapter_label_noise = False
        position_label_noise = False
        pass_label_into_classifier = True
        reuse_label_noise = False
        adapter_layer_context = False

    # cpg_head_positions = random.choice([True, False])
    cpg_head_positions = True
    cpg_down_dim = random.choice([10, 50, 100, 300])
    cpg_down_dim = 10
    sample_nota = random.choice([True, False])
    if sample_nota:
        nota_prob = random.choice([0.5, 0.25, 0.1, 0.05])
    else:
        nota_prob = 0.0

    training_args.max_steps = 12000
    hp_dict = {

        "dataset": dataset,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "only_head": only_head,
        "label_adapter": label_adapter,
        "add_label_noise": add_label_noise,
        "adapter_label_noise": adapter_label_noise,
        "position_label_noise": position_label_noise,
        "pass_label_into_classifier": pass_label_into_classifier,
        "reuse_label_noise": reuse_label_noise,
        "adapter_layer_context": adapter_layer_context,
        "cpg_head_positions": cpg_head_positions,
        "cpg_down_dim": cpg_down_dim,
        "sample_nota": sample_nota,
        "nota_prob": nota_prob,
        "seed": seed,
    }

    # hp_dict = {
    #
    #     "dataset": 'social_i_qa',
    #     "batch_size": 8,
    #     "lr": lr,
    #     "epochs": epochs,
    #     "only_head": False,
    #     "label_adapter": True,
    #     "add_label_noise": False,
    #     "adapter_label_noise": False,
    #     "position_label_noise": False,
    #     "pass_label_into_classifier": False,
    #     "reuse_label_noise": False,
    #     "adapter_layer_context": False,
    #     "cpg_head_positions": True,
    #     "cpg_down_dim": 10,
    #     "sample_nota": True,
    #     "nota_prob": 0.1,
    #     "seed": seed,
    # }


    with open(model_args.score_file, 'a') as f:
        f.write('\n')
        for k,v in hp_dict.items():
            f.write(str(v) + '\t')

    training_args.per_device_train_batch_size = hp_dict['batch_size']
    training_args.learning_rate = hp_dict['lr']
    training_args.num_train_epochs = hp_dict['epochs']
    data_args.task_name = hp_dict['dataset']



    task_name = hp_dict['dataset']
    num_labels = cpg_tasks_num_labels[task_name]
    output_mode = cpg_output_modes[task_name]

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=model_args.cache_dir,
    )

    config.hp_dict = hp_dict

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    model.add_cpg_head(data_args.task_name,
                        bias=True,
                        max_label_length=data_args.max_label_length,
                        max_length=data_args.max_seq_length,)




    # Setup adapters
    if not hp_dict["only_head"] and adapter_args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
            # resolve the adapter config
            # adapter_config = AdapterConfig.load(
            #     adapter_args.adapter_config,
            #     non_linearity=adapter_args.adapter_non_linearity,
            #     reduction_factor=adapter_args.adapter_reduction_factor,
            # )

            cpg_adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )

            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter, AdapterType.text_task, config=cpg_adapter_config, load_as='cpg',
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter('cpg', AdapterType.text_task, config=cpg_adapter_config)
                if hp_dict["label_adapter"]:
                    model.add_adapter('labels', AdapterType.text_task, config=cpg_adapter_config)
            if hp_dict["adapter_layer_context"]:
                adapter_config = CpgAdapterConfig(**{"cpg": {"language_embedding_dim":768 *2, "down_dim":hp_dict["cpg_down_dim"], "languages":None, "layer_embedding_dim": None, "use_typoligy": False}})
            else:
                adapter_config = CpgAdapterConfig(**{"cpg": {"language_embedding_dim":768, "down_dim":hp_dict["cpg_down_dim"], "languages":None, "layer_embedding_dim": None, "use_typoligy": False}})

            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter, AdapterType.text_task, config=adapter_config, load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, AdapterType.text_task, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                AdapterType.text_lang,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter

        if hp_dict["label_adapter"]:
            model.train_adapter(['cpg','labels', task_name])
        else:
            model.train_adapter(['cpg', task_name])

        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters([lang_adapter_name, task_name])
        else:
            model.set_active_adapters([task_name])

    # Get datasets
    train_dataset = [(
        CPGDataset(data_args,
                   tokenizer=tokenizer,
                   cache_dir=model_args.cache_dir,
                   task_name=task_name_) if training_args.do_train else None
    # ) for task_name_ in ['mnli']]
    ) for task_name_ in ['clinic', 'banking', 'hwu', 'ukp_abortion', 'ukp_cloning', 'ukp_death_penalty', 'ukp_gun_control', 'ukp_marijuana_legalization',
                         'ukp_minimum_wage', 'ukp_nuclear_energy', 'ukp_school_uniforms', 'piqa','commonsense_qa', 'social_i_qa']]
    eval_datasets = [(
        CPGDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir, task_name=t)
        if training_args.do_eval
        else None
    ) for t in cpg_tasks_num_labels.keys()]

    test_datasets = [
        'clinic', 'banking', 'hwu',
        'ukp_abortion', 'ukp_cloning', 'ukp_death_penalty', 'ukp_gun_control', 'ukp_marijuana_legalization',
                         'ukp_minimum_wage', 'ukp_nuclear_energy', 'ukp_school_uniforms',
        'race',
        'copa',
        "bzs_situation",
        "bzs_emotion_fairytale_sentences",
        "bzs_emotion_artificial_sentences",
                     "bzs_emotion_tweets",
                     "bzs_emotion_emotional_events",
    "tweeteval_emotion",
    "tweeteval_hate",
    "tweeteval_irony",
    "tweeteval_offensive",
    "tweeteval_sentiment",
    "tweeteval_stance_abortion",
    "tweeteval_stance_atheism",
    "tweeteval_stance_climate",
    "tweeteval_stance_feminist",
    "tweeteval_stance_hillary",
    ]
    test_dataset = [(
    # eval_datasets += [ (
        CPGDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir, task_name=test_task_name)
        if training_args.do_eval
        else None ) for test_task_name in test_datasets]

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)

            def simple_accuracy(preds, labels):

                if isinstance(labels[0], list):
                    correct_counter = 0.0
                    for pred, label in zip(preds, labels):
                        if int(pred) in label:
                            correct_counter += 1
                    return correct_counter / len(labels)

                return (preds == labels).mean()

            return {"acc": simple_accuracy(preds, p.label_ids)}

        return compute_metrics_fn


    noneoftheabove= tokenizer(
            [tokenizer.sep_token + "None of the above" + tokenizer.sep_token],
            max_length=cpg_seq_lengths[hp_dict['dataset']]['train'][1],
            padding="max_length",
            truncation=True,
            add_special_tokens=False
        )

    dc = CPGCollator(max_label_length=cpg_seq_lengths[hp_dict['dataset']]['train'][1],
                        max_length=cpg_seq_lengths[hp_dict['dataset']]['train'][0],
                     noneoftheabove=noneoftheabove,
                     hp_dict=hp_dict
                     )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(),
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
        data_collator=dc.default_data_collator,
        data_collator_class=dc,
        hp_dict=hp_dict
    )

    # for parameters in model.base_model.parameters():
    #     parameters.requires_grad = False
    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # trainer.save_model()
        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        # eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn()
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

            with open(model_args.score_file, 'a') as f:
                f.write(str(eval_result['eval_acc']) + '\t')

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]


        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
