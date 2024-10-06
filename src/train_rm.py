# The code is modified from https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
import logging
import warnings
import torch
import os
import sys
import yaml
import random
import datasets
import transformers

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config

from utils import set_seed, DEFAULT_CHAT_TEMPLATE
from arguments import DatasetArguments
from rm_dataset import DatasetProcessor


tqdm.pandas()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Arguments
    parser = HfArgumentParser((RewardConfig, ModelConfig, DatasetArguments))
    reward_config, model_config, dataset_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    if reward_config.report_to == "":
        reward_config.report_to = []

    # set random seed to ensure model initialization
    set_seed(reward_config.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if reward_config.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = reward_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {reward_config.local_rank}, device: {reward_config.device}, n_gpu: {reward_config.n_gpu}"
        + f"distributed training: {reward_config.parallel_mode.value == 'distributed'},"
        + f" 16-bits training: {reward_config.fp16}"
    )
    logger.info(f"Training parameters {reward_config}")

    # Update data arguments
    dataset_config = dataset_config.__dict__
    with open(dataset_config["dataset_config_path"], 'r') as f:
        dataset_cf = yaml.safe_load(f)[dataset_config["dataset_name"]]
        print(dataset_cf)
        if dataset_cf:
            dataset_config.update(**dataset_cf)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        use_fast=True,
        truncation=False,
        model_max_length=reward_config.max_length,
    )
    # set chat template (for llama2)
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    tokenizer.padding_side = "right"
    print(f'tokenizer: {tokenizer}')

    # Load Model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    # quantization
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )

    # Add pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )
    
    # Datasets
    train_dataset_processor = DatasetProcessor(dataset_config, reward_config.max_length, tokenizer, dataset_config['train_split'])
    train_dataset = train_dataset_processor.get()
    if dataset_config['test_split']:
        eval_dataset_processor = DatasetProcessor(dataset_config, reward_config.max_length, tokenizer, dataset_config['test_split']) if dataset_config['test_split'] else None
        eval_dataset = eval_dataset_processor.get()
    elif dataset_config['test_size'] > 0:
        print("split datasets")
        all_datasets = train_dataset.train_test_split(test_size=dataset_config['test_size'])
        train_dataset = all_datasets['train']
        eval_dataset = all_datasets['test']
    else:
        eval_dataset = None

    # Random sample data and print
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the processed training set, chosen:\n\n{train_dataset[index]['chosen']}")
        print(f"Sample {index} of the processed training set, rejected:\n\n{train_dataset[index]['rejected']}")

    print(f'training_set: {train_dataset}')
    print(f'evaluation_set: {eval_dataset}')

    # Training
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )

    # Start training
    trainer.train()
    trainer.save_model(reward_config.output_dir)
