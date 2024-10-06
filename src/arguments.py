from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class DatasetArguments:
    """ Dataset arguments for train_rm.py """
    dataset_name: str
    dataset_config_path: str
    dataset_shuffle_seed: int = None
    chat_template: str = "default"
    
    hp_sweep: bool = False
    train_split: str = "train"
    test_split: str = None
    data_dir: str = None

    # Whether to run preprocess or not.
    run_preprocess: bool = True
    preprocess_func_args: Dict = field(default_factory=dict)
    num_workers: int = 1

    test_size: float = 0

@dataclass
class SFTDataArguments:
    dataset_path: str = field(
        default="tatsu-lab/alpaca_farm",
        metadata={
            "help": "Path to the dataset. Either points to a location on Hugging Face hub or a local folder. "
            "If the path points to a local folder, the folder must be structured properly "
            "(see documentation for datasets.load_dataset)."
        },
    )
    data_dir: Optional[str] = field(
        default="alpaca_instructions",
        metadata={"help": "Name of the dataset to load -- the argument `name` passed to `datasets.load_dataset`."},
    )
    train_split: str = field(
        default="sft",
        metadata={"help": "Splits to use for training. This must not be an empty list."},
    )
    eval_split: Optional[str] = field(
        default="val",
        metadata={
            "help": "Splits to use for evaluation. "
            "If None, empty, or the splits are not found in the dataset, no evaluation is performed."
        },
    )
    additional_special_tokens: Optional[List[str]] = field(
        default_factory=lambda: ["<|prompter|>", "<|assistant|>"],
        metadata={"help": "Special tokens to be added to tokenizers. e.g. chat control tokens."}
    )
    inst_key: Optional[str] = field(
        default="instruction",
        metadata={"help": "Feature contains the instruction."}
    )
    input_key: Optional[str] = field(
        default="input",
        metadata={"help": "Feature contains additional input."}
    )
    output_key: Optional[str] = field(
        default="output",
        metadata={"help": "Feature contains the expected output. This feature is used for calculating LM loss."}
    )
