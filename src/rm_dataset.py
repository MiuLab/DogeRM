from transformers import AutoTokenizer
from datasets import load_dataset
from utils import (
    default_func,
    preprocess_shp,
    preprocess_hh,
    preprocess_metamath,
    preprocess_distilabel_math,
    preprocess_alpaca,
    preprocess_pku_alignment,
    preprocess_summarization,
    preprocess_ultrafeedback,
    preprocess_ultrafeedback_alpaca,
    preprocess_autoj
)

PREPROCESS_FUNC = {
    "default": default_func,
    "stanfordnlp/SHP-2": preprocess_shp,
    "Anthropic/hh-rlhf": preprocess_hh,
    "abacusai/MetaMath_DPO_FewShot": preprocess_metamath,
    "argilla/distilabel-math-preference-dpo": preprocess_distilabel_math,
    "tatsu-lab/alpaca_farm": preprocess_alpaca,
    "PKU-Alignment/PKU-SafeRLHF-30K": preprocess_pku_alignment,
    "HuggingFaceH4/summarize_from_feedback": preprocess_summarization,
    "openai/summarize_from_feedback": preprocess_summarization,
    "HuggingFaceH4/ultrafeedback_binarized": preprocess_ultrafeedback,
    "argilla/ultrafeedback-binarized-preferences-cleaned": preprocess_ultrafeedback,
    # "argilla/ultrafeedback-binarized-preferences-cleaned": preprocess_ultrafeedback_alpaca,
    "/work/hank0316/auto-j/data/test/testdata_pairwise.jsonl": preprocess_autoj
}

class DatasetProcessor:
    def __init__(self, config: dict, max_len: int, tokenizer: AutoTokenizer, split: str):
        self.config = config
        self.split = split
        # Load dataset
        load_args = {"split": self.split, "verification_mode": "no_checks"}
        if self.config.get('data_dir', False):
            load_args['data_dir'] = self.config['data_dir']
        if self.config.get('name', False):
            load_args['name'] = self.config['name']
        if self.config.get('axis', False):
            load_args['axis'] = self.config['axis']
        
        self.dataset = load_dataset(self.config["dataset_name"], **load_args)
        
        # self.dataset = load_dataset("json", data_files="/work/hank0316/auto-j/data/test/testdata_pairwise.jsonl", **load_args)

        if self.config.get('dataset_shuffle_seed', False):
            self.dataset = self.dataset.shuffle(seed=self.config['dataset_shuffle_seed'])

        self.max_len = max_len
        self.tokenizer = tokenizer

        if PREPROCESS_FUNC.get(self.config['dataset_name'], False):
            self.dataset_preprocess_func = PREPROCESS_FUNC[self.config['dataset_name']]
        else:
            self.dataset_preprocess_func = PREPROCESS_FUNC['default']

        # preprocess the raw dataset
        if self.config.get("run_preprocess", False):
            # preprocess the dataset
            self.dataset = self.dataset_preprocess_func(
                self.dataset,
                tokenizer=tokenizer,
                **self.config['preprocess_func_args'],
            )

            # tokenization & filtration
            self.dataset = self.dataset.map(
                lambda x: self.preprocess(x),
                batched=True,
                num_proc=self.config["num_workers"],
            )
            self.dataset = self.dataset.filter(
                lambda x: len(x["input_ids_chosen"]) <= self.max_len
                and len(x["input_ids_rejected"]) <= self.max_len
                and x["input_ids_chosen"] != x["input_ids_rejected"] # There might be some case that chosen = rejected, which have to be removed
            )

    def preprocess(self, examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = self.tokenizer(chosen)
            tokenized_rejected = self.tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples
    
    def get(self):
        return self.dataset
