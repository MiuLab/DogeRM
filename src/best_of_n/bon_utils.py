import re
import sys
import json

import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict, Counter
from typing import List, Dict
from transformers import TextClassificationPipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


MAX_INT = sys.maxsize


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(JSONEncoder, self).default(obj)


class BoNDataset(Dataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        best_of, 
        size=MAX_INT, 
        system_prompt="",
        prompt_key='prompt', 
        output_key='answer'
    ):
        self.data = list(data.values())[:size]
        self.tokenizer = tokenizer
        self.best_of = best_of
        self.preprocess(system_prompt, prompt_key, output_key)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]["query"]

    def preprocess(self, system_prompt, prompt_key, output_key):
        for i in range(len(self.data)):
            queries = []
            for question, ans in zip(self.data[i][prompt_key][:self.best_of], self.data[i][output_key][:self.best_of]):
                # query = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response: Let's think step by step. {ans}"
                answer = " Let's think step by step." + ans
                msg = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer} # Use COT when prompting LM
                ]
                if system_prompt != "":
                    msg = [{"role": "system", "content": system_prompt}] + msg
                query = self.tokenizer.apply_chat_template(msg, tokenize=False)
                queries.append(query)

            self.data[i]["query"] = queries


class HumanEvalDataset(Dataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        best_of, 
        size=MAX_INT,
        system_prompt="",
        output_key='outputs',
        completion_only=False
    ):
        self.data = list(data.values())[:size]
        self.tokenizer = tokenizer
        self._preprocess(tokenizer, best_of, system_prompt, output_key, completion_only)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['rm_query']

    def _preprocess(self, tokenizer, best_of, system_prompt, output_key, completion_only):
        for i in range(len(self.data)):
            instructions = self.data[i]['instruction'][:best_of]
            prompts = self.data[i]['prompt'][:best_of]
            responses = self.data[i][output_key][:best_of]
            rm_queries = []
            for inst, prompt, response in zip(instructions, prompts, responses):
                msg = []
                if system_prompt:
                    msg.append({'role': 'system', 'content': system_prompt})
                msg.append({'role': 'user', 'content': inst})
                request = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

                if completion_only:
                    # truncate to the end of the first code block (```python\n{code}\n```)
                    end_str = '\n```'
                    response = response[:response.find(end_str) + len(end_str)]   
                 
                rm_query = request + f" ```python\n{prompt}" + response
                rm_queries.append(rm_query)
            
            self.data[i]['rm_query'] = rm_queries


class HumanEvalProcessor():
    # reference 1: https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
    """
    The MIT License

    Copyright (c) OpenAI (https://openai.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """

    language_map = {
        "python": "Python",
        "cpp": "cpp",
        "js": "JavaScript",
        "java": "Java",
        "go": "Go",
        "rust": "Rust"
    }

    @classmethod
    def prepare_example(self, example, tokenizer, system_prompt=""):
        msg = []
        if system_prompt:
            msg.append({'role': 'system', 'content': system_prompt})
        msg.append({'role': 'user', 'content': example['instruction']})

        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        generation_prefix = f" ```python\n{example['prompt']}"
        
        example['query'] = prompt + generation_prefix

        return example


class MbppDataset(Dataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        best_of, 
        size=MAX_INT,
        system_prompt="",
        output_key='outputs'
    ):
        self.data = list(data.values())[:size]
        self.tokenizer = tokenizer
        self._preprocess(tokenizer, best_of, system_prompt, output_key)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['rm_query']

    def _preprocess(self, tokenizer, best_of, system_prompt, output_key):
        for i in range(len(self.data)):
            self.data[i]['rm_query'] = []
            for j in range(best_of):
                description = self.data[i]['text'][j]
                test_case = self.data[i]['test_list'][j][0]
                docstring = f'"""\n{description}\n{test_case}\n"""'

                msg = []
                if system_prompt:
                    msg.append({'role': 'system', 'content': system_prompt})
                msg.append({'role': 'user', 'content': docstring})
                msg.append({'role': 'assistant', 'content': self.data[i][output_key][j]})

                rm_query = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                self.data[i]['rm_query'].append(rm_query)


class MbppProcessor():
    @classmethod
    def prepare_example(
        self, 
        example, 
        tokenizer, 
        system_prompt="", # the system prompt
    ):
        msg = []
        if system_prompt:
            msg.append({'role': 'system', 'content': system_prompt})
        
        description = example['text']
        test_case = example['test_list'][0]
        docstring = f'"""\n{description}\n{test_case}\n"""'

        msg.append({'role': 'user', 'content': docstring})

        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        # generation_prefix = f"```python\n{example['prompt']}"
        
        example['query'] = prompt

        return example


class EurusPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        # `_legacy` is used to determine if we're running the naked pipeline and in backward
        # compatibility mode, or if running the pipeline with `pipeline(..., top_k=1)` we're running
        # the more natural result containing the list.
        # Default value before `set_parameters`
        outputs = model_outputs[0]
        outputs = outputs.numpy()

        scores = outputs

        if top_k == 1 and _legacy:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}

        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]
        if not _legacy:
            dict_scores.sort(key=lambda x: x["score"], reverse=True)
            if top_k is not None:
                dict_scores = dict_scores[:top_k]
        return dict_scores


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


# For Self-consistency
def most_frequent_element(lst):
    if not lst:
        return None  # Handle empty list case
    counter = Counter(lst)
    most_common_element, count = counter.most_common(1)[0]
    return most_common_element


# For GSM8K
# Borrow from https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=ReheKSODEXiq
def find_numbers(x: str) -> list[str]:
  """Finds all numbers in a string."""
  # Search for number, possibly negative (hyphen), with thousand separators
  # (comma), and with a decimal point (period inbetween digits).
  numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
  ).findall(x)
  return numbers


def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
  """Finds the most relevant number in a string."""
  # If model uses the answer delimiter, then select the first number following
  # that format.
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  # In general, select the last number in the string.
  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ''


def maybe_remove_comma(x: str) -> str:
  # Example: 5,600 -> 5600
  return x.replace(',', '')
