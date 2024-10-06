import math
import torch

from torch import nn
from safetensors import safe_open
from trl import AutoModelForCausalLMWithValueHead


class ValueHeadRM(nn.Module):
    def __init__(self, model_path, head_path, **kwagrs):
        super().__init__()

        dtype = kwagrs.get('torch_dtype', torch.float16)
        device = kwagrs.get('device_map', 'cuda')

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, torch_dtype=dtype)
        self.config = self.model.config
        head_tensors = {}
        with safe_open(head_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                head_tensors[key] = f.get_tensor(key).to(dtype)

        v_head_tensors = {key[key.find('.') + 1:]: head_tensors[key] for key in head_tensors.keys()}
        self.model.v_head.load_state_dict(v_head_tensors)
        print('finish loading value head.')
        self.model = self.model.to("cuda")

    def forward(
        self,
        **inputs
    ):
        return self.model(**inputs)


class ValueHeadRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
    
    def __call__(self, samples, **kwargs):
        batch_size = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        encoding_dict = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        input_ids = encoding_dict["input_ids"]
        attention_masks = encoding_dict["attention_mask"]
        out = []
        with torch.no_grad():
            for i in range(math.ceil(len(samples) / batch_size)):
                inputs = {
                    "input_ids": input_ids[i * batch_size : (i + 1) * batch_size],
                    "attention_mask": attention_masks[i * batch_size : (i + 1) * batch_size]
                }
                _, _, rewards = self.model(**inputs)
                masks = inputs["attention_mask"]
                scores = rewards.gather(dim=-1, index=(masks.sum(dim=-1, keepdim=True) - 1))
                if len(scores) > 1:
                    scores = scores.squeeze()
                # print(scores)
                out.extend(scores)

                # batch_size = inputs["input_ids"].size(0) // 2
                # chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
                # chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)
                # chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
                # rejected_scores = rejected_rewards.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
                # chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()
                # # if scores are dict (for Yi model), extract them from tensor.
                # if isinstance(rewards, dict):
                #     rewards = rewards["scores"]
                # out.extend(rewards)

        return torch.hstack(out)


def build_value_head_rm(model_path: str, head_path: str, **kwargs) -> ValueHeadRM:
    model = ValueHeadRM(model_path, head_path, **kwargs)
    return model