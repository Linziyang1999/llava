import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
# 加载模型
model_path = "../modelfile/llava-v1.6-mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
config = AutoConfig.from_pretrained(model_path)
input = "how can I login?"
input_ids = tokenizer(input, return_tensors="pt").input_ids
print(input_ids[0].shape)