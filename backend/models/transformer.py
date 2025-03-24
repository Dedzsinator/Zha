import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TransformerModel:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    def to_device(self, device):
        self.model.to(device)
    
    def generate(self, input_text, max_length=200):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0])