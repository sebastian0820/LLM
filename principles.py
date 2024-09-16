import re
from SimpleTokenizerV2 import SimpleTokenizerV2
import tiktoken
import torch
import torch.nn as nn
from GPTModel import GPTModel
from LayerNorm import LayerNorm
from GELU import GELU
from generate_text_simple import generate_text_simple
from TransformerBlock import TransformerBlock
from print_gradients import print_gradients

with open("short story.txt", "r", encoding="utf-8") as f : raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>", "<|BOS|>", "<|EOS|>", "<|PAD|>"])
vocab_size = len(all_words)
# print(f"Vocabulary size: {vocab_size}")

GPT_CONFIG_124M = {
    "vocab_size": vocab_size, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
# torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
vocab = {token:integer for integer , token in enumerate(all_words)}
tokenizer = SimpleTokenizerV2(vocab)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
print("Token embedding layer shape: ", model.tok_emb.weight.shape)
print("Output layer shape: ", model.out_head.weight.shape)
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
total_size_bytes = total_params * 4 #A
total_size_mb = total_size_bytes / (1024 * 1024) #B
print(f"Total size of the model: {total_size_mb:.2f} MB")
print()

text = "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it"
encoded = tokenizer.encode(text)
tensor = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
model.eval()
out = generate_text_simple(
    model=model,
    idx=tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)