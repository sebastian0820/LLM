import tiktoken
import torch
import torch.nn as nn
from GPTModel import GPTModel
from LayerNorm import LayerNorm
from GELU import GELU
from generate_text_simple import generate_text_simple
from TransformerBlock import TransformerBlock
from print_gradients import print_gradients

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters : {total_params:,}")
print("Token embedding layer shape : ", model.tok_emb.weight.shape)
print("Output layer shape : ", model.out_head.weight.shape)
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying : {total_params_gpt2:,}")
total_size_bytes = total_params * 4 #A
total_size_mb = total_size_bytes / (1024 * 1024) #B
print(f"Total size of the model : {total_size_mb:.2f} MB")
print()

start_context = "Hello, I am a"
print("Input : ", start_context)
encoded = tokenizer.encode(start_context)
print("encoded : ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
print("encoded_tensor : ", encoded_tensor)
print("encoded_tensor.shape : ", encoded_tensor.shape)
print()
# logits = model(encoded_tensor)
# print("logits shape : ", logits.shape)
# print("logits : ", logits)

model.eval() #A
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output : ", out)
print("Output length : ", len(out[0]))
print()

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

