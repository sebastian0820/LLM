import torch  
import torch.nn as nn  

class CausalAttention(nn.Module):  
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):  
        super().__init__()  
        self.d_out = d_out  
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.dropout = nn.Dropout(dropout)  
        self.register_buffer(  
            'mask',  
            torch.triu(torch.ones(context_length, context_length), diagonal=1)  
        )  

    def forward(self, x):  
        b, num_tokens, d_in = x.shape  # Correctly unpack the shape  
        keys = self.W_key(x)           # Correctly compute keys  
        queries = self.W_query(x)      # Keep queries as is  
        values = self.W_value(x)  

        attn_scores = queries @ keys.transpose(1, 2)  
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)  
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  
        attn_weights = self.dropout(attn_weights)  
        context_vec = attn_weights @ values  
        return context_vec
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) #C
        queries = self.W_query(x) #C
        values = self.W_value(x) #C
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)#D
        keys = keys.transpose(1, 2) #E
        queries = queries.transpose(1, 2) #E
        values = values.transpose(1, 2) #E
        attn_scores = queries @ keys.transpose(2, 3) #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) #I
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #K
        return context_vec