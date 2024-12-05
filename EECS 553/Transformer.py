import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum sequence length the model can handle
    n_layer: int = 12  # Number of transformer blocks (depth of the model)
    n_head: int = 12  # Number of attention heads in each block
    n_embd: int = 768  # Embedding size (dimensionality of token representations)
    n_feature: int = 2  # Number of features in the input x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # C should be the same as config.n_embd
        qkv = self.c_attn(x)  # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=2)  # [B, T, C]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_size]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_size]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, n_head, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # [B, n_head, T, T] x [B, n_head, T, head_size] -> [B, n_head, T, head_size]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        y = self.c_proj(y)
        return y  # [B, T, C]


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # usually approximates none nowadays
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(config.n_feature, config.n_embd),  # token/feature embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),  # positional encoding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # # weight sharing scheme
        # self.transformer.wte.weight = self.lm_head.weight  # both (vocab_size, n_embd)

        # init parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            std = 0.02
            torch.nn.init.normal(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # default init will be used for layer norm

    def forward(self, x):
        B, T, n_feature = x.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)  # [T]
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(x)  # (B, T, n_embd)
        y = tok_emb + pos_emb

        for block in self.transformer.h:
            y = block(y)
        y = self.transformer.ln_f(y)
        return y
