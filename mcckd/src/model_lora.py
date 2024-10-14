import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
)

from src.model import Attention, ModelArgs, apply_rotary_emb, FeedForward, TransformerBlock, LLaMA


@dataclass
class LoraModelArgs(ModelArgs):
    r: int = None  # Rank of lora


class LoraAttention(Attention):
    def __init__(self, args: LoraModelArgs):
        super().__init__(args)

        self.lora_a_wq = RowParallelLinear(
            args.dim,
            args.r,
            bias=False,
            input_is_parallel=False,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_wq = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wk = RowParallelLinear(
            args.dim,
            args.r,
            bias=False,
            input_is_parallel=False,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_wk = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wv = RowParallelLinear(
            args.dim,
            args.r,
            bias=False,
            input_is_parallel=False,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_wv = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_wo = ColumnParallelLinear(
            args.r,
            args.dim,
            bias=False,
            gather_output=True,
            init_method=init.zeros_,
        ).float()

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                use_cache=False):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x) + self.lora_b_wq(self.lora_a_wq(x.float())).to(x.dtype)
        xk = self.wk(x) + self.lora_b_wk(self.lora_a_wk(x.float())).to(x.dtype)
        xv = self.wv(x) + self.lora_b_wv(self.lora_a_wv(x.float())).to(x.dtype)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output) + self.lora_b_wo(self.lora_a_wo(output.float())).to(output.dtype)


class LoraFeedForward(FeedForward):
    def __init__(self, r: int, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__(dim, hidden_dim, multiple_of)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.lora_a_w1 = RowParallelLinear(
            dim,
            r,
            bias=False,
            input_is_parallel=False,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_w1 = ColumnParallelLinear(
            r,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_w2 = RowParallelLinear(
            hidden_dim,
            r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_w2 = ColumnParallelLinear(
            r,
            dim,
            bias=False,
            gather_output=True,
            init_method=init.zeros_,
        ).float()
        self.lora_a_w3 = RowParallelLinear(
            dim,
            r,
            bias=False,
            input_is_parallel=False,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_w3 = ColumnParallelLinear(
            r,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()

    def forward(self, x):
        w1_x = self.w1(x) + self.lora_b_w1(self.lora_a_w1(x.float())).to(x.dtype)
        w3_x = self.w3(x) + self.lora_b_w3(self.lora_a_w3(x.float())).to(x.dtype)
        out = F.silu(w1_x) * w3_x
        return self.w2(out) + self.lora_b_w2(self.lora_a_w2(out.float())).to(out.dtype)


class LoraTransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, args: LoraModelArgs):
        super().__init__(layer_id, args)
        self.attention = LoraAttention(args)
        self.feed_forward = LoraFeedForward(
            args.r, dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)


class LoraLLaMA(LLaMA):
    def __init__(self, params: LoraModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(LoraTransformerBlock(layer_id, params))

        self.lora_a_output = ColumnParallelLinear(
            params.dim,
            params.r,
            bias=False,
            gather_output=False,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_output = RowParallelLinear(
            params.r,
            params.vocab_size,
            bias=False,
            input_is_parallel=True,
            init_method=init.zeros_
        ).float()

        # Freeze parameters
        self._freeze()

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h)

        output = self.output(h) + self.lora_b_output(self.lora_a_output(h.float())).to(h.dtype)
        return output.float()

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)

    def load(self, ckpt_dir: str):
        super().load(ckpt_dir)
        self._freeze()
