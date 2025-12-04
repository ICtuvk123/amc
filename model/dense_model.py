import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import math
from math import pi, log
from einops import rearrange, repeat
from model.model_util import Modality
from model.rotary import apply_rotary_emb

from typing import Iterable, Dict, List
from scattermoe.scattermoe.mlp import MLP as MoEMLP
from scattermoe.scattermoe.mlp import GLUMLP

import numpy as np

def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x

def contrastive_loss(modality_1, modality_2, temperature=0.5):
    if type(temperature) is torch.Tensor:
            with torch.no_grad(): 
                temperature.clamp_(0.001, 0.5)
    # Modality: Batch x Feature Dimension
    modality_1 = F.normalize(modality_1, dim=-1)
    modality_2 = F.normalize(modality_2, dim=-1)
    pos_idx = torch.arange(len(modality_1)).view(-1, 1).to(modality_1.device)
    pos_idx = torch.eq(pos_idx, pos_idx.t()).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    sim_1to2 = modality_1 @ modality_2.T / temperature
    sim_2to1 = modality_2 @ modality_1.T / temperature

    loss = -torch.sum(F.log_softmax(sim_1to2, dim=1) * sim_targets, dim=1).mean() - torch.sum(F.log_softmax(sim_2to1, dim=1) * sim_targets, dim=1).mean()

    return loss / 2

def contrastive_lossv2(modality_1, modality_2, temperature=0.5):
    if type(temperature) is torch.Tensor:
            with torch.no_grad(): 
                temperature.clamp_(0.001, 0.5)
    # Modality: Batch x Feature Dimension
    feature_dim = modality_1.shape[-1]
    modality_1 = F.normalize(modality_1, dim=-1).reshape(-1, feature_dim)
    modality_2 = F.normalize(modality_2, dim=-1).reshape(-1, feature_dim)
    
    m1_logits = math.exp(temperature) * modality_1 @ modality_2.t()
    m2_logits = m1_logits.t()
    
    n_logits = m1_logits.shape[0]
    labels = torch.arange(n_logits, device=m1_logits.device, dtype=torch.long)
    total_loss = (
            F.cross_entropy(m1_logits, labels) +
            F.cross_entropy(m2_logits, labels)
        ) / 2
    return total_loss

def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts = 8, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of tensors. Shape: [batch_size, seqeunce_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None:
        return 0

    if isinstance(gate_logits, tuple):
        # cat along the layers?
        gate_logits = torch.cat(gate_logits, dim=0)

    routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1)

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask:torch.Tensor = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)) * (num_experts**2)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_score = None

    def forward(self, x, context = None):
        B, N, C = x.shape
        if context is None:
            context = x
        # print(x.shape, context.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v  = kv.unbind(0) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_score = attn.mean(dim=1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class DiffAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qk2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.gp_norm = nn.GroupNorm(num_groups=self.num_heads, num_channels=dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_init = 0.1
        
        self.attn_score = None
        
        self.attn1_norm = nn.LayerNorm(dim)
        self.attn2_norm = nn.LayerNorm(dim)

    def forward(self, x, context = None):
        B, N, C = x.shape
        if context is None:
            context = x

        qk = self.qk(self.attn1_norm(context)).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k  = qk.unbind(0) 
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q2 = self.q2(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k2 = self.k2(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        qk2 = self.qk2(self.attn2_norm(context)).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2  = qk2.unbind(0) 

        attn1 = (q @ k.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        
        
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn = attn1 - lambda_full * attn2
        
        # print(attn.shape)
        # self.attn_score = attn.mean(dim=1).mean(dim=-2)
        self.attn_score = attn.mean(dim=1)#.max(dim=-2)
        # print(self.attn_score.shape)
        
        # attn = self.

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x * (1-self.lambda_init))
        x = self.proj_drop(x)
        
        return x
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
    
def get_rotary_embeddings(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len, dtype=torch.float)
    angles = torch.einsum("i,d->id", positions, inv_freq)
    sin, cos = angles.sin(), angles.cos()
    return sin, cos

class MultiheadDiffAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    # def __init__(
    #     self,
    #     args,
    #     embed_dim,
    #     num_heads,
    # ):
        super().__init__()
        # self.args = args
        self.embed_dim = dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        # self.head_dim
        
        self.head_dim = dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim // self.n_rep, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim // self.n_rep, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim // self.n_rep)

        self.lambda_init = 0.1
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self, x, context = None, attn_mask = None):
        bsz, tgt_len, embed_dim = x.size()
        if context is None:
            context = x
            
        src_len = tgt_len

        q = self.q_proj(context)
        k = self.k_proj(context)
        v = self.v_proj(x)
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        
        sin, cos = get_rotary_embeddings(tgt_len, self.head_dim//2)
        sin = sin.type_as(q)
        cos = cos.type_as(q)
        # print(q.shape, sin.shape, cos.shape, q.dtype, k.dtype, sin.dtype, x.dtype) # batch, seqlen, nheads, headdim

        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        
        # attn_weights = self.attn_drop(attn_weights)
        
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        self.attn_score = attn_weights.mean(dim=1)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        return attn

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class VanillaTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class DiffTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DiffAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DiffSparseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_experts = 8, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadDiffAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = SparseMoeBlock(
            dim, mlp_hidden_dim, num_experts, topk, act_layer
        )
        self.router_logits = []

    def calc_balance_loss(self):
        if len(self.router_logits) < 1:
            return
        blloss = load_balancing_loss_func(torch.cat(self.router_logits, dim=0), num_experts=self.num_experts, top_k=self.topk)
        self.router_logits = []
        return blloss

    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        mlp_x, router_logits = self.mlp(self.norm2(x))
        self.router_logits.append(router_logits)
        x = x + mlp_x
        return x

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, ox):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        x = x + self.attn(self.norm1(x), ox)
        x = x + self.mlp(self.norm2(x))
        return x

class TorchExpert(nn.Module):
    def __init__(self, input_size, hidden_size, activation):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.wi = nn.Linear(input_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, input_size, bias=False)
        
    
    def forward(self, x):
        h = self.activation(self.wi(x))
        y = self.wo(h)
        return y

class PureTorchSparseMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, top_k, activation):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.top_k = min(top_k, self.num_experts)
        
        self.experts = nn.ModuleDict()
        for idx in range(self.num_experts):
            self.experts[f"expert_{idx}"] = TorchExpert(input_size, hidden_size, activation)
        
    def forward(self, x, routing_weights, selected_experts):
        # batch_size, sequence_length, hidden_dim = x.shape
        expert_index_list = []
        for i in range(self.top_k):
            expert_index = torch.nn.functional.one_hot(selected_experts[:, :, i].squeeze(dim=-1), num_classes=self.num_experts)
            expert_index_list.append(expert_index)
        
        finall_hidden_states = None
        for i in range(len(expert_index_list)):
            router_mask = expert_index_list[i]
            router_probs = routing_weights[:,:,i].unsqueeze(-1)
            next_states = x.clone()
            for idx, expert in enumerate(self.experts.values()):
                token_indices = router_mask[:, :, idx].bool()
                next_states[token_indices] = expert(x[token_indices]).to(dtype=x.dtype)
            # print(routing_weights.shape, next_states.shape)
            if finall_hidden_states == None:
                finall_hidden_states = router_probs * next_states
            else:
                finall_hidden_states += router_probs * next_states
        return finall_hidden_states

class SparseMoeBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, act_layer):
        super().__init__()
        self.hidden_dim = hidden_size
        self.ffn_dim = intermediate_size
        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # self.moe_mlp = GLUMLP(
        #     input_size=self.hidden_dim,
        #     hidden_size=self.ffn_dim,
        #     num_experts=self.num_experts,
        #     top_k=self.top_k,
        #     activation=act_layer()
        # )
        
        self.moe_mlp = PureTorchSparseMLP(
            input_size=self.hidden_dim,
            hidden_size=self.ffn_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation=act_layer()
        )
        self.routing_weights = None

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # print(hidden_states.shape)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        self.routing_weights = routing_weights.reshape(batch_size, sequence_length, -1)
        # print(self.routing_weights.std(dim=-1).mean())
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = self.moe_mlp(hidden_states, routing_weights, selected_experts)
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits.reshape(-1, self.num_experts)

class SparseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_experts=16, topk=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_experts = num_experts
        self.topk = topk
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.gate = nn.Linear(dim, num_experts, bias=False)
        self.mlp = SparseMoeBlock(
            dim, mlp_hidden_dim, num_experts, topk, act_layer
        )
        self.router_logits = []

    def calc_balance_loss(self):
        if len(self.router_logits) < 1:
            return
        blloss = load_balancing_loss_func(torch.cat(self.router_logits, dim=0), num_experts=self.num_experts, top_k=self.topk)
        self.router_logits = []
        return blloss

    def forward(self, x):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        
        x = x + self.attn(self.norm1(x))
        mlp_x, router_logits = self.mlp(self.norm2(x))
        self.router_logits.append(router_logits)
        x = x + mlp_x
        return x

class SinglemodalTransformer(nn.Module):
    def __init__(self, device, modality: Modality,
                 hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modality = modality
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        self.latents = nn.Parameter(torch.randn(latent_length, hidden_size))
        self.latents_norm = nn.LayerNorm(self.hidden_size)
        
        # Calculate the actual input dimension after adding positional encodings
        # For 2D data, we have 2 coordinates, and each coordinate gets freq_bands*2 + 1 features from fourier encoding
        num_coords = 2  # For 2D image data
        pos_encoding_dim = num_coords * ((self.modality.freq_bands * 2) + 1)
        actual_input_dim = self.modality.input_dim + pos_encoding_dim

        self.input_norm = nn.LayerNorm(actual_input_dim)
        self.input_proj = nn.Linear(actual_input_dim, self.hidden_size)
        self.input_cross_attn = Attention(
            self.hidden_size, 
            n_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )
        self.n_blocks = n_blocks
        
        self.blocks = nn.ModuleList([
                VanillaTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
            ])
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim)]
        )
        
        
    def forward(self, data):
        data = data[list(data.keys())[0]]
        
        batch_size, *axis, _ = data.shape
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=self.device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(pos, self.modality.max_freq, self.modality.freq_bands, self.modality.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
        
        data = torch.cat([data, enc_pos], dim=-1)
        data = rearrange(data, 'b ... d -> b (...) d')
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)
        x = self.latents_norm(x)
        
        data = self.input_norm(data)
        data = self.input_proj(data)
        
        x = self.input_cross_attn(x, data)

        for i, layer in enumerate(self.blocks):
            x = layer(x)
        x = self.head(x.mean(dim=1))

        return x
    
class ScorePredictorv2(nn.Module):

    def __init__(self, n_modalities, embed_dim=384):
        super().__init__()
        self.n_modalities = n_modalities
        input_dim = embed_dim * n_modalities
        self.score_net =nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Linear(input_dim * 2, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, n_modalities),
                nn.Softmax(dim=-1)
            )

    def forward(self, x: torch.Tensor, modality_len):
        x = x.split(modality_len, dim=0)
        x = torch.cat(x, dim=-1)
        x = self.score_net(x)
        # print(x.shape)
        # x = torch.cat(x, dim=0)
        return x
    
class ScorePredictor(nn.Module):

    def __init__(self, n_modalities, embed_dim=384):
        super().__init__()
        self.n_modalities = n_modalities
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        ) for _ in range(n_modalities)])

    def forward(self, x: torch.Tensor, modality_len):
        x = x.split(modality_len, dim=0)
        x = [self.score_nets[i](x[i]) for i in range(self.n_modalities)]
        # x = torch.cat(x, dim=0)
        return x
    
class MultimodalTransformer(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        # self.max_freq = max_freq
        # self.freq_bands = freq_bands
        # self.freq_base = freq_base
        
        self.hidden_size = hidden_size
        # self.input_dim = input_dim
        self.n_heads = n_heads
        # print(self.input_dim)
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))
        # self.latents = nn.Parameter(torch.randn(latent_length, hidden_size))
        self.latents_dict = nn.ParameterDict(self.latents_dict)
        self.latents_norm = nn.LayerNorm(self.hidden_size)
        
        # self.input_projector = nn.ModuleDict()
        input_projector = {}
        for kk in self.modalities:
            # Calculate the actual input dimension after adding positional encodings
            # For 2D data, we have 2 coordinates, and each coordinate gets freq_bands*2 + 1 features from fourier encoding
            num_coords = 2  # For 2D image data
            pos_encoding_dim = num_coords * ((self.modalities[kk].freq_bands * 2) + 1)
            actual_input_dim = self.modalities[kk].input_dim + pos_encoding_dim

            input_projector[kk] = nn.ModuleDict(
                {
                    'input_norm': nn.LayerNorm(actual_input_dim),
                    'input_proj': nn.Linear(actual_input_dim, self.hidden_size)
                }
            )
        self.input_projector = nn.ModuleDict(input_projector)
        # self.input_norm = nn.LayerNorm(self.input_dim)
        # self.input_proj = nn.Linear(self.input_dim, self.hidden_size)
        self.input_cross_attn = Attention(
            self.hidden_size, 
            n_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )
        self.n_blocks = n_blocks
        
        self.blocks = nn.ModuleList([
                VanillaTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
            ])
        n_modalities = len(self.modalities)
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )
        
        
    def forward(self, all_modalities):
        modality_x = []
        modality_len = []
        for kk in all_modalities:
            data = all_modalities[kk]
            batch_size, *axis, _ = data.shape
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=self.device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.modalities[kk].max_freq, self.modalities[kk].freq_bands, self.modalities[kk].freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
            # print(data.shape, enc_pos.shape)
            # print(data)
            data = torch.cat([data, enc_pos], dim=-1)
            data = rearrange(data, 'b l d -> b l d')
            latents = self.latents_dict[kk]
            x = repeat(latents, 'n d -> b n d', b=batch_size)
            # print(data.shape)
            x = self.latents_norm(x)
            
            
            data = self.input_projector[kk]['input_norm'](data)
            data = self.input_projector[kk]['input_proj'](data)
        
            x = self.input_cross_attn(x, data)
            modality_x.append(x)
            modality_len.append(x.shape[1])
        modality_x = torch.cat(modality_x, dim=1)
        for i, layer in enumerate(self.blocks):
            modality_x = layer(modality_x)
        modality_x = modality_x.split(modality_len, dim=1)
        modality_x = [item.mean(dim=1) for item in modality_x]
        modality_x = torch.cat(modality_x, -1)
        x = self.head(modality_x)

        return x
    
class PatchEmbeddings(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(feature_size)
        kernel_size = 4
        
        self.projection = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            nn.Linear(patch_size, embed_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False)
        )

    def forward(self, x):

        # x = self.norm(x)
        # x = x.unsqueeze(1)
        # x = self.projection(x)

        x = self.norm(x)
        # x = x.unsqueeze(1)
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        x = self.projection(x)
        return x
    
class QFormerEncoder(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, act_layer=nn.GELU):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_patches, embed_dim))
        
        self.img_encoder = Custom3DCNN(embed_dim)
        
        self.img_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.input_projector = nn.Linear(16 * 16, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x:torch.Tensor):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        # print(x.shape)
        # x = x.view(B, -1, H)
        # x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        x = x.unsqueeze(dim=1)
        x = self.img_encoder(x)

        B, C, D = x.shape
        # print(x.shape)
        
        query = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)
        x = query + self.attn(self.norm1(query), x)
        x = x + self.mlp(self.norm2(x))
        return x

class MulTFusionAlign(nn.Module):
    def __init__(self, modalities, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norms = nn.ModuleDict(
            {kk: norm_layer(dim) for kk in modalities}
        )
        self.crossmodel_transformer = nn.ModuleDict(
            {kk: nn.ModuleList([CrossTransformerBlock(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer) for _ in range(3)]) for kk in modalities}
        )
        self.transformers = nn.ModuleDict(
            {kk: nn.Sequential(*[VanillaTransformerBlock(
                dim * (len(modalities)-1),
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer) for _ in range(1)]) for kk in modalities}
        )
        
        self.projector = nn.ModuleDict(
            {kk: nn.Linear(dim * (len(modalities)-1), dim) for kk in modalities}
        )

    def forward(self, modalities):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        m_out = {}
        for m in modalities:
            m_cross = []
            for mm in modalities:
                if m == mm:
                    continue
                m_feature = modalities[m]
                for layer in self.crossmodel_transformer[m]:
                    m_feature = layer(modalities[m], self.norms[mm](modalities[mm]))
                m_cross.append(
                    m_feature
                )
            m_out[m] = self.projector[m](self.transformers[m](torch.cat(m_cross, dim=-1)))
        return m_out


class TFFusionAlign(nn.Module):
    def __init__(self, modalities, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, n_layer = 4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norms = nn.ModuleDict(
            {kk: norm_layer(dim) for kk in modalities}
        )

        self.m_spc_transformer = nn.ModuleDict(
            {kk: nn.Sequential(*[VanillaTransformerBlock(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer) for _ in range(n_layer)]) for kk in modalities}
        )


    def forward(self, modalities):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        m_out = {}
        for m in modalities:
            # if m == 'image':
            #     m_out[m] = modalities[m]
            #     continue
            m_spc_in = self.norms[m](modalities[m])
            m_out[m] = self.m_spc_transformer[m](m_spc_in)
        return m_out

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)

        # Create a vector of positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create a vector of frequencies
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Add a batch dimension and register as a buffer
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [sequence_length, batch_size, embedding_dim]
        # Add positional encoding to input
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:x.size(1), :]# * 0.01
        return x

class Custom3DCNN(nn.Module):
    #Architecture provided by: End-To-End Alzheimer's Disease Diagnosis and Biomarker Identification
    def __init__(self, hidden_dim=128):
        super(Custom3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.dropout1 = nn.Dropout3d(0.2)


        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.dropout3 = nn.Dropout3d(0.2)

        # Flatten the output and add a fully connected layer to reduce to hidden_dim
        self.fc_norm = nn.LayerNorm(3 * 3 * 4)
        self.fc = nn.Linear(3 * 3 * 4, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))
        
        B, L, _, _, _ = x.shape
        x = x.view(B, L, -1)  # Flatten the output
        x = self.fc(x)  # Apply the fully connected layer
        
        return x

class MultimodalTransformerWF(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20,
                 mlp_ratio = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        # self.max_freq = max_freq
        # self.freq_bands = freq_bands
        # self.freq_base = freq_base

        self.hidden_size = hidden_size
        # self.input_dim = input_dim
        self.n_heads = n_heads
        # print(self.input_dim)
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))
        # self.latents = nn.Parameter(torch.randn(latent_length, hidden_size))
        self.latents_dict = nn.ParameterDict(self.latents_dict)
        self.latents_norm = nn.LayerNorm(self.hidden_size)

        # self.input_projector = nn.ModuleDict()
        input_projector = {}
        for kk in self.modalities:
            # Calculate the actual input dimension after adding positional encodings
            # For 2D data, we have 2 coordinates, and each coordinate gets freq_bands*2 + 1 features from fourier encoding
            num_coords = 2  # For 2D image data
            pos_encoding_dim = num_coords * ((self.modalities[kk].freq_bands * 2) + 1)
            actual_input_dim = self.modalities[kk].input_dim + pos_encoding_dim

            input_projector[kk] = nn.ModuleDict(
                {
                    'input_norm': nn.LayerNorm(actual_input_dim),
                    'input_proj': nn.Linear(actual_input_dim, self.hidden_size)
                }
            )
        self.input_projector = nn.ModuleDict(input_projector)
        # self.input_norm = nn.LayerNorm(self.input_dim)
        # self.input_proj = nn.Linear(self.input_dim, self.hidden_size)
        self.input_cross_attn = Attention(
            self.hidden_size,
            n_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList([
                VanillaTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
            ])
        n_modalities = len(self.modalities)
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim), nn.Sigmoid()]
        )
        
        
    def forward(self, all_modalities):
        modality_x = None
        modality_len = []
        for kk in all_modalities:
            data = all_modalities[kk]
            batch_size, *axis, _ = data.shape
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=self.device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.modalities[kk].max_freq, self.modalities[kk].freq_bands, self.modalities[kk].freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
            data = torch.cat([data, enc_pos], dim=-1)
            data = rearrange(data, 'b ... d -> b (...) d')
            latents = self.latents_dict[kk]
            x = repeat(latents, 'n d -> b n d', b=batch_size)
            # print(data.shape)
            x = self.latents_norm(x)
        
            data = self.input_projector[kk]['input_norm'](data)
            data = self.input_projector[kk]['input_proj'](data)

            x = self.input_cross_attn(x, data)
            if modality_x is None:
                modality_x = x * self.modalities[kk].modelity_weight
            else:
                modality_x = modality_x + x * self.modalities[kk].modelity_weight

            modality_len.append(x.shape[1])
        # modality_x = torch.cat(modality_x, dim=1)

        for i, layer in enumerate(self.blocks):
            modality_x = layer(modality_x)
        modality_x = modality_x.mean(dim=1)
        x = self.head(modality_x)

        return x
    
def token_exchange(x, mask, mask_threshold):
    ori_shape = x[0].shape
    x = [item.reshape(-1, ori_shape[-1]) for item in x]
    x0 = torch.zeros_like(x[0])
    mask = mask.reshape(-1)
    
    if (mask >= mask_threshold).sum() > 0:
        x0[mask >= mask_threshold,:] = x[0][mask >= mask_threshold,:]
    if (mask < mask_threshold).sum() > 0:
        x0[mask < mask_threshold,:] = x[1][mask < mask_threshold,:]
    x0 = x0.reshape(*ori_shape)
    # x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
    # x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
    return x0
    
def token_exchange_ratio(x, mask, mask_threshold):
    ori_shape = x[0].shape
    x = [item.reshape(-1, ori_shape[-1]) for item in x]
    x0 = torch.zeros_like(x[0], dtype=x[0].dtype)
    mask = mask.reshape(-1)
    
    mask_threshold_value = mask.topk(int(len(mask) * (1-mask_threshold)))[0][-1].to(dtype=x0.dtype)
    if (mask >= mask_threshold_value).sum() > 0:
        # print(x0.dtype, mask_threshold_value.dtype, x[0].dtype)
        x0[mask >= mask_threshold_value,:] = x[0][mask >= mask_threshold_value,:]
    if (mask < mask_threshold_value).sum() > 0:
        # print(x0.dtype, mask_threshold_value.dtype, x[0].dtype, x[1].dtype)
        x0[mask < mask_threshold_value,:] = x[1][mask < mask_threshold_value,:]
    x0 = x0.reshape(*ori_shape)
    # x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
    # x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
    return x0
    
def token_exchange_ratiov2(x, mask, mask_threshold):
    ori_shape = x[0].shape
    x = [item.reshape(-1, ori_shape[-1]) for item in x]
    x0 = torch.zeros_like(x[0], dtype=x[0].dtype)
    # print(mask_threshold)
    example_pos = (mask.shape[1] * mask_threshold).to(dtype=torch.int32)
    # print(example_pos.shape, mask_threshold.shape)
    example_pos[example_pos == mask.shape[1]] = mask.shape[1]-1
    # print(example_pos)
    # print(mask.shape)
    mask_threshold_value = mask[np.arange(mask.shape[0]), example_pos]
    # print(mask_threshold_value.shape)
    mask_threshold_value = mask_threshold_value.repeat(mask.shape[1], 1).t().reshape(-1)
    mask = mask.reshape(-1)
    # print(mask_threshold_value.shape, mask.shape)
    
    # mask_threshold_value = mask.topk(int(len(mask) * (1-mask_threshold)))[0][-1].to(dtype=x0.dtype)
    # print(mask, mask_threshold_value)
    if (mask >= mask_threshold_value).sum() > 0:
        # print(x0.dtype, mask_threshold_value.dtype, x[0].dtype)
        x0[mask >= mask_threshold_value,:] = x[0][mask >= mask_threshold_value,:]
    if (mask < mask_threshold_value).sum() > 0:
        # print(x0.dtype, mask_threshold_value.dtype, x[0].dtype, x[1].dtype)
        x0[mask < mask_threshold_value,:] = x[1][mask < mask_threshold_value,:]
    x0 = x0.reshape(*ori_shape)
    # x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
    # x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
    return x0
    
class MultimodalTransformerTokenF(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))

        self.latents_dict = nn.ParameterDict(self.latents_dict)
        self.latents_norm = nn.LayerNorm(self.hidden_size)

        input_projector = {}
        for kk in self.modalities:
            # Calculate the actual input dimension after adding positional encodings
            # For 2D data, we have 2 coordinates, and each coordinate gets freq_bands*2 + 1 features from fourier encoding
            num_coords = 2  # For 2D image data
            pos_encoding_dim = num_coords * ((self.modalities[kk].freq_bands * 2) + 1)
            actual_input_dim = self.modalities[kk].input_dim + pos_encoding_dim

            input_projector[kk] = nn.ModuleDict(
                {
                    'input_norm': nn.LayerNorm(actual_input_dim),
                    'input_proj': nn.Linear(actual_input_dim, self.hidden_size)
                }
            )
        self.input_projector = nn.ModuleDict(input_projector)

        self.input_cross_attn = Attention(
            self.hidden_size,
            n_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList([
                VanillaTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
            ])

        n_modalities = len(self.modalities)
        self.score_predictor = nn.ModuleList(
            [ScorePredictor(n_modalities, self.hidden_size) for i in range(n_blocks)]
        )

        # self.ratio_predictor = nn.Sequential(
        #     *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        # )
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )
        
        
    def forward(self, all_modalities):
        modality_x = []
        modality_len = []
        token_fusion_ratio = []
        for kk in all_modalities:
            data = all_modalities[kk]
            batch_size, *axis, _ = data.shape
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=self.device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.modalities[kk].max_freq, self.modalities[kk].freq_bands, self.modalities[kk].freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
        
            data = torch.cat([data, enc_pos], dim=-1)
            data = rearrange(data, 'b ... d -> b (...) d')
            latents = self.latents_dict[kk]
            x = repeat(latents, 'n d -> b n d', b=batch_size)
            # print(data.shape)
            x = self.latents_norm(x)
        
            data = self.input_projector[kk]['input_norm'](data)
            data = self.input_projector[kk]['input_proj'](data)
        
            x = self.input_cross_attn(x, data)
            modality_x.append(x)
            modality_len.append(x.shape[0])
            token_fusion_ratio.append(self.modalities[kk].token_fusion_topk_ratio)
            
        modality_x = torch.cat(modality_x, dim=0)
        for i, layer in enumerate(self.blocks):
            score = self.score_predictor[i](modality_x, modality_len)
            mask = [F.softmax(score_, dim=-1)[:, :, 0] for score_ in score]
            modality_x = layer(modality_x)
            
            modality_x = modality_x.split(modality_len, dim=0)
            modality_x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(modality_x, mask)]
            tmp_modality_x = []
            for s_id in range(len(all_modalities)):
                d_id = s_id + 1 if (s_id+1) < len(all_modalities) else 0
                tmp_modality_x.append(
                    token_exchange([modality_x[s_id], modality_x[d_id]], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                )
            modality_x = torch.cat(tmp_modality_x, dim=0)
            
        modality_x = modality_x.split(modality_len, dim=0)
        modality_x = [item.mean(dim=1) for item in modality_x]
        modality_x = torch.cat(modality_x, -1)
        x = self.head(modality_x)

        return x
    
    
class MultimodalTransformerUniTokenFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, fusion_type='circle', num_experts=16, topk=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))

        self.latents_dict = nn.ParameterDict(self.latents_dict)
        self.latents_norm = nn.LayerNorm(self.hidden_size)

        input_projector = {}
        for kk in self.modalities:
            # Calculate the actual input dimension after adding positional encodings
            # For 2D data, we have 2 coordinates, and each coordinate gets freq_bands*2 + 1 features from fourier encoding
            num_coords = 2  # For 2D image data
            pos_encoding_dim = num_coords * ((self.modalities[kk].freq_bands * 2) + 1)
            actual_input_dim = self.modalities[kk].input_dim + pos_encoding_dim

            input_projector[kk] = nn.ModuleDict(
                {
                    'input_norm': nn.LayerNorm(actual_input_dim),
                    'input_proj': nn.Linear(actual_input_dim, self.hidden_size)
                }
            )
        self.input_projector = nn.ModuleDict(input_projector)

        self.input_cross_attn = Attention(
            self.hidden_size,
            n_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList([
                SparseTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate, num_experts=num_experts, topk=topk) for i in range(n_blocks)
            ])

        n_modalities = len(self.modalities)
        self.score_predictor = nn.ModuleList(
            [ScorePredictorv2(n_modalities, self.hidden_size) for i in range(n_blocks)]
        )

        # self.ratio_predictor = nn.Sequential(
        #     *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        # )
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )
        
        modality_head = {}
        for kk in self.modalities:
            modality_head[kk] = nn.Sequential(
                *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim), nn.Sigmoid()]
            )
        self.modality_head = nn.ModuleDict(modality_head)
    
    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_loss(modality_x_feature[i].view(-1, feature_dim), modality_x_feature[j].view(-1, feature_dim)))
        
        return sum(loss_list)
        
    def forward(self, all_modalities):
        modality_x = []
        modality_len = []
        token_fusion_ratio = []
        for kk in all_modalities:
            # print(kk)
            data = all_modalities[kk]
            batch_size, *axis, _ = data.shape
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=self.device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.modalities[kk].max_freq, self.modalities[kk].freq_bands, self.modalities[kk].freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
        
            data = torch.cat([data, enc_pos], dim=-1)
            data = rearrange(data, 'b ... d -> b (...) d')
            latents = self.latents_dict[kk]
            x = repeat(latents, 'n d -> b n d', b=batch_size)
            # print(data.shape)
            x = self.latents_norm(x)
        
            data = self.input_projector[kk]['input_norm'](data)
            data = self.input_projector[kk]['input_proj'](data)
        
            x = self.input_cross_attn(x, data)
            modality_x.append(x)
            modality_len.append(x.shape[0])
            token_fusion_ratio.append(self.modalities[kk].token_fusion_topk_ratio)
            
        modality_x = torch.cat(modality_x, dim=0)
        modality_x_feature = modality_x.split(modality_len, dim=0)
        for i, layer in enumerate(self.blocks):
            score = self.score_predictor[i](modality_x, modality_len)
            # mask = [score_[:, :, 0] for score_ in score]
            mask = [score[:,:,i] for i in range(score.shape[-1])]
            modality_x = layer(modality_x)
            # print([(mm.min().cpu().item(),mm.max().cpu().item(), mm.std().cpu().item(), mm.mean().cpu().item()) for mm in mask])
            modality_x = modality_x.split(modality_len, dim=0)
            modality_x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(modality_x, mask)]
            if self.fusion_type == 'circle':
                tmp_modality_x = []
                for s_id in range(len(all_modalities)):
                    d_id = s_id + 1 if (s_id+1) < len(all_modalities) else 0
                    tmp_modality_x.append(
                        token_exchange([modality_x[s_id], modality_x[d_id]], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
                modality_x = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'max':
                tmp_modality_x = []
                tmp_modality_x = self.max_token_fusion(all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x)
                modality_x = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'mean':
                tmp_modality_x = []
                tmp_modality_x = self.mean_token_fusion(all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x)
                modality_x = torch.cat(tmp_modality_x, dim=0)
                    # exit()
        self.limoe_loss = self.calc_limoe_loss(modality_x_feature)
        modality_x = modality_x.split(modality_len, dim=0)
        modality_spc_x = {mm: modality_x[i] for i, mm in enumerate(all_modalities)}
        modality_x = [item.mean(dim=1) for item in modality_x]
        modality_x = torch.cat(modality_x, -1)
        x = self.head(modality_x)
        mspc_x = {}
        self.clear_router_cache()
        for mm in modality_spc_x:
            # print(modality_spc_x[mm].shape)
            mspc_x[mm] = self.modality_head[mm](modality_spc_x[mm].mean(dim=1))
        return x, mspc_x

    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss()
        return tmp_loss

    def clear_router_cache(self):
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                if len(mm.router_logits) > len(self.modalities):
                    mm.router_logits = []

    def max_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            modality_selected = tmp_mask.argmax(dim=-1)
            modality_selected = modality_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 64)
                    # print(modality_selected.shape, tmp_replace_modality.shape)
            tmp_replace_modality = torch.gather(tmp_replace_modality, 2, modality_selected).squeeze(2)
            tmp_modality_x.append(
                        token_exchange_ratio([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def mean_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratio([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    
class MultimodalTransformerUniTokenFusionV2(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, fusion_type='circle', num_experts=16, topk=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))

        self.latents_dict = nn.ParameterDict(self.latents_dict)
        # self.latents_norm = nn.LayerNorm(self.hidden_size)

        input_projector = {}
        for kk in self.modalities:
            # print(kk, self.modalities[kk].input_dim)
            input_projector[kk] = nn.ModuleDict(
                {
                    'patch_emb': PatchEmbeddings(self.modalities[kk].input_dim, self.modalities[kk].modality_latent_len,
                                                 self.hidden_size)
                }
            )
            self.pos_embedding = PositionalEncoding(
                self.hidden_size, self.modalities[kk].modality_latent_len
            )
        self.input_projector = nn.ModuleDict(input_projector)
        self.token_type_embeddings = nn.Embedding(len(self.modalities), self.hidden_size)
        self.mult_align = MulTFusionAlign(modalities, hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate)
        self.n_blocks = n_blocks
        
        self.blocks = nn.ModuleList([
                VanillaTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
            ])
        
        n_modalities = len(self.modalities)
        self.score_predictor = nn.ModuleList(
            [ScorePredictorv2(n_modalities, self.hidden_size) for i in range(n_blocks)]
        )
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )

        modality_head = {}
        for kk in self.modalities:
            modality_head[kk] = nn.Sequential(
                *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim), nn.Sigmoid()]
            )
        self.modality_head = nn.ModuleDict(modality_head)
    
    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_loss(modality_x_feature[i].mean(dim=-2), modality_x_feature[j].mean(dim=-2)))
        
        return sum(loss_list)
    
    def calc_limoe_loss_v2(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                # feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_lossv2(modality_x_feature[i], modality_x_feature[j]))
        
        return sum(loss_list)
        
    def forward(self, all_modalities):
        modality_x = {}
        modality_len = []
        token_fusion_ratio = []
        modality_list = [kk for kk in all_modalities]
        modality_list.sort()
        for i, kk in enumerate(modality_list):
            data = all_modalities[kk]
            # batch_size, *axis, _ = data.shape
            # print(kk, data.shape)
            x = self.input_projector[kk]['patch_emb'](data.squeeze())
            # x += self.token_type_embeddings(torch.tensor(i).to(x.device))
            x = self.pos_embedding(x)
            modality_x[kk] = x
        modality_x = self.mult_align(modality_x)
        
        token_fusion_input = []
        for kk in modality_list:
            x = modality_x[kk]
            token_fusion_input.append(x)
            modality_len.append(x.shape[0])
            token_fusion_ratio.append(self.modalities[kk].token_fusion_topk_ratio)
            
        token_fusion_input = torch.cat(token_fusion_input, dim=0)
        modality_x_feature = token_fusion_input.split(modality_len, dim=0)
        for i, layer in enumerate(self.blocks):
            score = self.score_predictor[i](token_fusion_input, modality_len)
            # mask = [score_[:, :, 0] for score_ in score]
            mask = [score[:,:,i] for i in range(score.shape[-1])]
            
            modality_score = score.mean(dim=1)
            # print(modality_score)
            modality_threshold = self.threshold_setting(modality_score)
            token_fusion_input = layer(token_fusion_input)
            # print([(mm.min().cpu().item(),mm.max().cpu().item(), mm.std().cpu().item(), mm.mean().cpu().item()) for mm in mask])
            token_fusion_input = token_fusion_input.split(modality_len, dim=0)
            # token_fusion_input = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(token_fusion_input, mask)]
            if self.fusion_type == 'circle':
                tmp_modality_x = []
                for s_id in range(len(modality_list)):
                    d_id = s_id + 1 if (s_id+1) < len(modality_list) else 0
                    tmp_modality_x.append(
                        token_exchange([token_fusion_input[s_id], token_fusion_input[d_id]], mask[s_id], mask_threshold=modality_threshold[s_id])
                    )
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'max':
                tmp_modality_x = []
                tmp_modality_x = self.max_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'mean':
                tmp_modality_x = []
                tmp_modality_x = self.mean_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'learnable':
                tmp_modality_x = []
                tmp_modality_x = self.learnable_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            else:
                token_fusion_input = torch.cat(token_fusion_input, dim=0)
                    # exit()
        token_fusion_input = token_fusion_input.split(modality_len, dim=0)
        self.limoe_loss = 0
        # self.limoe_loss = self.calc_limoe_loss_v2(token_fusion_input)
        
        modality_spc_x = {mm: token_fusion_input[i] for i, mm in enumerate(modality_list)}
        token_fusion_input = [item.mean(dim=1) for item in token_fusion_input]
        token_fusion_input = torch.cat(token_fusion_input, -1)
        x = self.head(token_fusion_input)
        mspc_x = {}
        self.clear_router_cache()
        for mm in modality_spc_x:
            # print(modality_spc_x[mm].shape)
            mspc_x[mm] = self.modality_head[mm](modality_spc_x[mm].mean(dim=1))
        return x, mspc_x

    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss()
        return tmp_loss

    def clear_router_cache(self):
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                if len(mm.router_logits) > len(self.modalities):
                    mm.router_logits = []

    def threshold_setting(self, modality_score: torch.Tensor, threshold_values=[0.1, 0.2, 0.3]):
        modality_rank = modality_score.argsort(dim=-1)
        modality_thresholds = []
        # print(modality_rank)
        for i in range(modality_rank.shape[-1]):
            mm_rank = modality_rank[:, i]
            tt = torch.zeros_like(mm_rank, dtype=torch.float32)
            for j in range(mm_rank.max()+1):
                # print(j, mm_rank==j, threshold_values[j])
                # print(mm_rank==j)
                tt[mm_rank==j] = threshold_values[j]
            # tt[:2]=0.3
            # print(tt)
            modality_thresholds.append(tt)
        # print(modality_thresholds)
        return modality_thresholds

    def max_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            modality_selected = tmp_mask.argmax(dim=-1)
            modality_selected = modality_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, modality_x[s_id].shape[-1])
                    # print(modality_selected.shape, tmp_replace_modality.shape)
            tmp_replace_modality = torch.gather(tmp_replace_modality, 2, modality_selected).squeeze(2)
            # print(tmp_replace_modality.shape, modality_x[s_id].shape)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def mean_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def learnable_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id, mod in enumerate(all_modalities):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            # tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            # tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], self.latents_dict[mod].repeat(modality_x[s_id].shape[0],1,1).to(dtype=modality_x[s_id].dtype)], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
def sigmoid_with_scale(normalized_data, scale=10):
    return 1 / (1 + torch.exp(-scale * (normalized_data - 0.5)))
    
class MultimodalTransformerUniTokenFusionV3(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, fusion_type='circle', num_experts=16, topk=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))

        self.latents_dict = nn.ParameterDict(self.latents_dict)
        # self.latents_norm = nn.LayerNorm(self.hidden_size)

        input_projector = {}
        for kk in self.modalities:
            # print(kk, self.modalities[kk].input_dim)
            input_projector[kk] = nn.ModuleDict(
                {
                    'patch_emb': PatchEmbeddings(self.modalities[kk].input_dim, self.modalities[kk].modality_latent_len,
                                                 self.hidden_size)
                }
            )
            self.pos_embedding = PositionalEncoding(
                self.hidden_size, self.modalities[kk].modality_latent_len
            )
        self.input_projector = nn.ModuleDict(input_projector)
        self.token_type_embeddings = nn.Embedding(len(self.modalities), self.hidden_size)
        self.mult_align = MulTFusionAlign(modalities, hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate)
        self.n_blocks = n_blocks
        
        # self.blocks = nn.ModuleList([
        #         DiffTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
        #     ])
        
        self.blocks = nn.ModuleList([
                DiffSparseTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate, num_experts=num_experts, topk=topk) for i in range(n_blocks)
            ])
        
        # self.blocks = nn.ModuleList([
        #         DiffTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
        #     ])
        
        n_modalities = len(self.modalities)
        self.score_predictor = nn.ModuleList(
            [ScorePredictorv2(n_modalities, self.hidden_size) for i in range(n_blocks)]
        )
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )

        modality_head = {}
        for kk in self.modalities:
            modality_head[kk] = nn.Sequential(
                *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim), nn.Sigmoid()]
            )
        self.modality_head = nn.ModuleDict(modality_head)
    
    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_loss(modality_x_feature[i].mean(dim=-2), modality_x_feature[j].mean(dim=-2)))
        
        return sum(loss_list)
    
    def calc_limoe_loss_v2(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                # feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_lossv2(modality_x_feature[i], modality_x_feature[j]))
        
        return sum(loss_list)
        
    def forward(self, all_modalities):
        modality_x = {}
        modality_len = []
        token_fusion_ratio = []
        modality_list = [kk for kk in all_modalities]
        modality_list.sort()
        for i, kk in enumerate(modality_list):
            data = all_modalities[kk]
            # Reshape data from [batch, 16, 16, feature_dim] to [batch*16*16, feature_dim]
            original_shape = data.shape
            reshaped_data = data.view(-1, data.shape[-1])

            x = self.input_projector[kk]['patch_emb'](reshaped_data)
            # x now has shape [batch*16*16, num_patches, embed_dim] where num_patches=modality_latent_len=16
            # Reshape to [batch, 16*16, num_patches, embed_dim]
            x = x.view(original_shape[0], -1, x.shape[1], x.shape[2])
            # Now we have [batch, 16*16, 16, embed_dim], need to aggregate spatial dimension
            # Average pooling across the spatial dimension (16*16) to get [batch, 16, embed_dim]
            x = x.mean(dim=1)

            x = self.pos_embedding(x)
            modality_x[kk] = x
        modality_x = self.mult_align(modality_x)

        token_fusion_input = []
        for kk in modality_list:
            x = modality_x[kk]
            token_fusion_input.append(x)
            modality_len.append(x.shape[0])
            token_fusion_ratio.append(self.modalities[kk].token_fusion_topk_ratio)

        token_fusion_input = torch.cat(token_fusion_input, dim=0)
        modality_x_feature = token_fusion_input.split(modality_len, dim=0)


        for i, layer in enumerate(self.blocks):
            
            token_fusion_input = layer(token_fusion_input)

            # score = self.score_predictor[i](token_fusion_input, modality_len)
            score = layer.attn.attn_score.split(modality_len, dim=0)
            # mask = [item.softmax(dim=-1) for item in score]
            stacked_score = torch.stack(score)
            original_shape = stacked_score.shape
            # Handle different dimensional inputs dynamically
            if stacked_score.dim() == 3:
                score = stacked_score.permute(1, 2, 0).softmax(dim=-1)
            elif stacked_score.dim() == 4:
                # If 4D, squeeze one dimension then permute
                squeezed_score = stacked_score.squeeze()
                if squeezed_score.dim() == 3:
                    score = squeezed_score.permute(1, 2, 0).softmax(dim=-1)
                else:
                    # If squeeze didn't reduce to 3D, manually handle the permutation
                    dims = list(range(1, stacked_score.dim())) + [0]
                    score = stacked_score.squeeze().permute(*dims).softmax(dim=-1)
            else:
                # For other dimensionalities, try to handle generally
                score = stacked_score.permute(1, 2, 0).softmax(dim=-1)

            mask = [score[:,:,i] for i in range(score.shape[-1])]
            modality_score = score.std(dim=1)
            modality_score = modality_score / modality_score.sum(dim=-1,keepdim=True)
            # modality_score = sigmoid_with_scale(modality_score)
            # modality_score = modality_score / modality_score.sum(dim=-1,keepdim=True)
            # modality_score = modality_score / modality_score.sum(dim=-1).squeeze
            # print(score.shape, (score[:,:,0].max()-score[:, :, 0].min()).mean().detach().cpu().item())
            # print(score[:,:,0].std(dim=0)[0].detach().item(), score[:,:,1].std(dim=0)[0].detach().item(), score[:,:,2].std(dim=0)[0].detach().item())
            modality_threshold = self.threshold_setting(modality_score)

            # print([(mm.min().cpu().item(),mm.max().cpu().item(), mm.std().cpu().item(), mm.mean().cpu().item()) for mm in mask])
            token_fusion_input = token_fusion_input.split(modality_len, dim=0)
            # Need to handle broadcasting properly - adjust mask dimensions to match x_
            # Properly expand mask to match x_ dimensions by adding sequence dim and expanding
            token_fusion_input = [x_ * mask_.unsqueeze(1).unsqueeze(-1).expand_as(x_) for (x_, mask_) in zip(token_fusion_input, mask)]
            if self.fusion_type == 'circle':
                tmp_modality_x = []
                for s_id in range(len(modality_list)):
                    d_id = s_id + 1 if (s_id+1) < len(modality_list) else 0
                    tmp_modality_x.append(
                        token_exchange([token_fusion_input[s_id], token_fusion_input[d_id]], mask[s_id], mask_threshold=modality_threshold[s_id])
                    )
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'max':
                tmp_modality_x = []
                tmp_modality_x = self.max_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'mean':
                tmp_modality_x = []
                tmp_modality_x = self.mean_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            elif self.fusion_type == 'learnable':
                tmp_modality_x = []
                tmp_modality_x = self.learnable_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                token_fusion_input = torch.cat(tmp_modality_x, dim=0)
            else:
                token_fusion_input = torch.cat(token_fusion_input, dim=0)
                    # exit()
        token_fusion_input = token_fusion_input.split(modality_len, dim=0)
        limoe_loss_input = modality_x_feature.split(modality_len, dim=0)
        self.limoe_loss = 0
        self.limoe_loss = self.calc_limoe_loss_v2(token_fusion_input)
        
        modality_spc_x = {mm: token_fusion_input[i] for i, mm in enumerate(modality_list)}
        token_fusion_input = [item.mean(dim=1) for item in token_fusion_input]
        token_fusion_input = torch.cat(token_fusion_input, -1)
        x = self.head(token_fusion_input)
        mspc_x = {}
        self.clear_router_cache()
        for mm in modality_spc_x:
            # print(modality_spc_x[mm].shape)
            mspc_x[mm] = self.modality_head[mm](modality_spc_x[mm].mean(dim=1))
        return x, mspc_x

    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss()
        return tmp_loss

    def clear_router_cache(self):
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                if len(mm.router_logits) > len(self.modalities):
                    mm.router_logits = []

    def threshold_setting(self, modality_score: torch.Tensor, threshold_values=[0.1, 0.1, 0.1]):
        modality_rank = modality_score.argsort(dim=-1)
        modality_thresholds = []
        # print(modality_rank)

        # print(modality_score[0, ])
        assign_weight = 0.3
        for i in range(modality_rank.shape[-1]):
            mm_rank = modality_rank[:, i]
            tt = torch.zeros_like(mm_rank, dtype=torch.float32)
            
            for j in range(mm_rank.max()+1):
                # print(j, mm_rank==j, threshold_values[j])
                # print(mm_rank==j)
                # tt[mm_rank==j] = threshold_values[j]
                tt= 0.6 * modality_score[:,j]
            # tt[:2]=0.3
            # print(tt)
            modality_thresholds.append(tt)
        # print(modality_thresholds)
        return modality_thresholds
    

    def max_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            modality_selected = tmp_mask.argmax(dim=-1)
            modality_selected = modality_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, modality_x[s_id].shape[-1])
                    # print(modality_selected.shape, tmp_replace_modality.shape)
            tmp_replace_modality = torch.gather(tmp_replace_modality, 2, modality_selected).squeeze(2)
            # print(tmp_replace_modality.shape, modality_x[s_id].shape)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def mean_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def learnable_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id, mod in enumerate(all_modalities):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            # tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            # tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], self.latents_dict[mod].repeat(modality_x[s_id].shape[0],1,1).to(dtype=modality_x[s_id].dtype)], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    
class SparseMultimodalTransformer(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, fusion_type='circle', num_experts=16, topk=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))

        self.latents_dict = nn.ParameterDict(self.latents_dict)
        # self.latents_norm = nn.LayerNorm(self.hidden_size)

        input_projector = {}
        for kk in self.modalities:
            # print(kk, self.modalities[kk].input_dim)
            input_projector[kk] = nn.ModuleDict(
                {
                    'patch_emb': PatchEmbeddings(self.modalities[kk].input_dim, self.modalities[kk].modality_latent_len,
                                                 self.hidden_size)
                }
            )
            self.pos_embedding = PositionalEncoding(
                self.hidden_size, self.modalities[kk].modality_latent_len
            )
        self.input_projector = nn.ModuleDict(input_projector)
        self.token_type_embeddings = nn.Embedding(len(self.modalities), self.hidden_size)
        self.n_blocks = n_blocks
        
        self.blocks = nn.ModuleList([
                SparseTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate, num_experts = num_experts, topk=topk) for i in range(n_blocks)
            ])
        
        n_modalities = len(self.modalities)
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )

        modality_head = {}
        for kk in self.modalities:
            modality_head[kk] = nn.Sequential(
                *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim), nn.Sigmoid()]
            )
        self.modality_head = nn.ModuleDict(modality_head)
    
    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_loss(modality_x_feature[i].mean(dim=-2), modality_x_feature[j].mean(dim=-2)))
        
        return sum(loss_list)
    
    def calc_limoe_loss_v2(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                # feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_lossv2(modality_x_feature[i], modality_x_feature[j]))
        
        return sum(loss_list)
        
    def forward(self, all_modalities):
        modality_x = {}
        modality_len = []
        modality_list = [kk for kk in all_modalities]
        modality_list.sort()
        for i, kk in enumerate(modality_list):
            data = all_modalities[kk]
            # batch_size, *axis, _ = data.shape
            # print(kk, data.shape)
            x = self.input_projector[kk]['patch_emb'](data.squeeze())
            # x += self.token_type_embeddings(torch.tensor(i).to(x.device))
            x = self.pos_embedding(x)
            modality_x[kk] = x
        
        token_fusion_input = []
        for kk in modality_list:
            x = modality_x[kk]
            token_fusion_input.append(x)
            modality_len.append(x.shape[0])
            
        token_fusion_input = torch.cat(token_fusion_input, dim=0)
        modality_x_feature = token_fusion_input.split(modality_len, dim=0)
        for i, layer in enumerate(self.blocks):
            
            token_fusion_input = layer(token_fusion_input)

        token_fusion_input = token_fusion_input.split(modality_len, dim=0)
        self.limoe_loss = 0
        self.limoe_loss = self.calc_limoe_loss_v2(token_fusion_input)
        
        modality_spc_x = {mm: token_fusion_input[i] for i, mm in enumerate(modality_list)}
        token_fusion_input = [item.mean(dim=1) for item in token_fusion_input]
        token_fusion_input = torch.cat(token_fusion_input, -1)
        x = self.head(token_fusion_input)
        mspc_x = {}
        self.clear_router_cache()
        for mm in modality_spc_x:
            # print(modality_spc_x[mm].shape)
            mspc_x[mm] = self.modality_head[mm](modality_spc_x[mm].mean(dim=1))
        return x, mspc_x

    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss()
        return tmp_loss

    def clear_router_cache(self):
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                if len(mm.router_logits) > len(self.modalities):
                    mm.router_logits = []
                    
                    
                    
                    
                    
                    
class MultimodalTransformerUniTokenFusionV4(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, latent_length, n_heads = 8, n_blocks=4,
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, fusion_type='circle', num_experts=16, n_mult_layer = 4, topk=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.modalities = modalities
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.latents_dict = {}
        for kk in self.modalities:
            self.latents_dict[kk] = nn.Parameter(torch.randn(self.modalities[kk].modality_latent_len, hidden_size))

        self.latents_dict = nn.ParameterDict(self.latents_dict)
        # self.latents_norm = nn.LayerNorm(self.hidden_size)

        input_projector = {}
        for kk in self.modalities:
            # print(kk, self.modalities[kk].input_dim)
            if kk == 'image':
                input_projector[kk] = nn.ModuleDict(
                    {
                        'patch_emb': QFormerEncoder(self.modalities[kk].input_dim, self.modalities[kk].modality_latent_len,
                                                    self.hidden_size)
                    }
                )
            else:
                input_projector[kk] = nn.ModuleDict(
                    {
                        'patch_emb': PatchEmbeddings(self.modalities[kk].input_dim, self.modalities[kk].modality_latent_len,
                                                    self.hidden_size)
                    }
                )
            self.pos_embedding = PositionalEncoding(
                self.hidden_size, self.modalities[kk].modality_latent_len
            )
        self.input_projector = nn.ModuleDict(input_projector)
        self.token_type_embeddings = nn.Embedding(len(self.modalities), self.hidden_size)
        self.mult_align = TFFusionAlign(modalities, hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate, n_layer=n_mult_layer)
        self.n_blocks = n_blocks
        
        # self.blocks = nn.ModuleList([
        #         DiffTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
        #     ])
        
        self.blocks = nn.ModuleList([
                DiffSparseTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate, num_experts=num_experts, topk=topk) for i in range(n_blocks)
            ])
        
        # self.blocks = nn.ModuleList([
        #         DiffTransformerBlock(hidden_size, n_heads, mlp_ratio, False, dropout_rate, dropout_rate) for i in range(n_blocks)
        #     ])
        
        n_modalities = len(self.modalities)
        # self.score_predictor = nn.ModuleList(
        #     [ScorePredictorv2(n_modalities, self.hidden_size) for i in range(n_blocks)]
        # )
        
        self.head = nn.Sequential(
            *[torch.nn.LayerNorm(self.hidden_size * n_modalities), nn.Linear(self.hidden_size * n_modalities, pred_dim), nn.Sigmoid()]
        )

        modality_head = {}
        for kk in self.modalities:
            modality_head[kk] = nn.Sequential(
                *[torch.nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, pred_dim), nn.Sigmoid()]
            )
        self.modality_head = nn.ModuleDict(modality_head)
    
    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_loss(modality_x_feature[i].mean(dim=-2), modality_x_feature[j].mean(dim=-2)))
        
        return sum(loss_list)
    
    def calc_limoe_loss_v2(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                # feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_lossv2(modality_x_feature[i], modality_x_feature[j]))
        
        return sum(loss_list)
        
    def forward(self, all_modalities, epoch_id = 0):
        modality_x = {}
        modality_len = []
        token_fusion_ratio = []
        modality_list = [kk for kk in all_modalities]
        modality_list.sort()
        for i, kk in enumerate(modality_list):
            data = all_modalities[kk]
            # batch_size, *axis, _ = data.shape
            # print(kk, data.shape, data.dtype)
            # print(data.shape)
            x = self.input_projector[kk]['patch_emb'](data.squeeze(1))
            # x += self.token_type_embeddings(torch.tensor(i).to(x.device))
            x = self.pos_embedding(x)
            modality_x[kk] = x
        modality_x = self.mult_align(modality_x)
        
        token_fusion_input = []
        for kk in modality_list:
            x = modality_x[kk]
            token_fusion_input.append(x)
            modality_len.append(x.shape[0])
            token_fusion_ratio.append(self.modalities[kk].token_fusion_topk_ratio)
            
        token_fusion_input = torch.cat(token_fusion_input, dim=0)
        modality_x_feature = token_fusion_input.split(modality_len, dim=0)
        
        
        for i, layer in enumerate(self.blocks):
            
            token_fusion_input = layer(token_fusion_input)
            # score = self.score_predictor[i](token_fusion_input, modality_len)
            score = layer.attn.attn_score.max(dim=-2)[0] # (n_modality * batch, n_seq, n_seq) -> (n_modality * batch, n_seq)
            # print(score.shape)
            score = score.split(modality_len, dim=0) # (n_modality * batch, n_seq) -> 3 * (batch, n_seq)
            # mask = [item.softmax(dim=-1) for item in score]
            score = torch.stack(score).permute(1, 2, 0)#.softmax(dim=-1)

            # Token Score
            m_score = score.detach().softmax(dim=-1)
            mask = [m_score[:,:,i] for i in range(m_score.shape[-1])]
            # Modality Score for Attention
            modality_score = score.mean(dim=1)

            modality_score = modality_score / (modality_score.sum(dim=-1,keepdim=True)+1e-4)

            modality_threshold = self.threshold_setting(modality_score)
            
            # print([(mm.min().cpu().item(),mm.max().cpu().item(), mm.std().cpu().item(), mm.mean().cpu().item()) for mm in mask])
            token_fusion_input = token_fusion_input.split(modality_len, dim=0)
            token_fusion_input = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(token_fusion_input, mask)]
            if epoch_id < 10 or len(token_fusion_input) == 1:
                token_fusion_input = torch.cat(token_fusion_input, dim=0)
            else:
                if self.fusion_type == 'circle':
                    tmp_modality_x = []
                    for s_id in range(len(modality_list)):
                        d_id = s_id + 1 if (s_id+1) < len(modality_list) else 0
                        tmp_modality_x.append(
                            token_exchange([token_fusion_input[s_id], token_fusion_input[d_id]], mask[s_id], mask_threshold=modality_threshold[s_id])
                        )
                    token_fusion_input = torch.cat(tmp_modality_x, dim=0)
                elif self.fusion_type == 'max':
                    tmp_modality_x = []
                    tmp_modality_x = self.max_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                    token_fusion_input = torch.cat(tmp_modality_x, dim=0)
                elif self.fusion_type == 'mean':
                    tmp_modality_x = []
                    tmp_modality_x = self.mean_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                    token_fusion_input = torch.cat(tmp_modality_x, dim=0)
                elif self.fusion_type == 'learnable':
                    tmp_modality_x = []
                    tmp_modality_x = self.learnable_token_fusion(modality_list, token_fusion_input, modality_threshold, mask, tmp_modality_x)
                    token_fusion_input = torch.cat(tmp_modality_x, dim=0)
                else:
                    token_fusion_input = torch.cat(token_fusion_input, dim=0)
                    # exit()
        token_fusion_input = token_fusion_input.split(modality_len, dim=0)
        # limoe_loss_input = modality_x_feature.split(modality_len, dim=0)
        self.limoe_loss = 0
        if len(modality_x_feature) > 1:
            self.limoe_loss = self.calc_limoe_loss_v2(modality_x_feature) + self.calc_limoe_loss(modality_x_feature)
        
        
        modality_spc_x = {mm: token_fusion_input[i] for i, mm in enumerate(modality_list)}
        token_fusion_input = [item.mean(dim=1) for item in token_fusion_input]
        token_fusion_input = torch.cat(token_fusion_input, -1)
        x = self.head(token_fusion_input)
        mspc_x = {}
        self.clear_router_cache()
        for mm in modality_spc_x:
            # print(modality_spc_x[mm].shape)
            mspc_x[mm] = self.modality_head[mm](modality_spc_x[mm].mean(dim=1))
        return x, mspc_x

    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss()
        return tmp_loss * 0.1

    def clear_router_cache(self):
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                if len(mm.router_logits) > len(self.modalities):
                    mm.router_logits = []

    def threshold_setting(self, modality_score: torch.Tensor, threshold_values=[0.1, 0.1, 0.1]):
        modality_rank = modality_score.argsort(dim=-1)
        modality_thresholds = []
        # print(modality_rank)

        # print(modality_score[0, ])
        # assign_weight = 0.3
        for i in range(modality_rank.shape[-1]):
            mm_rank = modality_rank[:, i]
            tt = torch.zeros_like(mm_rank, dtype=torch.float32)
            
            for j in range(mm_rank.max()+1):
                # print(j, mm_rank==j, threshold_values[j])
                # print(mm_rank==j)
                # tt[mm_rank==j] = threshold_values[j]
                tt= 1.0 * modality_score[:,j]
                # tt = 0.33 * torch.ones_like(modality_score[:,j], dtype=modality_score[:,j].dtype, device=modality_score[:,j].device)
            # tt[:2]=0.3
            # print(tt)
            modality_thresholds.append(tt)
        # print(modality_thresholds)
        return modality_thresholds
    

    def max_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            modality_selected = tmp_mask.argmax(dim=-1)
            modality_selected = modality_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, modality_x[s_id].shape[-1])
                    # print(modality_selected.shape, tmp_replace_modality.shape)
            tmp_replace_modality = torch.gather(tmp_replace_modality, 2, modality_selected).squeeze(2)
            # print(tmp_replace_modality.shape, modality_x[s_id].shape)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def mean_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id in range(len(all_modalities)):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], tmp_replace_modality], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
    
    def learnable_token_fusion(self, all_modalities, modality_x, token_fusion_ratio, mask, tmp_modality_x):
        for s_id, mod in enumerate(all_modalities):
            tmp_mask = []
            tmp_replace_modality = []
            for i, mm in enumerate(mask):
                if i != s_id:
                    tmp_mask.append(mm)
                    tmp_replace_modality.append(modality_x[i])
                    # print(tmp_replace_modality)
            tmp_mask = torch.stack(tmp_mask, dim=-1)
            # tmp_replace_modality = torch.stack(tmp_replace_modality, dim=-2)
            # tmp_replace_modality = tmp_replace_modality.mean(dim=-2)
            tmp_modality_x.append(
                        token_exchange_ratiov2([modality_x[s_id], self.latents_dict[mod].repeat(modality_x[s_id].shape[0],1,1).to(dtype=modality_x[s_id].dtype)], mask[s_id], mask_threshold=token_fusion_ratio[s_id])
                    )
        return tmp_modality_x
