import torch
import torch.nn.functional as F
import math


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
) -> tuple:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    freqs = torch.outer(position_ids.float().flatten(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    
    batch_size, num_heads, seq_len, hd = q.shape
    cos = cos.view(batch_size, 1, seq_len, hd)
    sin = sin.view(batch_size, 1, seq_len, hd)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    self_attn_norm_weight: torch.Tensor,
    self_attn_q_weight: torch.Tensor,
    self_attn_k_weight: torch.Tensor,
    self_attn_v_weight: torch.Tensor,
    self_attn_o_weight: torch.Tensor,
    cross_attn_norm_weight: torch.Tensor,
    cross_attn_q_weight: torch.Tensor,
    cross_attn_k_weight: torch.Tensor,
    cross_attn_v_weight: torch.Tensor,
    cross_attn_o_weight: torch.Tensor,
    mlp_norm_weight: torch.Tensor,
    mlp_gate_weight: torch.Tensor,
    mlp_up_weight: torch.Tensor,
    mlp_down_weight: torch.Tensor,
    norm_eps: float,
    rope_theta: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    encoder_seq_len = encoder_hidden_states.shape[1]
    
    num_attention_heads = 16
    num_key_value_heads = 4
    head_dim = 128
    cross_num_attention_heads = 16
    cross_num_key_value_heads = 16
    cross_head_dim = 128
    
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
    
    residual = hidden_states
    
    # Self-Attention Block
    hidden_states = rms_norm(hidden_states, self_attn_norm_weight, norm_eps)
    
    query_states = F.linear(hidden_states, self_attn_q_weight)
    key_states = F.linear(hidden_states, self_attn_k_weight)
    value_states = F.linear(hidden_states, self_attn_v_weight)
    
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    query_states, key_states = apply_rope(query_states, key_states, position_ids, head_dim, rope_theta)
    
    key_states = repeat_kv(key_states, num_attention_heads // num_key_value_heads)
    value_states = repeat_kv(value_states, num_attention_heads // num_key_value_heads)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
    
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_attention_heads * head_dim)
    attn_output = F.linear(attn_output, self_attn_o_weight)
    
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # Cross-Attention Block
    hidden_states = rms_norm(hidden_states, cross_attn_norm_weight, norm_eps)
    
    query_states = F.linear(hidden_states, cross_attn_q_weight)
    key_states = F.linear(encoder_hidden_states, cross_attn_k_weight)
    value_states = F.linear(encoder_hidden_states, cross_attn_v_weight)
    
    query_states = query_states.view(batch_size, seq_len, cross_num_attention_heads, cross_head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, encoder_seq_len, cross_num_key_value_heads, cross_head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, encoder_seq_len, cross_num_key_value_heads, cross_head_dim).transpose(1, 2)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(cross_head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, cross_num_attention_heads * cross_head_dim)
    attn_output = F.linear(attn_output, cross_attn_o_weight)
    
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # MLP Block
    hidden_states = rms_norm(hidden_states, mlp_norm_weight, norm_eps)
    
    gate = F.linear(hidden_states, mlp_gate_weight)
    up = F.linear(hidden_states, mlp_up_weight)
    hidden_states = F.silu(gate) * up
    hidden_states = F.linear(hidden_states, mlp_down_weight)
    
    output = residual + hidden_states
    
    return output
