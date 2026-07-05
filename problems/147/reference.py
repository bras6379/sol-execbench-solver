import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_theta: float,
) -> torch.Tensor:
    # Constants
    num_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    scaling = head_dim ** -0.5
    
    bsz, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    
    # RMSNorm helper
    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return weight * x.to(input_dtype)
    
    # RoPE helper
    def apply_rope(q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        
        inv_freq_expanded = inv_freq[None, :, None].float().expand(bsz, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        
        q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_embed = (q * cos) + (q_rotated * sin)
        
        k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = (k * cos) + (k_rotated * sin)
        
        return q_embed.to(q.dtype), k_embed.to(k.dtype)
    
    # Repeat KV helper
    def repeat_kv(x: torch.Tensor) -> torch.Tensor:
        batch, n_kv_heads, slen, hdim = x.shape
        if num_key_value_groups == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, num_key_value_groups, slen, hdim)
        return x.reshape(batch, n_kv_heads * num_key_value_groups, slen, hdim)
    
    # Self-attention block with residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)
    
    # Project Q, K, V
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape for multi-head attention
    query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE
    query_states, key_states = apply_rope(query_states, key_states, seq_len)
    
    # Repeat K/V for GQA
    key_states = repeat_kv(key_states)
    value_states = repeat_kv(value_states)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    causal_mask = attention_mask[:, :, :seq_len, :seq_len]
    attn_weights = attn_weights + causal_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, seq_len, -1)
    
    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output
    
    # MLP block with residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)
    
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(gate * up, down_proj_weight)
    hidden_states = residual + hidden_states
    
    return hidden_states
