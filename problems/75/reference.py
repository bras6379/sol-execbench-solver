import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars['batch_size']
    seq_len = axes_and_scalars['seq_len']
    hidden_size = 1024
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    rope_theta = 10000.0
    
    q_proj_size = num_heads * head_dim
    kv_proj_size = num_key_value_heads * head_dim
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
    
    q_proj_weight = torch.randn(q_proj_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    k_proj_weight = torch.randn(kv_proj_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    v_proj_weight = torch.randn(kv_proj_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    o_proj_weight = torch.randn(hidden_size, q_proj_size, device=device, dtype=torch.float32) * 0.02
    
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    is_causal = False
    
    return {
        'hidden_states': hidden_states,
        'position_ids': position_ids,
        'q_proj_weight': q_proj_weight,
        'k_proj_weight': k_proj_weight,
        'v_proj_weight': v_proj_weight,
        'o_proj_weight': o_proj_weight,
        'inv_freq': inv_freq,
        'is_causal': is_causal
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    is_causal: bool
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    num_key_value_groups = num_heads // num_key_value_heads
    scaling = 1.0
    
    # Q/K/V projections
    query_states = torch.matmul(hidden_states, q_proj_weight.t())
    key_states = torch.matmul(hidden_states, k_proj_weight.t())
    value_states = torch.matmul(hidden_states, v_proj_weight.t())
    
    # Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Compute RoPE embeddings
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * scaling
    sin = emb.sin() * scaling
    
    # Apply RoPE
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    q_half1 = query_states[..., :head_dim // 2]
    q_half2 = query_states[..., head_dim // 2:]
    q_rotated = torch.cat((-q_half2, q_half1), dim=-1)
    query_states = (query_states * cos) + (q_rotated * sin)
    
    k_half1 = key_states[..., :head_dim // 2]
    k_half2 = key_states[..., head_dim // 2:]
    k_rotated = torch.cat((-k_half2, k_half1), dim=-1)
    key_states = (key_states * cos) + (k_rotated * sin)
    
    # Repeat K/V for GQA
    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)
    
    # Attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    
    # Attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    # Output projection
    attn_output = torch.matmul(attn_output, o_proj_weight.t())
    
    return attn_output, attn_weights
