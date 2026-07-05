import torch
import torch.nn.functional as F
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_attention_heads = axes_and_scalars["num_attention_heads"]
    num_key_value_heads = axes_and_scalars["num_key_value_heads"]
    head_dim = axes_and_scalars["head_dim"]
    
    qkv_out_dim = num_attention_heads * head_dim
    kv_out_dim = num_key_value_heads * head_dim
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create causal mask
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attention_mask = attention_mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    q_proj_weight = torch.randn(qkv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_weight = torch.randn(kv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_weight = torch.randn(kv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, qkv_out_dim, dtype=torch.bfloat16, device=device) * 0.02
    
    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    
    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "o_proj_weight": o_proj_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_theta": 10000.0,
        "softcap": 30.0,
        "rms_norm_eps": 1e-6
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_theta: float,
    softcap: float,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 8
    num_key_value_heads = 1
    head_dim = 256
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    # Q projection
    query_states = F.linear(hidden_states, q_proj_weight)  # [batch, seq, num_heads * head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    
    # Q RMSNorm
    q_variance = query_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    query_states = query_states * torch.rsqrt(q_variance + rms_norm_eps)
    query_states = (query_states * q_norm_weight).to(hidden_states.dtype)
    
    # K projection
    key_states = F.linear(hidden_states, k_proj_weight)  # [batch, seq, num_kv_heads * head_dim]
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # K RMSNorm
    k_variance = key_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    key_states = key_states * torch.rsqrt(k_variance + rms_norm_eps)
    key_states = (key_states * k_norm_weight).to(hidden_states.dtype)
    
    # V projection
    value_states = F.linear(hidden_states, v_proj_weight)  # [batch, seq, num_kv_heads * head_dim]
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # V RMSNorm (without scale)
    v_variance = value_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    value_states = (value_states * torch.rsqrt(v_variance + rms_norm_eps)).to(hidden_states.dtype)
    
    # Compute RoPE embeddings
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=hidden_states.device) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [batch, seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq, head_dim]
    cos = emb.cos().to(hidden_states.dtype)
    sin = emb.sin().to(hidden_states.dtype)
    
    # Apply RoPE to Q
    cos_q = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin_q = sin.unsqueeze(2)
    q1 = query_states[..., :head_dim // 2]
    q2 = query_states[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_q) + (q_rotated * sin_q)
    
    # Apply RoPE to K
    cos_k = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin_k = sin.unsqueeze(2)
    k1 = key_states[..., :head_dim // 2]
    k2 = key_states[..., head_dim // 2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_k) + (k_rotated * sin_k)
    
    # Transpose for attention: [batch, heads, seq, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Store KV states for sharing before repeat
    key_states_out = key_states.clone()
    value_states_out = value_states.clone()
    
    # Repeat KV heads for GQA
    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = attn_weights / math.sqrt(head_dim)
    
    # Apply soft-capping
    attn_weights = attn_weights / softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * softcap
    
    # Apply attention mask
    attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output, key_states_out, value_states_out
