import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    cache_len = axes_and_scalars["cache_len"]
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    half_head_dim = 64
    max_position_embeddings = 262144
    rope_theta = 10000000.0
    rms_norm_eps = 1e-6
    
    query = torch.randn(batch_size, num_attention_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    
    position_ids = torch.arange(cache_len, cache_len + seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
    
    key_cache = torch.randn(batch_size, num_key_value_heads, max_position_embeddings, head_dim, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn(batch_size, num_key_value_heads, max_position_embeddings, head_dim, dtype=torch.bfloat16, device=device)
    
    cache_position = torch.arange(cache_len, cache_len + seq_len, dtype=torch.int64, device=device)
    
    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    return {
        "query": query,
        "key": key,
        "value": value,
        "position_ids": position_ids,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cache_position": cache_position,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "inv_freq": inv_freq,
        "rms_norm_eps": rms_norm_eps,
    }

@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    
    def rms_norm(x, weight, eps):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight.to(torch.float32) * x_normed).to(x.dtype)
    
    query_norm = rms_norm(query, q_norm_weight, rms_norm_eps)
    key_norm = rms_norm(key, k_norm_weight, rms_norm_eps)
    
    inv_freq_expanded = inv_freq[None, None, :].expand(batch_size, seq_len, -1)
    position_ids_expanded = position_ids[:, :, None].float()
    freqs = position_ids_expanded * inv_freq_expanded
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(query.dtype)
    sin = emb.sin().to(query.dtype)
    
    def rotate_half(x):
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rope(x, cos, sin):
        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)
        return (x * cos_expanded) + (rotate_half(x) * sin_expanded)
    
    query_rotated = apply_rope(query_norm, cos, sin)
    key_rotated = apply_rope(key_norm, cos, sin)
 
    key_cache[:, :, cache_position] = key_rotated
    value_cache[:, :, cache_position] = value

    return query_rotated, key_rotated, key_cache, value_cache
