import torch


@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Fused KV cache update with RoPE application.
    
    Applies rotary position embeddings to incoming key states and concatenates
    with existing cache to produce updated key and value caches.
    
    Args:
        key_states: New key states (batch, num_kv_heads, new_seq_len, head_dim)
        value_states: New value states (batch, num_kv_heads, new_seq_len, head_dim)
        cos: Cosine position embeddings (batch, 1, new_seq_len, head_dim)
        sin: Sine position embeddings (batch, 1, new_seq_len, head_dim)
        key_cache: Existing key cache (batch, num_kv_heads, current_seq_len, head_dim)
        value_cache: Existing value cache (batch, num_kv_heads, current_seq_len, head_dim)
        
    Returns:
        Tuple of (updated_key_cache, updated_value_cache)
    """
    # Apply RoPE to incoming key states
    # rotate_half: split in half along last dim, negate second half, swap
    head_dim = key_states.shape[-1]
    half_dim = head_dim // 2
    
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    
    # Apply rotation: (k * cos) + (rotate_half(k) * sin)
    key_states_rotated = (key_states * cos) + (k_rotated * sin)
    
    # Concatenate with existing cache along sequence dimension
    updated_key_cache = torch.cat([key_cache, key_states_rotated], dim=2)
    updated_value_cache = torch.cat([value_cache, value_states], dim=2)
    
    return updated_key_cache, updated_value_cache
