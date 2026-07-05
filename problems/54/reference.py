import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
):
    """
    Fused QKV projection with per-head RMS normalization.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size=1024]
        q_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        k_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        v_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        q_norm_weight: [head_dim=128]
        k_norm_weight: [head_dim=128]
        eps: epsilon for RMS norm
    
    Returns:
        query_states: [batch_size, seq_len, num_heads=8, head_dim=128]
        key_states: [batch_size, seq_len, num_heads=8, head_dim=128]
        value_states: [batch_size, seq_len, num_heads=8, head_dim=128]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 8
    head_dim = 128
    
    # Linear projections: [batch, seq, hidden] @ [hidden, qkv_out].T -> [batch, seq, qkv_out]
    # Using matmul with transposed weights
    query = torch.matmul(hidden_states, q_proj_weight.t())
    key = torch.matmul(hidden_states, k_proj_weight.t())
    value = torch.matmul(hidden_states, v_proj_weight.t())
    
    # Reshape to multi-head format: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)
    
    # Per-head RMS normalization for query
    # Compute variance over head_dim (last dimension)
    q_variance = query.pow(2).mean(dim=-1, keepdim=True)
    q_normed = query / torch.sqrt(q_variance + eps)
    query_states = q_normed * q_norm_weight
    
    # Per-head RMS normalization for key
    k_variance = key.pow(2).mean(dim=-1, keepdim=True)
    k_normed = key / torch.sqrt(k_variance + eps)
    key_states = k_normed * k_norm_weight
    
    # No RMS normalization for value in audio attention
    value_states = value
    
    return query_states, key_states, value_states
