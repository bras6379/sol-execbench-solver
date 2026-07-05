import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor, eps: float):
    """
    Per-head RMSNorm for both query and key tensors.
    
    Args:
        query: Query tensor [batch_size, seq_len, num_heads, head_dim]
        key: Key tensor [batch_size, seq_len, num_heads, head_dim]
        weight_q: Per-head scale for query [num_heads, head_dim]
        weight_k: Per-head scale for key [num_heads, head_dim]
        eps: Epsilon for numerical stability
    
    Returns:
        Tuple of (normalized_query, normalized_key)
    """
    input_dtype = query.dtype
    
    # Compute RMS for query per head (normalize over head_dim dimension)
    # variance shape: [batch, seq_len, num_heads, 1]
    q_variance = query.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    query_norm = query * torch.rsqrt(q_variance + eps)
    
    # Compute RMS for key per head
    k_variance = key.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    key_norm = key * torch.rsqrt(k_variance + eps)
    
    # Apply learnable scales
    # weight shape: [num_heads, head_dim]
    # Broadcast to [batch, seq_len, num_heads, head_dim]
    query_norm = query_norm * weight_q.unsqueeze(0).unsqueeze(0)
    key_norm = key_norm * weight_k.unsqueeze(0).unsqueeze(0)
    
    return query_norm.to(input_dtype), key_norm.to(input_dtype)
