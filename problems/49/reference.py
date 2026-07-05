import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    """
    Fused attention Q@K^T computation with GQA key repetition and scaling.
    
    Args:
        query: Query tensor of shape (batch, num_attention_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_key_value_heads, seq_len, head_dim)
        scaling: Scaling factor (1/sqrt(query_pre_attn_scalar))
    
    Returns:
        attn_scores: Attention scores of shape (batch, num_attention_heads, seq_len, seq_len)
    """
    batch, num_key_value_heads, slen, head_dim = key.shape
    num_attention_heads = query.shape[1]
    n_rep = num_attention_heads // num_key_value_heads  # 8
    
    # Step 1: Repeat key heads to match query heads (GQA)
    # Expand: (batch, num_kv_heads, 1, seq_len, head_dim) -> (batch, num_kv_heads, n_rep, seq_len, head_dim)
    key_expanded = key[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    # Reshape: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    key_states = key_expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    # Step 2: Compute attention scores Q@K^T
    # query: (batch, num_attention_heads, seq_len, head_dim)
    # key_states.transpose(2, 3): (batch, num_attention_heads, head_dim, seq_len)
    # Result: (batch, num_attention_heads, seq_len, seq_len)
    attn_weights = torch.matmul(query.to(torch.float32), key_states.transpose(2, 3).to(torch.float32))
    
    # Step 3: Apply scaling
    attn_weights = attn_weights * scaling
    
    return attn_weights.to(query.dtype)
