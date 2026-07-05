import torch

@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Apply rotary position embeddings to query and key tensors.
    
    RoPE rotation formula: x_rotated = x * cos + rotate_half(x) * sin
    where rotate_half([x0, x1, x2, x3, ...]) = [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)
    
    Returns:
        Tuple of (rotated_query, rotated_key) with same shapes as inputs
    """
    head_dim = query.shape[-1]
    half_dim = head_dim // 2
    
    # Reshape cos and sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Rotate half function: for input [x0, x1, ..., x_{d-1}]
    # returns [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    query_rotated = query * cos + rotate_half(query) * sin
    key_rotated = key * cos + rotate_half(key) * sin
    
    return query_rotated, key_rotated
