import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Fused GQA key-value repetition and attention score computation.
    
    Args:
        query: [batch_size, num_attention_heads, seq_len, head_dim]
        key: [batch_size, num_key_value_heads, seq_len, head_dim]
    
    Returns:
        attn_weights: [batch_size, num_attention_heads, seq_len, seq_len]
    """
    # Constants
    head_dim = 128
    num_key_value_groups = 1
    scaling = head_dim ** -0.5
    
    batch_size, num_key_value_heads, slen, _ = key.shape
    
    # Repeat KV heads to match query heads
    # [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]
    key_expanded = key[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, slen, head_dim
    )
    key_states = key_expanded.reshape(batch_size, num_key_value_heads * num_key_value_groups, slen, head_dim)
    
    # Compute attention scores: Q @ K^T with scaling
    # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # -> [batch, num_heads, seq_len, seq_len]
    attn_weights = torch.matmul(query.to(torch.float32), key_states.transpose(2, 3).to(torch.float32)) * scaling
    
    return attn_weights.to(torch.bfloat16)
