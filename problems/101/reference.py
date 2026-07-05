import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Grouped Query Attention with Multi-modal 3D Rotary Position Embeddings.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        q_weight: [num_heads * head_dim, hidden_size]
        q_bias: [num_heads * head_dim]
        k_weight: [num_kv_heads * head_dim, hidden_size]
        k_bias: [num_kv_heads * head_dim]
        v_weight: [num_kv_heads * head_dim, hidden_size]
        v_bias: [num_kv_heads * head_dim]
        o_weight: [hidden_size, num_heads * head_dim]
        cos: [3, batch_size, seq_len, head_dim]
        sin: [3, batch_size, seq_len, head_dim]
        attention_mask: [batch_size, 1, seq_len, seq_len]
    
    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    # Constants
    num_heads = 28
    num_kv_heads = 4
    num_kv_groups = 7
    head_dim = 128
    scaling = head_dim ** -0.5
    mrope_section = [16, 24, 24]  # Channel splits for temporal/height/width
    
    bsz, q_len, _ = hidden_states.size()
    
    # Project to Q, K, V using linear operations
    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Apply multi-modal rotary position embeddings
    # mrope_section * 2 for cos/sin pairs: [32, 48, 48]
    mrope_section_doubled = [s * 2 for s in mrope_section]
    
    # Split cos/sin into 3 sections
    cos_splits = cos.split(mrope_section_doubled, dim=-1)
    sin_splits = sin.split(mrope_section_doubled, dim=-1)
    
    # Concatenate sections: [temporal, height, width] pattern
    cos_combined = torch.cat([m[i % 3] for i, m in enumerate(cos_splits)], dim=-1).unsqueeze(1)
    sin_combined = torch.cat([m[i % 3] for i, m in enumerate(sin_splits)], dim=-1).unsqueeze(1)
    
    # Rotate half helper
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotary embeddings
    query_states = (query_states * cos_combined) + (rotate_half(query_states) * sin_combined)
    key_states = (key_states * cos_combined) + (rotate_half(key_states) * sin_combined)
    
    # Repeat K, V for grouped query attention (4 -> 28 heads)
    # Expand from [batch, num_kv_heads, seq_len, head_dim] to [batch, num_heads, seq_len, head_dim]
    key_states = key_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim
    ).reshape(bsz, num_heads, q_len, head_dim)
    
    value_states = value_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim
    ).reshape(bsz, num_heads, q_len, head_dim)
    
    # Compute attention scores: Q @ K^T
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply attention mask (causal)
    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output: softmax(QK^T) @ V
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    output = F.linear(attn_output, o_weight)
    
    return output
