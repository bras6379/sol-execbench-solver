import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, dim = hidden_states.shape
    num_heads = 24
    head_dim = 64
    scale = head_dim ** -0.5
    
    # QKV projection: (batch, seq_len, dim) -> (batch, seq_len, 3 * dim)
    qkv = torch.matmul(hidden_states, qkv_weight.t()) + qkv_bias
    
    # Reshape and split: (batch, seq_len, 3 * dim) -> 3 x (batch, num_heads, seq_len, head_dim)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply LayerNorm to Q: normalize across head_dim
    # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    q_mean = q.mean(dim=-1, keepdim=True)
    q_var = q.var(dim=-1, unbiased=False, keepdim=True)
    q_normalized = (q - q_mean) / torch.sqrt(q_var + eps)
    q = q_normalized * q_norm_weight + q_norm_bias
    
    # Apply LayerNorm to K: normalize across head_dim
    k_mean = k.mean(dim=-1, keepdim=True)
    k_var = k.var(dim=-1, unbiased=False, keepdim=True)
    k_normalized = (k - k_mean) / torch.sqrt(k_var + eps)
    k = k_normalized * k_norm_weight + k_norm_bias
    
    # Scale query
    q = q * scale
    
    # Compute attention scores: (batch, num_heads, seq_len, seq_len)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    
    # Softmax over last dimension
    attn_probs = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values: (batch, num_heads, seq_len, head_dim)
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, dim)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    
    # Output projection
    output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    
    return output
