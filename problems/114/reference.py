import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    # Constants
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 96
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    half_dim = head_dim // 2
    
    batch_size, seq_len, _ = hidden_states.shape
    
    # Helper: RMSNorm
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight * x_normed.to(input_dtype))
    
    # Helper: Repeat KV heads for GQA
    def repeat_kv(x, n_rep):
        batch, num_kv_heads, slen, hdim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hdim)
        return x.reshape(batch, num_kv_heads * n_rep, slen, hdim)
    
    # ===== ATTENTION BLOCK =====
    residual1 = hidden_states
    
    # Pre-attention normalization
    hidden_states = rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)
    
    # QKV projections
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape for attention
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE - cos and sin have shape (batch, seq_len, half_head_dim)
    # Expand to (batch, 1, seq_len, half_head_dim) for broadcasting
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    # Rotate query: split into first half and second half
    q1 = query_states[..., :half_dim]
    q2 = query_states[..., half_dim:]
    query_states = torch.cat([q1 * cos_expanded - q2 * sin_expanded, q1 * sin_expanded + q2 * cos_expanded], dim=-1)
    
    # Rotate key: split into first half and second half
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    key_states = torch.cat([k1 * cos_expanded - k2 * sin_expanded, k1 * sin_expanded + k2 * cos_expanded], dim=-1)
    
    # Repeat KV heads for GQA
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_output = F.linear(attn_output, o_proj_weight)
    
    # First residual connection
    hidden_states = residual1 + attn_output
    
    # ===== MLP BLOCK =====
    residual2 = hidden_states
    
    # Pre-MLP normalization
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, rms_norm_eps)
    
    # SwiGLU MLP
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(gate * up, down_proj_weight)
    
    # Second residual connection
    output = residual2 + hidden_states
    
    return output
