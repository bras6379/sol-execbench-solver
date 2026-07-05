import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pre_sa_norm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_sa_norm_weight: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    norm_eps: float,
):
    # Constants
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    num_key_value_groups = num_heads // num_key_value_heads
    
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Helper: RMSNorm
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(input_dtype)
    
    # Helper: Rotate half for RoPE
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Helper: Repeat KV for GQA
    def repeat_kv(x, n_rep):
        batch, num_kv_heads, slen, hdim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hdim)
        return x.reshape(batch, num_kv_heads * n_rep, slen, hdim)
    
    # ============ First Residual-Norm Pattern: Self-Attention ============
    residual_1 = hidden_states
    
    # Pre-attention RMSNorm
    normed_1 = rms_norm(residual_1, pre_sa_norm_weight, norm_eps)
    
    # Self-attention
    # Q, K, V projections
    query_states = torch.matmul(normed_1, q_proj_weight.t())
    key_states = torch.matmul(normed_1, k_proj_weight.t())
    value_states = torch.matmul(normed_1, v_proj_weight.t())
    
    # Reshape for multi-head attention
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE
    cos_expanded = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin_expanded = sin.unsqueeze(1)
    q_embed = (query_states * cos_expanded) + (rotate_half(query_states) * sin_expanded)
    k_embed = (key_states * cos_expanded) + (rotate_half(key_states) * sin_expanded)
    
    # Repeat K, V for grouped query attention
    key_states = repeat_kv(k_embed, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Attention computation (non-causal for encoder)
    attn_weights = torch.matmul(q_embed, key_states.transpose(2, 3)) / (head_dim ** 0.5)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_out = torch.matmul(attn_output, o_proj_weight.t())
    
    # First residual connection
    hidden_states = residual_1 + attn_out
    
    # ============ Second Residual-Norm Pattern: MLP ============
    residual_2 = hidden_states
    
    # Post-attention RMSNorm
    normed_2 = rms_norm(residual_2, post_sa_norm_weight, norm_eps)
    
    # Gated MLP with SiLU
    up_states = torch.matmul(normed_2, gate_up_proj_weight.t())
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = up_states * F.silu(gate)
    mlp_out = torch.matmul(up_states, down_proj_weight.t())
    
    # Second residual connection
    output = residual_2 + mlp_out
    
    return output
