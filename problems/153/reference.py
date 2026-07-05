import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
    attention_scale: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)
    
    residual = hidden_states
    
    # 1. Input RMSNorm
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)
    
    # 2. QKV Projection
    q = F.linear(hidden_states, q_proj_weight)
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight)
    
    # Reshape to separate heads
    q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
    k = k.view(batch_size, seq_len, num_key_value_heads, head_dim)
    v = v.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # 3. QK Normalization
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)
    
    # Transpose to [batch, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # 4. Compute RoPE embeddings
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    
    # Apply RoPE
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k_rotated = torch.cat((-k2, k1), dim=-1)
    q = (q * cos) + (q_rotated * sin)
    k = (k * cos) + (k_rotated * sin)
    
    # 5. Repeat KV heads for GQA
    k = k[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)
    k = k.reshape(batch_size, num_attention_heads, seq_len, head_dim)
    v = v[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)
    v = v.reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # 6. Attention computation
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * attention_scale
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # 7. Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    # 8. First residual
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # 9. Post-attention RMSNorm
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)
    
    # 10. SwiGLU MLP
    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
    gate = F.silu(gate)
    intermediate = gate * up
    mlp_output = F.linear(intermediate, down_proj_weight)
    
    # 11. Final residual
    output = residual + mlp_output
    
    return output
