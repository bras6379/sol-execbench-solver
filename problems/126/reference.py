import torch
import torch.nn.functional as F
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    text_seq_len = axes_and_scalars["text_seq_len"]
    hidden_size = 3072
    joint_attention_dim = 4096
    modulation_dim = hidden_size * 6
    rope_axis0_dim = 16
    rope_axis1_dim = 56
    rope_axis2_dim = 56
    head_dim = 128
    rope_dim_per_axis = head_dim // 3  # 42
    is_joint_block = axes_and_scalars["is_joint_block"]
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    timestep_embedding = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, joint_attention_dim, device=device, dtype=torch.float32)
    
    adaln_linear_weight = torch.randn(modulation_dim, hidden_size, device=device, dtype=torch.float32) * 0.02
    adaln_linear_bias = torch.zeros(modulation_dim, device=device, dtype=torch.float32)
    
    to_q_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_q_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    to_k_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_k_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    to_v_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_v_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    to_k_context_weight = torch.randn(hidden_size, joint_attention_dim, device=device, dtype=torch.float32) * 0.02
    to_k_context_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    to_v_context_weight = torch.randn(hidden_size, joint_attention_dim, device=device, dtype=torch.float32) * 0.02
    to_v_context_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    to_out_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_out_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    pos_idx_axis0 = torch.randint(0, rope_axis0_dim, (seq_len,), device=device, dtype=torch.int64)
    pos_idx_axis1 = torch.randint(0, rope_axis1_dim, (seq_len,), device=device, dtype=torch.int64)
    pos_idx_axis2 = torch.randint(0, rope_axis2_dim, (seq_len,), device=device, dtype=torch.int64)
    
    rope_theta = 10000.0
    inv_freq0 = 1.0 / (rope_theta ** (torch.arange(0, rope_dim_per_axis, dtype=torch.float32, device=device) / rope_dim_per_axis))
    positions0 = torch.arange(rope_axis0_dim, dtype=torch.float32, device=device)
    freqs0 = torch.outer(positions0, inv_freq0)
    rope_cos_axis0 = freqs0.cos()
    rope_sin_axis0 = freqs0.sin()
    
    inv_freq1 = 1.0 / (rope_theta ** (torch.arange(0, rope_dim_per_axis, dtype=torch.float32, device=device) / rope_dim_per_axis))
    positions1 = torch.arange(rope_axis1_dim, dtype=torch.float32, device=device)
    freqs1 = torch.outer(positions1, inv_freq1)
    rope_cos_axis1 = freqs1.cos()
    rope_sin_axis1 = freqs1.sin()
    
    inv_freq2 = 1.0 / (rope_theta ** (torch.arange(0, rope_dim_per_axis, dtype=torch.float32, device=device) / rope_dim_per_axis))
    positions2 = torch.arange(rope_axis2_dim, dtype=torch.float32, device=device)
    freqs2 = torch.outer(positions2, inv_freq2)
    rope_cos_axis2 = freqs2.cos()
    rope_sin_axis2 = freqs2.sin()
    
    return {
        "hidden_states": hidden_states,
        "timestep_embedding": timestep_embedding,
        "encoder_hidden_states": encoder_hidden_states,
        "adaln_linear_weight": adaln_linear_weight,
        "adaln_linear_bias": adaln_linear_bias,
        "to_q_weight": to_q_weight,
        "to_q_bias": to_q_bias,
        "to_k_weight": to_k_weight,
        "to_k_bias": to_k_bias,
        "to_v_weight": to_v_weight,
        "to_v_bias": to_v_bias,
        "to_k_context_weight": to_k_context_weight,
        "to_k_context_bias": to_k_context_bias,
        "to_v_context_weight": to_v_context_weight,
        "to_v_context_bias": to_v_context_bias,
        "to_out_weight": to_out_weight,
        "to_out_bias": to_out_bias,
        "pos_idx_axis0": pos_idx_axis0,
        "pos_idx_axis1": pos_idx_axis1,
        "pos_idx_axis2": pos_idx_axis2,
        "rope_cos_axis0": rope_cos_axis0,
        "rope_sin_axis0": rope_sin_axis0,
        "rope_cos_axis1": rope_cos_axis1,
        "rope_sin_axis1": rope_sin_axis1,
        "rope_cos_axis2": rope_cos_axis2,
        "rope_sin_axis2": rope_sin_axis2,
        "is_joint_block": is_joint_block,
    }


def apply_rope_axis(x_part, pos_idx, rope_cos, rope_sin):
    """Apply RoPE to a single axis portion of the tensor.
    
    Args:
        x_part: (batch, seq_len, num_heads, rope_dim_per_axis)
        pos_idx: (seq_len,)
        rope_cos: (max_pos, rope_dim_per_axis)
        rope_sin: (max_pos, rope_dim_per_axis)
    
    Returns:
        Rotated tensor of same shape as x_part
    """
    cos_gathered = rope_cos[pos_idx]  # (seq_len, rope_dim_per_axis)
    sin_gathered = rope_sin[pos_idx]  # (seq_len, rope_dim_per_axis)
    
    cos_gathered = cos_gathered.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, rope_dim_per_axis)
    sin_gathered = sin_gathered.unsqueeze(0).unsqueeze(2)
    
    rope_dim = x_part.shape[-1]
    half_dim = rope_dim // 2
    
    x_first = x_part[..., :half_dim]
    x_second = x_part[..., half_dim:]
    
    cos_first = cos_gathered[..., :half_dim]
    sin_first = sin_gathered[..., :half_dim]
    
    rotated_first = x_first * cos_first - x_second * sin_first
    rotated_second = x_first * sin_first + x_second * cos_first
    
    return torch.cat([rotated_first, rotated_second], dim=-1)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    timestep_embedding: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    adaln_linear_weight: torch.Tensor,
    adaln_linear_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_q_bias: torch.Tensor,
    to_k_weight: torch.Tensor,
    to_k_bias: torch.Tensor,
    to_v_weight: torch.Tensor,
    to_v_bias: torch.Tensor,
    to_k_context_weight: torch.Tensor,
    to_k_context_bias: torch.Tensor,
    to_v_context_weight: torch.Tensor,
    to_v_context_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    pos_idx_axis0: torch.Tensor,
    pos_idx_axis1: torch.Tensor,
    pos_idx_axis2: torch.Tensor,
    rope_cos_axis0: torch.Tensor,
    rope_sin_axis0: torch.Tensor,
    rope_cos_axis1: torch.Tensor,
    rope_sin_axis1: torch.Tensor,
    rope_cos_axis2: torch.Tensor,
    rope_sin_axis2: torch.Tensor,
    is_joint_block: int,
):
    batch, seq_len, hidden_size = hidden_states.shape
    num_heads = 24
    head_dim = 128
    rope_dim_per_axis = head_dim // 3  # 42
    remaining_dim = head_dim - 3 * rope_dim_per_axis  # 128 - 126 = 2
    
    residual = hidden_states
    
    timestep_activated = timestep_embedding * torch.sigmoid(timestep_embedding)
    
    modulation = F.linear(timestep_activated, adaln_linear_weight, adaln_linear_bias)
    
    scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    
    hidden_states_normalized = F.layer_norm(hidden_states, (hidden_size,))
    
    hidden_states_modulated = hidden_states_normalized * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    
    q = F.linear(hidden_states_modulated, to_q_weight, to_q_bias)
    k = F.linear(hidden_states_modulated, to_k_weight, to_k_bias)
    v = F.linear(hidden_states_modulated, to_v_weight, to_v_bias)
    
    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_heads, head_dim)
    v = v.view(batch, seq_len, num_heads, head_dim)
    
    q_axis0 = q[..., :rope_dim_per_axis]
    q_axis1 = q[..., rope_dim_per_axis:2*rope_dim_per_axis]
    q_axis2 = q[..., 2*rope_dim_per_axis:3*rope_dim_per_axis]
    q_rest = q[..., 3*rope_dim_per_axis:]
    
    k_axis0 = k[..., :rope_dim_per_axis]
    k_axis1 = k[..., rope_dim_per_axis:2*rope_dim_per_axis]
    k_axis2 = k[..., 2*rope_dim_per_axis:3*rope_dim_per_axis]
    k_rest = k[..., 3*rope_dim_per_axis:]
    
    q_axis0_rot = apply_rope_axis(q_axis0, pos_idx_axis0, rope_cos_axis0, rope_sin_axis0)
    q_axis1_rot = apply_rope_axis(q_axis1, pos_idx_axis1, rope_cos_axis1, rope_sin_axis1)
    q_axis2_rot = apply_rope_axis(q_axis2, pos_idx_axis2, rope_cos_axis2, rope_sin_axis2)
    
    k_axis0_rot = apply_rope_axis(k_axis0, pos_idx_axis0, rope_cos_axis0, rope_sin_axis0)
    k_axis1_rot = apply_rope_axis(k_axis1, pos_idx_axis1, rope_cos_axis1, rope_sin_axis1)
    k_axis2_rot = apply_rope_axis(k_axis2, pos_idx_axis2, rope_cos_axis2, rope_sin_axis2)
    
    q = torch.cat([q_axis0_rot, q_axis1_rot, q_axis2_rot, q_rest], dim=-1)
    k = torch.cat([k_axis0_rot, k_axis1_rot, k_axis2_rot, k_rest], dim=-1)
    
    if is_joint_block == 1:
        text_seq_len = encoder_hidden_states.shape[1]
        
        encoder_hidden_states_normalized = F.layer_norm(encoder_hidden_states, (4096,))
        
        k_context = F.linear(encoder_hidden_states_normalized, to_k_context_weight, to_k_context_bias)
        v_context = F.linear(encoder_hidden_states_normalized, to_v_context_weight, to_v_context_bias)
        
        k_context = k_context.view(batch, text_seq_len, num_heads, head_dim)
        v_context = v_context.view(batch, text_seq_len, num_heads, head_dim)
        
        k = torch.cat([k, k_context], dim=1)
        v = torch.cat([v, v_context], dim=1)
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    
    attn_output = torch.matmul(attn_probs, v)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
    
    attn_output = F.linear(attn_output, to_out_weight, to_out_bias)
    
    output = residual + gate_msa.unsqueeze(1) * attn_output
    
    return output
