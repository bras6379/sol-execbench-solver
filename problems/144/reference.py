import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    resnet1_norm1_weight: torch.Tensor,
    resnet1_norm1_bias: torch.Tensor,
    resnet1_conv1_weight: torch.Tensor,
    resnet1_conv1_bias: torch.Tensor,
    resnet1_time_emb_proj_weight: torch.Tensor,
    resnet1_time_emb_proj_bias: torch.Tensor,
    resnet1_norm2_weight: torch.Tensor,
    resnet1_norm2_bias: torch.Tensor,
    resnet1_conv2_weight: torch.Tensor,
    resnet1_conv2_bias: torch.Tensor,
    attn_group_norm_weight: torch.Tensor,
    attn_group_norm_bias: torch.Tensor,
    attn_to_q_weight: torch.Tensor,
    attn_to_q_bias: torch.Tensor,
    attn_to_k_weight: torch.Tensor,
    attn_to_k_bias: torch.Tensor,
    attn_to_v_weight: torch.Tensor,
    attn_to_v_bias: torch.Tensor,
    attn_to_out_weight: torch.Tensor,
    attn_to_out_bias: torch.Tensor,
    resnet2_norm1_weight: torch.Tensor,
    resnet2_norm1_bias: torch.Tensor,
    resnet2_conv1_weight: torch.Tensor,
    resnet2_conv1_bias: torch.Tensor,
    resnet2_time_emb_proj_weight: torch.Tensor,
    resnet2_time_emb_proj_bias: torch.Tensor,
    resnet2_norm2_weight: torch.Tensor,
    resnet2_norm2_bias: torch.Tensor,
    resnet2_conv2_weight: torch.Tensor,
    resnet2_conv2_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1  # Single-head attention at VAE bottleneck
    head_dim = channels  # head_dim equals channels when num_heads=1
    scale = head_dim ** -0.5
    
    # ============ ResNet Block 1 ============
    residual1 = hidden_states
    
    # GroupNorm1 + SiLU + Conv1
    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)
    
    # Add time embedding
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    
    # GroupNorm2 + SiLU + Conv2
    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)
    
    # Residual connection
    hidden_states = h + residual1
    
    # ============ Attention Block ============
    attn_residual = hidden_states
    
    # GroupNorm
    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    
    # Reshape to [B, H*W, C]
    h = h.view(batch, channels, height * width).transpose(1, 2)
    
    # QKV projections
    query = F.linear(h, attn_to_q_weight, attn_to_q_bias)
    key = F.linear(h, attn_to_k_weight, attn_to_k_bias)
    value = F.linear(h, attn_to_v_weight, attn_to_v_bias)
    
    # Reshape for multi-head attention [B, num_heads, H*W, head_dim]
    seq_len = height * width
    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention scores [B, num_heads, H*W, H*W]
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    
    # Apply attention to values [B, num_heads, H*W, head_dim]
    h = torch.matmul(attention_probs, value)
    
    # Reshape back [B, H*W, C]
    h = h.transpose(1, 2).reshape(batch, seq_len, channels)
    
    # Output projection
    h = F.linear(h, attn_to_out_weight, attn_to_out_bias)
    
    # Reshape to [B, C, H, W]
    h = h.transpose(1, 2).view(batch, channels, height, width)
    
    # Residual connection
    hidden_states = h + attn_residual
    
    # ============ ResNet Block 2 ============
    residual2 = hidden_states
    
    # GroupNorm1 + SiLU + Conv1
    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)
    
    # Add time embedding
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    
    # GroupNorm2 + SiLU + Conv2
    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)
    
    # Residual connection
    output = h + residual2
    
    return output
