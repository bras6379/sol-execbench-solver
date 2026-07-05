import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def run(
    video_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    temporal_norm_weight: torch.Tensor,
    temporal_norm_bias: torch.Tensor,
    temporal_qkv_weight: torch.Tensor,
    temporal_qkv_bias: torch.Tensor,
    temporal_out_proj_weight: torch.Tensor,
    temporal_out_proj_bias: torch.Tensor,
    spatial_norm_weight: torch.Tensor,
    spatial_norm_bias: torch.Tensor,
    spatial_qkv_weight: torch.Tensor,
    spatial_qkv_bias: torch.Tensor,
    spatial_out_proj_weight: torch.Tensor,
    spatial_out_proj_bias: torch.Tensor,
    cross_attn_norm_weight: torch.Tensor,
    cross_attn_norm_bias: torch.Tensor,
    cross_attn_q_weight: torch.Tensor,
    cross_attn_q_bias: torch.Tensor,
    cross_attn_kv_weight: torch.Tensor,
    cross_attn_kv_bias: torch.Tensor,
    cross_attn_out_proj_weight: torch.Tensor,
    cross_attn_out_proj_bias: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    ffn_norm_bias: torch.Tensor,
    ffn_fc1_weight: torch.Tensor,
    ffn_fc1_bias: torch.Tensor,
    ffn_fc2_weight: torch.Tensor,
    ffn_fc2_bias: torch.Tensor,
    num_frames_scalar: int,
    num_spatial_tokens_scalar: int,
):
    # Constants
    hidden_size = 1024
    num_attention_heads = 16
    head_dim = hidden_size // num_attention_heads
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-5
    
    batch_size = video_latents.shape[0]
    video_seq_len = video_latents.shape[1]
    text_seq_len = text_embeddings.shape[1]
    
    num_frames = int(num_frames_scalar)
    num_spatial_tokens = int(num_spatial_tokens_scalar)
    
    x = video_latents
    
    # ============ 1. Temporal Self-Attention ============
    residual = x
    
    # Layer norm
    x_norm = F.layer_norm(x, (hidden_size,), temporal_norm_weight, temporal_norm_bias, eps)
    
    # QKV projection: (B, F*S, 3*H)
    qkv = F.linear(x_norm, temporal_qkv_weight, temporal_qkv_bias)
    qkv = qkv.view(batch_size, num_frames, num_spatial_tokens, 3, hidden_size)
    
    # Rearrange for temporal attention: (B, S, F, 3, H)
    qkv = qkv.permute(0, 2, 1, 3, 4)
    q, k, v = qkv.chunk(3, dim=3)
    q, k, v = q.squeeze(3), k.squeeze(3), v.squeeze(3)  # (B, S, F, H)
    
    # Reshape for multi-head attention: (B*S, NH, F, HD)
    q = q.reshape(batch_size * num_spatial_tokens, num_frames, hidden_size)
    k = k.reshape(batch_size * num_spatial_tokens, num_frames, hidden_size)
    v = v.reshape(batch_size * num_spatial_tokens, num_frames, hidden_size)
    
    q = q.view(batch_size * num_spatial_tokens, num_frames, num_attention_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size * num_spatial_tokens, num_frames, num_attention_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size * num_spatial_tokens, num_frames, num_attention_heads, head_dim).transpose(1, 2)
    
    # Compute attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape back: (B*S, NH, F, HD) -> (B, F*S, H)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size * num_spatial_tokens, num_frames, hidden_size)
    attn_output = attn_output.view(batch_size, num_spatial_tokens, num_frames, hidden_size)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    attn_output = attn_output.view(batch_size, num_frames * num_spatial_tokens, hidden_size)
    
    # Output projection and residual
    output = F.linear(attn_output, temporal_out_proj_weight, temporal_out_proj_bias)
    x = output + residual
    
    # ============ 2. Spatial Self-Attention ============
    residual = x
    
    # Layer norm
    x_norm = F.layer_norm(x, (hidden_size,), spatial_norm_weight, spatial_norm_bias, eps)
    
    # Reshape to separate frames: (B, F, S, H)
    x_reshaped = x_norm.view(batch_size, num_frames, num_spatial_tokens, hidden_size)
    
    # QKV projection: (B*F, S, 3*H)
    qkv = F.linear(x_reshaped.view(batch_size * num_frames, num_spatial_tokens, hidden_size), 
                   spatial_qkv_weight, spatial_qkv_bias)
    qkv = qkv.view(batch_size * num_frames, num_spatial_tokens, 3, hidden_size)
    q, k, v = qkv.chunk(3, dim=2)
    q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
    
    # Reshape for multi-head attention: (B*F, NH, S, HD)
    q = q.view(batch_size * num_frames, num_spatial_tokens, num_attention_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size * num_frames, num_spatial_tokens, num_attention_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size * num_frames, num_spatial_tokens, num_attention_heads, head_dim).transpose(1, 2)
    
    # Compute attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape back: (B*F, NH, S, HD) -> (B, F*S, H)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size * num_frames, num_spatial_tokens, hidden_size)
    attn_output = attn_output.view(batch_size, num_frames * num_spatial_tokens, hidden_size)
    
    # Output projection and residual
    output = F.linear(attn_output, spatial_out_proj_weight, spatial_out_proj_bias)
    x = output + residual
    
    # ============ 3. Cross-Attention with Text ============
    residual = x
    
    # Layer norm
    x_norm = F.layer_norm(x, (hidden_size,), cross_attn_norm_weight, cross_attn_norm_bias, eps)
    
    # Query from video latents: (B, F*S, H)
    q = F.linear(x_norm, cross_attn_q_weight, cross_attn_q_bias)
    
    # Key and Value from text embeddings: (B, T, 2*H)
    kv = F.linear(text_embeddings, cross_attn_kv_weight, cross_attn_kv_bias)
    kv = kv.view(batch_size, text_seq_len, 2, hidden_size)
    k, v = kv.chunk(2, dim=2)
    k, v = k.squeeze(2), v.squeeze(2)
    
    # Reshape for multi-head attention
    q = q.view(batch_size, video_seq_len, num_attention_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, text_seq_len, num_attention_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, text_seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    # Compute cross-attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape back: (B, NH, F*S, HD) -> (B, F*S, H)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, video_seq_len, hidden_size)
    
    # Output projection and residual
    output = F.linear(attn_output, cross_attn_out_proj_weight, cross_attn_out_proj_bias)
    x = output + residual
    
    # ============ 4. Feedforward Network ============
    residual = x
    
    # Layer norm
    x_norm = F.layer_norm(x, (hidden_size,), ffn_norm_weight, ffn_norm_bias, eps)
    
    # FFN: H -> I -> H with GELU
    x_ffn = F.linear(x_norm, ffn_fc1_weight, ffn_fc1_bias)
    x_ffn = F.gelu(x_ffn)
    x_ffn = F.linear(x_ffn, ffn_fc2_weight, ffn_fc2_bias)
    
    # Residual connection
    output = x_ffn + residual
    
    return output
