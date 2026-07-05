import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = x.shape
    num_groups = 32
    
    # Store residual
    residual = x
    
    # Group normalization
    # Reshape to (B, num_groups, channels_per_group, H, W)
    channels_per_group = channels // num_groups
    x_grouped = x.view(batch, num_groups, channels_per_group, height, width)
    
    # Compute mean and variance per group
    mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    
    # Normalize
    x_norm = (x_grouped - mean) / torch.sqrt(var + eps)
    
    # Reshape back to (B, C, H, W)
    x_norm = x_norm.view(batch, channels, height, width)
    
    # Apply scale and shift
    x_norm = x_norm * group_norm_weight.view(1, channels, 1, 1) + group_norm_bias.view(1, channels, 1, 1)
    
    # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
    seq_len = height * width
    x_seq = x_norm.view(batch, channels, seq_len)
    x_seq = x_seq.permute(0, 2, 1).contiguous()  # (B, H*W, C)
    
    # Compute Q, K, V projections using linear layers
    # q = x_seq @ query_weight.T + query_bias
    q = torch.matmul(x_seq, query_weight.t()) + query_bias  # (B, H*W, C)
    k = torch.matmul(x_seq, key_weight.t()) + key_bias      # (B, H*W, C)
    v = torch.matmul(x_seq, value_weight.t()) + value_bias  # (B, H*W, C)
    
    # Compute attention scores: Q @ K^T with scaling
    scale = channels ** -0.5
    attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, H*W, H*W)
    
    # Softmax over key dimension
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values
    attn_output = torch.bmm(attn_weights, v)  # (B, H*W, C)
    
    # Output projection
    attn_output = torch.matmul(attn_output, proj_out_weight.t()) + proj_out_bias
    
    # Reshape back to spatial format: (B, H*W, C) -> (B, C, H, W)
    attn_output = attn_output.permute(0, 2, 1).contiguous()
    attn_output = attn_output.view(batch, channels, height, width)
    
    # Residual connection
    output = residual + attn_output
    
    return output
