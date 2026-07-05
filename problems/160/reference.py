import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    time_emb: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    time_emb_proj_weight: torch.Tensor,
    time_emb_proj_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    norm_eps: float,
):
    """
    ResNet block with time embedding injection.
    
    Flow:
    1. GroupNorm + SiLU + Conv2d on input
    2. SiLU on time_emb, project and add to features
    3. GroupNorm + SiLU + Conv2d
    4. Residual connection (in_channels == out_channels, so no shortcut conv needed)
    """
    # Store input for residual connection
    residual = x
    
    # First conv block: GroupNorm -> SiLU -> Conv2d
    # GroupNorm with 32 groups
    h = F.group_norm(x, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=norm_eps)
    # SiLU activation: x * sigmoid(x)
    h = h * torch.sigmoid(h)
    # Conv2d with 3x3 kernel, padding=1
    h = F.conv2d(h, conv1_weight, conv1_bias, stride=1, padding=1)
    
    # Time embedding injection
    # Apply SiLU to time embedding
    t = time_emb * torch.sigmoid(time_emb)
    # Project time embedding: Linear(time_emb_channels -> out_channels)
    t = F.linear(t, time_emb_proj_weight, time_emb_proj_bias)
    # Reshape for broadcasting: (batch, out_channels) -> (batch, out_channels, 1, 1)
    t = t[:, :, None, None]
    # Add time embedding to features (default mode, not scale-shift)
    h = h + t
    
    # Second conv block: GroupNorm -> SiLU -> Conv2d
    h = F.group_norm(h, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=norm_eps)
    h = h * torch.sigmoid(h)
    # Dropout with p=0.0 is identity, so we skip it
    h = F.conv2d(h, conv2_weight, conv2_bias, stride=1, padding=1)
    
    # Residual connection (no shortcut conv needed since in_channels == out_channels)
    # out_scale_factor = 1.0, so no scaling needed
    output = h + residual
    
    return output
