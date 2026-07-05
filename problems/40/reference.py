import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    conv_in_weight: torch.Tensor,
    conv_in_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    conv_out_bias: torch.Tensor,
):
    """
    Convolutional residual block:
    1. conv_in: (B, 3, H, W) -> (B, 32, H, W)
    2. conv_out: (B, 32, H, W) -> (B, 3, H, W)
    3. residual add: output + input
    """
    # Store input for residual connection
    identity = x
    
    # Feature extraction convolution: (B, 3, H, W) -> (B, 32, H, W)
    out = F.conv2d(x, conv_in_weight, conv_in_bias, padding=1)
    
    # Feature reconstruction convolution: (B, 32, H, W) -> (B, 3, H, W)
    out = F.conv2d(out, conv_out_weight, conv_out_bias, padding=1)
    
    # Residual connection: element-wise addition
    out = out + identity
    
    return out
