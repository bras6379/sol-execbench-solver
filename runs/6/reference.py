import torch
import torch.nn.functional as F

@torch.no_grad()
def run(u: torch.Tensor, short_filter_weight: torch.Tensor, short_filter_bias: torch.Tensor):
    """
    Depthwise 1D convolution with split and gating for Hyena.
    
    Args:
        u: Input tensor (batch, inner_width, seq_len)
        short_filter_weight: Depthwise conv weights (inner_width, 1, kernel_size)
        short_filter_bias: Depthwise conv bias (inner_width,)
    
    Returns:
        v_gated: Gated output v * x[0] (batch, d_model, seq_len)
        x0: First component (batch, d_model, seq_len)
        x1: Second component (batch, d_model, seq_len)
    """
    batch_size, inner_width, seq_len = u.shape
    d_model = 256
    
    # Apply depthwise grouped convolution with padding=2
    # This uses groups=inner_width for depthwise convolution
    uc = F.conv1d(
        u,
        short_filter_weight,
        bias=short_filter_bias,
        padding=2,
        groups=inner_width
    )
    
    # Truncate to original sequence length (padding causes extra length)
    uc = uc[..., :seq_len]
    
    # Split into (order + 1) = 3 components of size d_model each
    # For order=2: x[0], x[1], v
    x0 = uc[:, :d_model, :]
    x1 = uc[:, d_model:2*d_model, :]
    v = uc[:, 2*d_model:3*d_model, :]
    
    # Initial gating: v * x[0]
    v_gated = v * x0
    
    return v_gated, x0, x1
