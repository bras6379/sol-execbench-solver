import torch

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    GroupNorm operation.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Scale parameter of shape (C,)
        bias: Shift parameter of shape (C,)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    num_groups = 32
    
    # Reshape to separate groups: (B, num_groups, C//num_groups, H, W)
    x_grouped = x.view(B, num_groups, C // num_groups, H, W)
    
    # Compute mean and variance per group using float32 for numerical stability
    x_grouped_f32 = x_grouped.to(torch.float32)
    mean = x_grouped_f32.mean(dim=[2, 3, 4], keepdim=True)
    var = x_grouped_f32.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
    
    # Normalize: (x - mean) / sqrt(var + eps)
    x_normalized = (x_grouped_f32 - mean) / torch.sqrt(var + eps)
    
    # Reshape back to (B, C, H, W)
    x_normalized = x_normalized.view(B, C, H, W)
    
    # Apply affine transformation
    # Reshape weight and bias for broadcasting: (1, C, 1, 1)
    weight_reshaped = weight.view(1, C, 1, 1)
    bias_reshaped = bias.view(1, C, 1, 1)
    output = x_normalized * weight_reshaped + bias_reshaped
    
    return output.to(x.dtype)
