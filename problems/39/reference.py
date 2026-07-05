import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    """
    Backward pass for Adaptive Instance Normalization 2D.
    
    Computes gradients with respect to input, weight (gamma), and bias (beta).
    
    Args:
        grad_output: Gradient of loss w.r.t. output, shape (N, C, H, W)
        x: Original input tensor from forward pass, shape (N, C, H, W)
        weight: Scale parameter (gamma), shape (C,)
        mean: Mean computed in forward pass, shape (N, C, 1, 1)
        std: Standard deviation computed in forward pass, shape (N, C, 1, 1)
        
    Returns:
        grad_input: Gradient w.r.t. input, shape (N, C, H, W)
        grad_weight: Gradient w.r.t. weight (gamma), shape (C,)
        grad_bias: Gradient w.r.t. bias (beta), shape (C,)
    """
    N, C, H, W = x.shape
    spatial_size = H * W
    
    # Compute centered and normalized input
    x_centered = x - mean  # Shape: (N, C, H, W)
    x_normalized = x_centered / std  # Shape: (N, C, H, W)
    
    # Gradient w.r.t. bias (beta)
    # d(loss)/d(beta) = sum(d(loss)/d(output)) over N, H, W
    grad_bias = grad_output.sum(dim=(0, 2, 3))  # Shape: (C,)
    
    # Gradient w.r.t. weight (gamma)
    # d(loss)/d(gamma) = sum(d(loss)/d(output) * x_normalized) over N, H, W
    grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))  # Shape: (C,)
    
    # Gradient w.r.t. input
    # Scale grad_output by weight
    weight_reshaped = weight.view(1, C, 1, 1)
    grad_output_scaled = grad_output * weight_reshaped
    
    # Gradient w.r.t. variance
    # d(loss)/d(var) = sum(d(loss)/d(x_norm) * (x - mean) * -0.5 * (var + eps)^(-3/2))
    grad_var = (grad_output_scaled * x_centered).sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)
    
    # Gradient w.r.t. mean
    # d(loss)/d(mean) = sum(d(loss)/d(x_norm) * -1/std) + d(loss)/d(var) * sum(-2 * (x - mean)) / (H*W)
    grad_mean = (grad_output_scaled / (-std)).sum(dim=(2, 3), keepdim=True)
    grad_mean = grad_mean + grad_var * (-2.0 * x_centered).sum(dim=(2, 3), keepdim=True) / spatial_size
    
    # Gradient w.r.t. input
    # d(loss)/d(x) = d(loss)/d(x_norm) * 1/std + d(loss)/d(var) * 2*(x-mean)/(H*W) + d(loss)/d(mean) * 1/(H*W)
    grad_input = grad_output_scaled / std
    grad_input = grad_input + grad_var * 2.0 * x_centered / spatial_size
    grad_input = grad_input + grad_mean / spatial_size
    
    return grad_input, grad_weight, grad_bias
