import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    dt_with_bias: torch.Tensor,
    dt_activated: torch.Tensor,
    time_step_min: float,
    time_step_max: float,
):
    """
    Backward pass for fused dt projection with bias, softplus, and clamping.
    
    Computes gradients through:
    1. Clamp operation: gradient passes through only if value is strictly within bounds
    2. Softplus operation: gradient is sigmoid(dt_with_bias)
    3. Bias addition: gradient passes through for dt, summed over batch/seq for bias
    
    Args:
        grad_output: Gradient from downstream [batch_size, seq_len, num_heads]
        dt_with_bias: Saved tensor dt + dt_bias from forward [batch_size, seq_len, num_heads]
        dt_activated: Saved tensor softplus(dt + dt_bias) from forward [batch_size, seq_len, num_heads]
        time_step_min: Minimum clamp value (0.001)
        time_step_max: Maximum clamp value (0.1)
    
    Returns:
        grad_dt: Gradient w.r.t. dt input [batch_size, seq_len, num_heads]
        grad_dt_bias: Gradient w.r.t. dt_bias [num_heads]
    """
    # Convert to float32 for numerical stability
    grad = grad_output.to(torch.float32)
    dt_with_bias_f32 = dt_with_bias.to(torch.float32)
    dt_activated_f32 = dt_activated.to(torch.float32)
    
    # Step 1: Gradient through clamp operation
    # Gradient passes through only if value is strictly within bounds
    # If clamped to min or max, gradient is zero (non-differentiable boundary)
    clamp_mask = (dt_activated_f32 > time_step_min) & (dt_activated_f32 < time_step_max)
    grad = grad * clamp_mask.to(grad.dtype)
    
    # Step 2: Gradient through softplus operation
    # d/dx softplus(x) = sigmoid(x) = 1 / (1 + exp(-x))
    softplus_grad = torch.sigmoid(dt_with_bias_f32)
    grad = grad * softplus_grad
    
    # Step 3: Gradient through bias addition
    # For dt input: gradient passes through directly (d(x+b)/dx = 1)
    grad_dt = grad.to(torch.bfloat16)
    
    # For dt_bias: sum gradients over batch and sequence dimensions
    # dt_bias shape: [num_heads]
    # grad shape: [batch, seq_len, num_heads] -> [num_heads]
    grad_dt_bias = grad.sum(dim=(0, 1)).to(torch.bfloat16)
    
    return grad_dt, grad_dt_bias
