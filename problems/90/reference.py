import torch

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for batched 2D RoPE position encoding.
    
    Computes gradient w.r.t. idx_theta using chain rule:
    - d(cos(x))/dx = -sin(x)
    - d(sin(x))/dx = cos(x)
    
    Therefore:
    grad_idx_theta = -grad_cos * sin(idx_theta) + grad_sin * cos(idx_theta)
    
    Note: In practice, this gradient doesn't propagate further since:
    - idx_theta = position * theta
    - position: discrete indices (non-differentiable)
    - theta: derived from configuration constants (non-differentiable)
    
    Args:
        grad_cos: Gradient w.r.t. cos output [batch_size, seq_len, head_dim]
        grad_sin: Gradient w.r.t. sin output [batch_size, seq_len, head_dim]
        idx_theta: Saved angles from forward pass [batch_size, seq_len, head_dim]
        
    Returns:
        grad_idx_theta: Gradient w.r.t. idx_theta [batch_size, seq_len, head_dim]
    """
    # Compute sin and cos of saved angles
    sin_theta = torch.sin(idx_theta)  # [batch_size, seq_len, head_dim]
    cos_theta = torch.cos(idx_theta)  # [batch_size, seq_len, head_dim]
    
    # Convert gradients to float32 for computation
    grad_cos_f32 = grad_cos.to(torch.float32)
    grad_sin_f32 = grad_sin.to(torch.float32)
    
    # Apply chain rule:
    # d(loss)/d(idx_theta) = d(loss)/d(cos) * d(cos)/d(idx_theta) + d(loss)/d(sin) * d(sin)/d(idx_theta)
    # d(cos)/d(idx_theta) = -sin(idx_theta)
    # d(sin)/d(idx_theta) = cos(idx_theta)
    grad_idx_theta = -grad_cos_f32 * sin_theta + grad_sin_f32 * cos_theta
    
    return grad_idx_theta
