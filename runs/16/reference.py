import torch

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    """
    Computes inverse frequency tensor for Rotary Position Embeddings.
    
    inv_freq[i] = 1.0 / (rope_theta^(2*i / head_dim))
    for i in [0, 1, 2, ..., head_dim//2 - 1]
    
    This is equivalent to:
    inv_freq = 1.0 / (rope_theta^(arange(0, head_dim, 2) / head_dim))
    
    Args:
        rope_theta: Base frequency for RoPE computation (e.g., 1000000.0)
        
    Returns:
        Inverse frequency tensor of shape (head_dim // 2,) = (64,)
    """
    head_dim = 128
    
    # Create range [0, 2, 4, ..., head_dim-2] = [0, 2, 4, ..., 126]
    # Shape: (head_dim // 2,) = (64,)
    indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device='cuda')
    
    # Compute exponents: indices / head_dim
    # Shape: (64,)
    exponents = indices / float(head_dim)
    
    # Compute theta^exponents
    # Shape: (64,)
    theta_powers = torch.pow(float(rope_theta), exponents)
    
    # Compute inverse: 1.0 / theta^exponents
    # Shape: (64,)
    inv_freq = 1.0 / theta_powers
    
    return inv_freq
