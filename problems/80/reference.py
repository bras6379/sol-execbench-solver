import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    eps: float,
):
    """
    Adaptive Layer Normalization with continuous modulation.
    
    1. Compute mean and variance across the feature dimension
    2. Normalize: (x - mean) / sqrt(variance + eps)
    3. Apply learned linear projection to temb to get scale and shift
    4. Modulate: normalized * (1 + scale) + shift
    
    Args:
        hidden_states: (batch, seq_len, inner_dim) - Input features to normalize
        temb: (batch, inner_dim) - Timestep conditioning embeddings
        linear_weight: (inner_dim*2, inner_dim) - Weight for temb projection
        linear_bias: (inner_dim*2,) - Bias for temb projection
        eps: Epsilon for numerical stability
    
    Returns:
        (batch, seq_len, inner_dim) - Normalized and modulated features
    """
    # Layer normalization: compute mean and variance
    # Shape: (batch, seq_len, inner_dim)
    mean = hidden_states.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
    variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, 1)
    
    # Normalize
    normalized = (hidden_states - mean) / torch.sqrt(variance + eps)
    
    # Generate scale and shift from temb via linear projection
    # temb: (batch, inner_dim) @ linear_weight.T: (inner_dim, inner_dim*2) -> (batch, inner_dim*2)
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    
    # Split into scale and shift
    # Each is (batch, inner_dim)
    inner_dim = hidden_states.shape[-1]
    scale = modulation[:, :inner_dim]
    shift = modulation[:, inner_dim:]
    
    # Apply modulation: normalized * (1 + scale) + shift
    # Unsqueeze scale and shift to broadcast over seq_len dimension
    # (batch, inner_dim) -> (batch, 1, inner_dim)
    scale = scale.unsqueeze(1)
    shift = shift.unsqueeze(1)
    
    # Final modulated output
    output = normalized * (1.0 + scale) + shift
    
    return output
