import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """
    GEGLU activation: GELU(x_gate) * x_linear
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, inner_dim * 2)
        
    Returns:
        Output tensor of shape (batch_size, seq_len, inner_dim)
    """
    # Split input into two halves along last dimension
    # x_gate and x_linear each have shape (batch_size, seq_len, inner_dim)
    x_gate, x_linear = x.chunk(2, dim=-1)
    
    # Apply approximate GELU (tanh-based) to gate and multiply with linear part
    # GELU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    output = F.gelu(x_gate, approximate='tanh') * x_linear
    
    return output
