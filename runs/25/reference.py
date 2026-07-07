import torch
import math

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation using tanh approximation.
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute inner term: sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Apply tanh and compute final result
    output = 0.5 * x * (1.0 + torch.tanh(inner))
    
    return output
