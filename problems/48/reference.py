import torch
import math

@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    """
    Fused gate and up projection with GELU-tanh activation.
    
    Computes: gelu_tanh(x @ gate_proj.T) * (x @ up_proj.T)
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        gate_proj: Gate projection weights of shape (intermediate_size, hidden_size)
        up_proj: Up projection weights of shape (intermediate_size, hidden_size)
    
    Returns:
        Output tensor of shape (batch_size, seq_len, intermediate_size)
    """
    # Compute gate projection: x @ gate_proj.T
    # x: (batch_size, seq_len, hidden_size)
    # gate_proj: (intermediate_size, hidden_size)
    # gate_output: (batch_size, seq_len, intermediate_size)
    gate_output = torch.matmul(x, gate_proj.t())
    
    # Compute up projection: x @ up_proj.T
    # up_output: (batch_size, seq_len, intermediate_size)
    up_output = torch.matmul(x, up_proj.t())
    
    # Apply GELU with tanh approximation to gate output
    # GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    gate_float = gate_output.to(torch.float32)
    inner = sqrt_2_over_pi * (gate_float + 0.044715 * gate_float.pow(3))
    activated_gate = 0.5 * gate_float * (1.0 + torch.tanh(inner))
    activated_gate = activated_gate.to(gate_output.dtype)
    
    # Element-wise multiplication: activated_gate * up_output
    output = activated_gate * up_output
    
    return output
