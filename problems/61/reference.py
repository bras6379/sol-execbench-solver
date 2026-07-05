import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Backward pass for tanh-gated residual addition.
    
    Forward was: output = residual + tanh(gate) * hidden_states * mask
    
    Gradients:
    - grad_residual = grad_output (identity)
    - grad_hidden_states = grad_output * tanh(gate) * mask
    - grad_gate = sum(grad_output * hidden_states * mask * sech^2(gate))
              = sum(grad_output * hidden_states * mask * (1 - tanh^2(gate)))
    """
    # Compute tanh(gate) for gradient computation
    gate_float = gate.to(torch.float32)
    gate_value = torch.tanh(gate_float)
    
    # Gradient w.r.t. residual: dy/d(residual) = 1
    grad_residual = grad_output.clone()
    
    # Gradient w.r.t. hidden_states: dy/d(hidden_states) = tanh(gate) * mask
    grad_hidden_states = grad_output * gate_value * mask
    
    # Gradient w.r.t. gate: dy/d(gate) = sum(hidden_states * mask * sech^2(gate) * grad_output)
    # Using identity: sech^2(x) = 1 - tanh^2(x)
    sech_squared = 1.0 - gate_value * gate_value
    
    # Apply mask to hidden_states
    masked_hidden_states = hidden_states * mask
    
    # Compute gradient: sum over all elements
    grad_gate = torch.sum(grad_output.to(torch.float32) * masked_hidden_states.to(torch.float32)) * sech_squared
    
    return grad_residual.to(torch.bfloat16), grad_hidden_states.to(torch.bfloat16), grad_gate.to(torch.bfloat16)
