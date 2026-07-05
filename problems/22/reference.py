import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_state: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    fc1_output: torch.Tensor,
    gelu_output: torch.Tensor,
):
    """
    Backward pass for Vision MLP with GELU activation.
    
    Computes gradients through:
    1. FC2 backward: grad_output -> grad_fc2_weight, grad_fc2_bias, grad_gelu_output
    2. GELU backward: grad_gelu_output -> grad_fc1_output
    3. FC1 backward: grad_fc1_output -> grad_fc1_weight, grad_fc1_bias, grad_hidden_state
    """
    # Backward through second linear layer (fc2)
    # grad_fc2_bias = sum over batch dimension
    grad_fc2_bias = grad_output.sum(dim=0)
    
    # grad_fc2_weight = grad_output^T @ gelu_output
    # Shape: (hidden_size, seq_len) @ (seq_len, intermediate_size) = (hidden_size, intermediate_size)
    grad_fc2_weight = grad_output.t().mm(gelu_output)
    
    # Gradient w.r.t. gelu_output (input to fc2)
    # grad_gelu_output = grad_output @ fc2_weight
    # Shape: (seq_len, hidden_size) @ (hidden_size, intermediate_size) = (seq_len, intermediate_size)
    grad_gelu_output = grad_output.mm(fc2_weight)
    
    # Backward through GELU activation using tanh approximation derivative
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    const = 0.044715
    
    x = fc1_output
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + const * x_cubed)
    tanh_out = torch.tanh(tanh_arg)
    
    # GELU derivative: 0.5 * (1 + tanh(arg)) + 0.5 * x * sech^2(arg) * d(arg)/dx
    # sech^2(x) = 1 - tanh^2(x)
    sech_sq = 1.0 - tanh_out * tanh_out
    d_tanh_arg = sqrt_2_over_pi * (1.0 + 3.0 * const * x * x)
    
    gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * d_tanh_arg
    
    # Apply chain rule: grad w.r.t. fc1_output
    grad_fc1_output = grad_gelu_output * gelu_grad
    
    # Backward through first linear layer (fc1)
    # grad_fc1_bias = sum over batch dimension
    grad_fc1_bias = grad_fc1_output.sum(dim=0)
    
    # grad_fc1_weight = grad_fc1_output^T @ hidden_state
    # Shape: (intermediate_size, seq_len) @ (seq_len, hidden_size) = (intermediate_size, hidden_size)
    grad_fc1_weight = grad_fc1_output.t().mm(hidden_state)
    
    # grad_hidden_state = grad_fc1_output @ fc1_weight
    # Shape: (seq_len, intermediate_size) @ (intermediate_size, hidden_size) = (seq_len, hidden_size)
    grad_hidden_state = grad_fc1_output.mm(fc1_weight)
    
    return grad_hidden_state, grad_fc1_weight, grad_fc1_bias, grad_fc2_weight, grad_fc2_bias
