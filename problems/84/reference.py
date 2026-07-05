import torch

@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    """
    Backward pass for SiLU activation.
    
    Gradient formula:
        grad_input = grad_output * sigmoid(x) * [1 + x * (1 - sigmoid(x))]
    
    Args:
        grad_output: Upstream gradient, shape [num_elements]
        x: Original input from forward pass, shape [num_elements]
        sigmoid_x: Cached sigmoid(x) from forward pass, shape [num_elements]
    
    Returns:
        grad_input: Gradient with respect to input x, shape [num_elements]
    """
    # Compute: 1 - sigmoid(x)
    one_minus_sigmoid = 1.0 - sigmoid_x
    
    # Compute: x * (1 - sigmoid(x))
    x_times_one_minus_sigmoid = x * one_minus_sigmoid
    
    # Compute: 1 + x * (1 - sigmoid(x))
    bracket_term = 1.0 + x_times_one_minus_sigmoid
    
    # Compute: sigmoid(x) * [1 + x * (1 - sigmoid(x))]
    local_grad = sigmoid_x * bracket_term
    
    # Apply chain rule: grad_output * local_gradient
    grad_input = grad_output * local_grad
    
    return grad_input
