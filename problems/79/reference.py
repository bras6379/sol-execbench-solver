import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    ssm_output: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gate_activated: torch.Tensor,
    gated_output: torch.Tensor,
    use_silu_gate: bool,
):
    """
    Backward pass for SSM output projection with gated multiplication.
    
    Computes gradients for:
    - ssm_output: input to the gating operation
    - gate: gate values (before activation)
    - weight: projection weight matrix
    - bias: projection bias
    """
    # Get dimensions
    batch_size, seq_len, hidden_dim = grad_output.shape
    expanded_dim = ssm_output.shape[2]
    
    # Reshape for batch operations
    grad_output_2d = grad_output.reshape(-1, hidden_dim)
    gated_output_2d = gated_output.reshape(-1, expanded_dim)
    
    # Gradient w.r.t. weight (projection matrix)
    # d_loss/d_weight = grad_output.T @ gated_output
    # Shape: (hidden_dim, expanded_dim)
    grad_weight = grad_output_2d.t() @ gated_output_2d
    
    # Gradient w.r.t. bias
    # d_loss/d_bias = sum(grad_output, dim=[0, 1])
    # Shape: (hidden_dim,)
    grad_bias = grad_output_2d.sum(dim=0)
    
    # Gradient w.r.t. gated_output (before projection)
    # d_loss/d_gated_output = grad_output @ weight
    # Shape: (batch * seq_len, expanded_dim)
    grad_gated_output_2d = grad_output_2d @ weight
    grad_gated_output = grad_gated_output_2d.view(batch_size, seq_len, expanded_dim)
    
    # Gradient w.r.t. ssm_output
    # gated_output = ssm_output * gate_activated
    # d_loss/d_ssm_output = grad_gated_output * gate_activated
    grad_ssm_output = grad_gated_output * gate_activated
    
    # Gradient w.r.t. gate_activated
    # d_loss/d_gate_activated = grad_gated_output * ssm_output
    grad_gate_activated = grad_gated_output * ssm_output
    
    # Gradient w.r.t. gate (before activation)
    if use_silu_gate:
        # SiLU(x) = x * sigmoid(x)
        # d_SiLU/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sigmoid_gate = torch.sigmoid(gate)
        silu_grad = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        grad_gate = grad_gate_activated * silu_grad
    else:
        # No activation, gradient passes through directly
        grad_gate = grad_gate_activated
    
    return grad_ssm_output, grad_gate, grad_weight, grad_bias
