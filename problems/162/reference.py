import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs with properly constrained tanh_inner values."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    dim = axes_and_scalars["dim"]
    hidden_dim = axes_and_scalars["hidden_dim"]

    grad_output = torch.randn(batch_size, seq_len, dim, device=device)
    hidden_states = torch.randn(batch_size, seq_len, dim, device=device)
    weight1 = torch.randn(hidden_dim, dim, device=device) * 0.02
    weight2 = torch.randn(dim, hidden_dim, device=device) * 0.02

    # Compute linear1_out from forward pass
    linear1_out = hidden_states.matmul(weight1.t())

    # Compute tanh_inner from forward pass (must be in [-1, 1])
    inner = 0.7978845608028654 * (linear1_out + 0.044715 * linear1_out.pow(3))
    tanh_inner = torch.tanh(inner)

    # Compute gelu_out from forward pass
    gelu_out = 0.5 * linear1_out * (1.0 + tanh_inner)

    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "weight1": weight1,
        "weight2": weight2,
        "linear1_out": linear1_out,
        "tanh_inner": tanh_inner,
        "gelu_out": gelu_out,
    }


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    linear1_out: torch.Tensor,
    tanh_inner: torch.Tensor,
    gelu_out: torch.Tensor,
):
    """
    Backward pass for GELU approximate feedforward.

    Gradient flow:
    grad_output -> grad_linear2 -> grad_gelu -> grad_linear1 -> grad_input

    GELU(x) = 0.5 * x * (1 + tanh(inner))
    where inner = sqrt(2/pi) * (x + 0.044715 * x^3)

    d(GELU)/dx = 0.5 * (1 + tanh(inner)) +
                 0.5 * x * sech^2(inner) * d(inner)/dx

    d(inner)/dx = sqrt(2/pi) * (1 + 0.134145 * x^2)
    """
    # ============================================================
    # Backward through Linear2: output = gelu_out @ weight2.T
    # ============================================================

    # Gradient w.r.t. gelu_out
    grad_gelu_out = grad_output.matmul(weight2)

    # Gradient w.r.t. weight2
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    gelu_out_2d = gelu_out.reshape(-1, gelu_out.shape[-1])
    grad_weight2 = grad_output_2d.t().matmul(gelu_out_2d)

    # ============================================================
    # Backward through GELU approximate activation
    # ============================================================

    # Part 1: 0.5 * (1 + tanh(inner))
    gelu_grad_part1 = 0.5 * (1.0 + tanh_inner)

    # Part 2: 0.5 * x * sech^2(inner) * d(inner)/dx
    sech_squared = 1.0 - tanh_inner * tanh_inner
    d_inner_dx = 0.7978845608028654 * (1.0 + 0.134145 * linear1_out * linear1_out)
    gelu_grad_part2 = 0.5 * linear1_out * sech_squared * d_inner_dx

    # Total GELU gradient
    gelu_grad = gelu_grad_part1 + gelu_grad_part2

    # Apply chain rule
    grad_linear1_out = grad_gelu_out * gelu_grad

    # ============================================================
    # Backward through Linear1: linear1_out = hidden_states @ weight1.T
    # ============================================================

    # Gradient w.r.t. hidden_states
    grad_hidden_states = grad_linear1_out.matmul(weight1)

    # Gradient w.r.t. weight1
    grad_linear1_out_2d = grad_linear1_out.reshape(-1, grad_linear1_out.shape[-1])
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2
