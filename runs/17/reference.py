import torch
import torch.nn.functional as F
import math


def gen_inputs(axes_and_scalars, device):
    num_tokens = axes_and_scalars['num_tokens']
    hidden_size = 2048
    intermediate_size = 768

    # Xavier-scaled weights
    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)
    up_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) / math.sqrt(intermediate_size)

    # Activation-scale input
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)

    # Run forward pass to produce consistent saved tensors
    with torch.no_grad():
        gate = F.linear(x, gate_weight)
        gate_sigmoid = torch.sigmoid(gate.to(torch.float32)).to(torch.bfloat16)
        gate_silu = gate * gate_sigmoid
        up = F.linear(x, up_weight)
        intermediate = gate_silu * up

    # Small-magnitude upstream gradient
    grad_output = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)

    return {
        'grad_output': grad_output,
        'x': x,
        'gate_weight': gate_weight,
        'up_weight': up_weight,
        'down_weight': down_weight,
        'gate': gate,
        'gate_sigmoid': gate_sigmoid,
        'gate_silu': gate_silu,
        'up': up,
        'intermediate': intermediate,
    }


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_sigmoid: torch.Tensor,
    gate_silu: torch.Tensor,
    up: torch.Tensor,
    intermediate: torch.Tensor,
):
    # Convert to float32 for numerical stability
    grad_output_f32 = grad_output.to(torch.float32)
    x_f32 = x.to(torch.float32)
    gate_weight_f32 = gate_weight.to(torch.float32)
    up_weight_f32 = up_weight.to(torch.float32)
    down_weight_f32 = down_weight.to(torch.float32)
    gate_f32 = gate.to(torch.float32)
    gate_sigmoid_f32 = gate_sigmoid.to(torch.float32)
    gate_silu_f32 = gate_silu.to(torch.float32)
    up_f32 = up.to(torch.float32)
    intermediate_f32 = intermediate.to(torch.float32)
    
    # Gradient w.r.t. down_weight
    # down_proj: output = intermediate @ down_weight.T
    # grad_down_weight = grad_output.T @ intermediate
    grad_down_weight = grad_output_f32.t().mm(intermediate_f32)
    
    # Gradient w.r.t. intermediate
    grad_intermediate = grad_output_f32.mm(down_weight_f32)
    
    # Gradient w.r.t. gate_silu and up (element-wise multiplication)
    # intermediate = gate_silu * up
    grad_gate_silu = grad_intermediate * up_f32
    grad_up = grad_intermediate * gate_silu_f32
    
    # Gradient w.r.t. gate (through SiLU activation)
    # gate_silu = gate * sigmoid(gate)
    # d(gate_silu)/d(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    grad_gate = grad_gate_silu * gate_sigmoid_f32 * (1.0 + gate_f32 * (1.0 - gate_sigmoid_f32))
    
    # Gradient w.r.t. up_weight
    # up_proj: up = x @ up_weight.T
    # grad_up_weight = grad_up.T @ x
    grad_up_weight = grad_up.t().mm(x_f32)
    
    # Gradient w.r.t. gate_weight
    # gate_proj: gate = x @ gate_weight.T
    # grad_gate_weight = grad_gate.T @ x
    grad_gate_weight = grad_gate.t().mm(x_f32)
    
    # Gradient w.r.t. x (input)
    # Accumulate gradients from both gate and up projections
    grad_x = grad_gate.mm(gate_weight_f32) + grad_up.mm(up_weight_f32)
    
    return (
        grad_x.to(torch.bfloat16),
        grad_gate_weight.to(torch.bfloat16),
        grad_up_weight.to(torch.bfloat16),
        grad_down_weight.to(torch.bfloat16),
    )
