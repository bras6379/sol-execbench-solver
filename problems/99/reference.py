import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    activated_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    """
    Backward pass for SwiGLU MLP.
    
    Gradient flow:
    1. grad_output -> through down_proj
    2. grad_gated_output -> through element-wise multiply
    3. grad_activated_gate -> through SiLU
    4. grad_gate_output -> through gate_proj
    5. grad_up_output -> through up_proj
    6. Accumulate grad_x from both gate and up paths
    """
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = gate_output.shape[-1]
    
    # Step 1: Gradient through down_proj
    # output = gated_output @ down_weight.T
    # grad_gated_output = grad_output @ down_weight
    grad_gated_output = grad_output.matmul(down_weight)  # [B, S, I]
    
    # grad_down_weight = grad_output.T @ gated_output
    grad_output_2d = grad_output.reshape(-1, hidden_size)  # [B*S, H]
    gated_output = activated_gate * up_output  # Recompute
    gated_output_2d = gated_output.reshape(-1, intermediate_size)  # [B*S, I]
    grad_down_weight = grad_output_2d.t().matmul(gated_output_2d)  # [H, I]
    
    # Step 2: Gradient through element-wise multiply
    # gated_output = activated_gate * up_output
    grad_activated_gate = grad_gated_output * up_output  # [B, S, I]
    grad_up_output = grad_gated_output * activated_gate  # [B, S, I]
    
    # Step 3: Gradient through SiLU activation
    # activated_gate = silu(gate_output) = gate_output * sigmoid(gate_output)
    # d/dx[silu(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sigmoid_gate = torch.sigmoid(gate_output.to(torch.float32))  # [B, S, I]
    gate_output_f32 = gate_output.to(torch.float32)
    silu_grad = sigmoid_gate * (1.0 + gate_output_f32 * (1.0 - sigmoid_gate))
    grad_gate_output = (grad_activated_gate.to(torch.float32) * silu_grad).to(grad_output.dtype)  # [B, S, I]
    
    # Step 4: Gradient through gate_proj
    # gate_output = x @ gate_weight.T
    # grad_x_gate = grad_gate_output @ gate_weight
    grad_x_gate = grad_gate_output.matmul(gate_weight)  # [B, S, H]
    
    # grad_gate_weight = grad_gate_output.T @ x
    grad_gate_output_2d = grad_gate_output.reshape(-1, intermediate_size)  # [B*S, I]
    x_2d = x.reshape(-1, hidden_size)  # [B*S, H]
    grad_gate_weight = grad_gate_output_2d.t().matmul(x_2d)  # [I, H]
    
    # Step 5: Gradient through up_proj
    # up_output = x @ up_weight.T
    # grad_x_up = grad_up_output @ up_weight
    grad_x_up = grad_up_output.matmul(up_weight)  # [B, S, H]
    
    # grad_up_weight = grad_up_output.T @ x
    grad_up_output_2d = grad_up_output.reshape(-1, intermediate_size)  # [B*S, I]
    grad_up_weight = grad_up_output_2d.t().matmul(x_2d)  # [I, H]
    
    # Step 6: Accumulate input gradients from both paths
    grad_x = grad_x_gate + grad_x_up  # [B, S, H]
    
    return grad_x, grad_gate_weight, grad_up_weight, grad_down_weight
