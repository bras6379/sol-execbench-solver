import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    # hidden_states: (batch_size, seq_len, hidden_size)
    # gate_up_weight: (2 * intermediate_size, hidden_size)
    # down_weight: (hidden_size, intermediate_size)
    
    # Fused gate and up projection: (B, S, H) @ (H, 2*I) -> (B, S, 2*I)
    up_states = F.linear(hidden_states, gate_up_weight)
    
    # Split into gate and up components along last dimension
    # Each has shape (B, S, I)
    gate, up_states = up_states.chunk(2, dim=-1)
    
    # Apply SiLU gating: SiLU(x) = x * sigmoid(x)
    # Then multiply: up_states = up_states * silu(gate)
    silu_gate = gate * torch.sigmoid(gate)
    up_states = up_states * silu_gate
    
    # Down projection: (B, S, I) @ (I, H) -> (B, S, H)
    output = F.linear(up_states, down_weight)
    
    return output
