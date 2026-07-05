import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
):
    """
    Fused residual + RMSNorm + SwiGLU MLP.
    
    Args:
        hidden_states: Output from attention block [batch_size, seq_len, hidden_size]
        residual: Residual connection from before attention [batch_size, seq_len, hidden_size]
        norm_weight: RMSNorm weight [hidden_size]
        gate_proj_weight: Gate projection weight [intermediate_size, hidden_size]
        up_proj_weight: Up projection weight [intermediate_size, hidden_size]
        down_proj_weight: Down projection weight [hidden_size, intermediate_size]
        eps: Epsilon for numerical stability
        
    Returns:
        Output tensor after MLP [batch_size, seq_len, hidden_size]
    """
    # Step 1: Residual connection
    # Shape: [batch_size, seq_len, hidden_size]
    hidden_states = residual + hidden_states
    
    # Step 2: RMSNorm
    # Convert to float32 for numerical stability
    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    
    # Compute variance: mean of squared values
    # Shape: [batch_size, seq_len, 1]
    variance = hidden_states_f32.pow(2).mean(dim=-1, keepdim=True)
    
    # Normalize and scale
    # Shape: [batch_size, seq_len, hidden_size]
    hidden_states_f32 = hidden_states_f32 * torch.rsqrt(variance + eps)
    hidden_states = (norm_weight * hidden_states_f32).to(input_dtype)
    
    # Step 3: SwiGLU MLP
    # Gate projection: [batch_size, seq_len, hidden_size] @ [hidden_size, intermediate_size]
    # Shape: [batch_size, seq_len, intermediate_size]
    gate_output = F.linear(hidden_states, gate_proj_weight)
    
    # Up projection
    # Shape: [batch_size, seq_len, intermediate_size]
    up_output = F.linear(hidden_states, up_proj_weight)
    
    # SwiGLU activation: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    # Shape: [batch_size, seq_len, intermediate_size]
    intermediate = F.silu(gate_output) * up_output
    
    # Down projection
    # Shape: [batch_size, seq_len, hidden_size]
    output = F.linear(intermediate, down_proj_weight)
    
    return output
