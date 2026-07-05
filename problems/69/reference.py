import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Residual addition
    x = residual + hidden_states
    
    # RMSNorm computation in float32 for numerical stability
    x_fp32 = x.to(torch.float32)
    
    # Compute variance (mean of squares) along last dimension
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    
    # Normalize by RMS (root mean square)
    x_normalized = x_fp32 * torch.rsqrt(variance + eps)
    
    # Apply learned scaling and convert back to bfloat16
    output = weight * x_normalized.to(torch.bfloat16)
    
    return output
