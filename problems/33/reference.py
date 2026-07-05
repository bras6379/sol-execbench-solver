import torch

@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Post-normalization residual connection: output = residual + RMSNorm(sublayer_output)
    
    RMSNorm computation:
    1. Compute variance: mean of squared values along hidden dimension
    2. Normalize: x * rsqrt(variance + eps)
    3. Apply learned scale (weight parameter)
    4. Add residual connection
    
    Args:
        sublayer_output: Output from attention or MLP sublayer [batch, seq_len, hidden_size]
        residual: Residual connection input [batch, seq_len, hidden_size]
        weight: Learned scale parameter [hidden_size]
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor with residual added [batch, seq_len, hidden_size]
    """
    # Store input dtype for final conversion
    input_dtype = sublayer_output.dtype
    
    # RMSNorm computation in float32 for numerical stability
    normalized = sublayer_output.to(torch.float32)
    
    # Compute variance: mean of squared values along hidden dimension
    # Shape: [batch, seq_len, 1]
    variance = normalized.pow(2).mean(-1, keepdim=True)
    
    # Normalize: x * rsqrt(variance + eps)
    # rsqrt is more efficient than 1/sqrt
    normalized = normalized * torch.rsqrt(variance + eps)
    
    # Apply learned scale (weight parameter)
    normalized = weight.to(torch.float32) * normalized
    
    # Convert back to input dtype
    normalized = normalized.to(input_dtype)
    
    # Add residual connection
    output = residual + normalized
    
    return output
