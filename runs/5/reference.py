import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    """
    Complete fused convolution layer with gated projections and causal convolution.
    
    Operation flow:
    1. Triple linear projection: x -> (B, C, x_proj) via in_proj
    2. Element-wise gating: Bx = B * x_proj
    3. Grouped causal 1D convolution on Bx with kernel_size=4
    4. Output gating: y = C * conv_out
    5. Final output projection: y -> out_proj(y)
    """
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]
    
    # Step 1: Triple linear projection
    # Shape: (batch_size, seq_len, 3 * hidden_size)
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    
    # Transpose for conv1d: (batch_size, 3 * hidden_size, seq_len)
    BCx = BCx.transpose(-1, -2)
    
    # Split into B, C, x_proj along channel dimension
    # Each has shape: (batch_size, hidden_size, seq_len)
    B, C, x_proj = BCx.chunk(3, dim=1)
    
    # Step 2: Element-wise gating
    # Shape: (batch_size, hidden_size, seq_len)
    Bx = B * x_proj
    
    # Step 3: Grouped causal 1D convolution
    # Apply conv with causal padding
    # Padding is kernel_size - 1 on the left for causal
    Bx_padded = F.pad(Bx, (conv_kernel_size - 1, 0))
    
    # Grouped conv1d with groups=hidden_size (depthwise)
    conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=hidden_size)
    
    # Shape: (batch_size, hidden_size, seq_len)
    
    # Step 4: Output gating with C
    # Shape: (batch_size, hidden_size, seq_len)
    y = C * conv_out
    
    # Transpose back: (batch_size, seq_len, hidden_size)
    y = y.transpose(-1, -2).contiguous()
    
    # Step 5: Final output projection
    # Shape: (batch_size, seq_len, hidden_size)
    output = F.linear(y, out_proj_weight, out_proj_bias)
    
    return output
