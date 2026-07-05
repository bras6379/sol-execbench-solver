import torch

@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    """
    Fused attention output reshape and projection.
    
    Args:
        attn_output: [batch_size, num_heads, seq_len, v_head_dim] - attention output
        o_proj_weight: [hidden_size, intermediate_size] - output projection weight
        
    Returns:
        output: [batch_size, seq_len, hidden_size] - projected output
    """
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    hidden_size = o_proj_weight.shape[0]
    intermediate_size = num_heads * v_head_dim
    
    # Step 1: Transpose [batch, num_heads, seq_len, v_head_dim] -> [batch, seq_len, num_heads, v_head_dim]
    # This creates a non-contiguous tensor with strided memory access
    attn_output_transposed = attn_output.transpose(1, 2)
    
    # Step 2: Reshape to [batch, seq_len, num_heads * v_head_dim]
    # The contiguous() call triggers a full memory copy
    attn_output_reshaped = attn_output_transposed.reshape(bsz, seq_len, intermediate_size)
    
    # Step 3: Output projection via matrix multiplication
    # [batch, seq_len, intermediate_size] @ [intermediate_size, hidden_size] -> [batch, seq_len, hidden_size]
    # Using F.linear equivalent: output = input @ weight.T
    output = torch.matmul(attn_output_reshaped, o_proj_weight.t())
    
    return output
