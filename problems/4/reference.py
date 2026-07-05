import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for attention output projection with reshape.
    
    Forward was:
        1. Transpose: (B, H, S, D) -> (B, S, H, D)
        2. Reshape: (B, S, H, D) -> (B, S, H*D)
        3. Linear: y = x @ W^T
    
    Backward computes:
        - grad_weight = grad_output^T @ reshaped
        - grad_attn_output = (grad_output @ W^T) reshaped and transposed back
    """
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    head_dim = 64
    
    # Convert to float32 for numerical stability
    grad_output_f32 = grad_output.to(torch.float32)
    reshaped_f32 = reshaped.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    
    # Gradient w.r.t. weight
    # d_loss/d_W = grad_output^T @ reshaped
    # grad_output: (B, S, H*D), reshaped: (B, S, H*D)
    # Reshape to 2D: (B*S, H*D)
    grad_output_2d = grad_output_f32.reshape(-1, hidden_size)  # (B*S, H*D)
    reshaped_2d = reshaped_f32.reshape(-1, hidden_size)  # (B*S, H*D)
    
    # Matrix multiply: (H*D, B*S) @ (B*S, H*D) -> (H*D, H*D)
    grad_weight = grad_output_2d.t().mm(reshaped_2d)
    
    # Gradient w.r.t. input (attn_output)
    # d_loss/d_reshaped = grad_output @ W
    # grad_output shape: (B*S, H*D), weight shape: (H*D, H*D)
    grad_reshaped_2d = grad_output_2d.mm(weight_f32)  # (B*S, H*D)
    grad_reshaped = grad_reshaped_2d.reshape(batch_size, seq_len, hidden_size)
    
    # Backward through reshape: (B, S, H*D) -> (B, S, H, D)
    grad_transposed = grad_reshaped.reshape(batch_size, seq_len, num_heads, head_dim)
    
    # Backward through transpose: (B, S, H, D) -> (B, H, S, D)
    grad_attn_output = grad_transposed.transpose(1, 2).contiguous()
    
    return grad_attn_output.to(torch.bfloat16), grad_weight.to(torch.bfloat16)
