import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_weights: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for fused attention QK matmul + scale + mask + softmax.
    
    Gradient flow:
    grad_output -> softmax_backward -> scale_backward -> matmul_backward -> grad_Q, grad_K
    
    Args:
        grad_output: (batch, num_heads, seq_len_q, seq_len_k) gradient from downstream
        query: (batch, num_heads, seq_len_q, head_dim) saved from forward
        key: (batch, num_heads, seq_len_k, head_dim) saved from forward
        attn_weights: (batch, num_heads, seq_len_q, seq_len_k) softmax output saved from forward
        scaling: float, scaling factor (1/sqrt(head_dim))
    
    Returns:
        grad_query: (batch, num_heads, seq_len_q, head_dim)
        grad_key: (batch, num_heads, seq_len_k, head_dim)
    """
    # Convert to float32 for numerical stability in softmax backward
    grad_output_f32 = grad_output.to(torch.float32)
    attn_weights_f32 = attn_weights.to(torch.float32)
    
    # Step 1: Gradient through softmax
    # For softmax: d_softmax/d_logits = softmax * (grad_out - sum(grad_out * softmax))
    # This is the efficient formulation that avoids computing the full Jacobian
    sum_grad = torch.sum(grad_output_f32 * attn_weights_f32, dim=-1, keepdim=True)
    grad_attn_logits = attn_weights_f32 * (grad_output_f32 - sum_grad)
    
    # Convert back to original dtype
    grad_attn_logits = grad_attn_logits.to(query.dtype)
    
    # Step 2: Gradient through scaling
    # d(x * scaling)/dx = scaling
    grad_scaled_logits = grad_attn_logits * scaling
    
    # Step 3: Gradient through Q @ K^T matrix multiplication
    # Forward: attn_logits = Q @ K^T
    # Backward: grad_Q = grad_attn_logits @ K
    #           grad_K = grad_attn_logits^T @ Q
    
    # grad_Q = grad_scaled_logits @ K
    # Shape: (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, head_dim)
    #     -> (batch, num_heads, seq_len_q, head_dim)
    grad_query = torch.matmul(grad_scaled_logits, key)
    
    # grad_K = grad_scaled_logits^T @ Q
    # Shape: (batch, num_heads, seq_len_k, seq_len_q) @ (batch, num_heads, seq_len_q, head_dim)
    #     -> (batch, num_heads, seq_len_k, head_dim)
    grad_key = torch.matmul(grad_scaled_logits.transpose(-2, -1), query)
    
    return grad_query, grad_key
