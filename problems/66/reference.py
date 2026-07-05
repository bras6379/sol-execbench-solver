import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    """
    Backward pass for masked softmax with dropout.
    
    Computes gradients through three operations in reverse order:
    1. Dropout backward: scale by dropout mask and inverse probability
    2. Softmax backward: y * (grad - sum(y * grad))
    3. Masked fill backward: zero out gradients at masked positions
    
    Args:
        grad_output: Gradient w.r.t. output [B, H, T, T]
        p_attn: Softmax output (before dropout) [B, H, T, T]
        mask: Attention mask [B, 1, T, T] (True=unmasked, False=masked)
        dropout_mask: Dropout mask [B, H, T, T] (True=kept, False=dropped)
        p_dropout: Dropout probability
    
    Returns:
        grad_scores: Gradient w.r.t. input scores [B, H, T, T]
    """
    # Step 1: Gradient through dropout
    # Dropout backward: grad = grad_output * dropout_mask / (1 - p_dropout)
    if p_dropout > 0.0:
        # Scale by dropout mask and inverse keep probability
        grad_softmax_output = grad_output * dropout_mask.float() / (1.0 - p_dropout)
    else:
        # No dropout, gradient passes through unchanged
        grad_softmax_output = grad_output
    
    # Step 2: Gradient through softmax
    # Softmax backward formula: grad_input = y * (grad_output - sum(y * grad_output))
    # This is derived from the Jacobian of softmax:
    # J_ij = y_i * (delta_ij - y_j)
    
    # Compute sum of (y * grad) along the softmax dimension (dim=-1)
    # Shape: [B, H, T, T] -> [B, H, T, 1]
    sum_term = (p_attn * grad_softmax_output).sum(dim=-1, keepdim=True)
    
    # Compute softmax gradient: y * (grad - sum_term)
    # Shape: [B, H, T, T]
    grad_softmax_input = p_attn * (grad_softmax_output - sum_term)
    
    # Step 3: Gradient through masked_fill
    # Masked positions should have zero gradient
    # Only unmasked positions (mask == True) receive gradient flow
    # Zero out gradients at masked positions (mask == False)
    grad_scores = grad_softmax_input.masked_fill(~mask, 0.0)
    
    return grad_scores
