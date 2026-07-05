import torch

@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Backward pass for fused RoPE application.
    
    Mathematical derivation:
    Given forward: output = (x * cos) + (rotate_half(x) * sin)
    Where: rotate_half(x) = [-x2, x1] for x = [x1, x2]
    
    Gradient w.r.t. x:
        grad_x = grad_output * cos - rotate_half(grad_output) * sin
    
    Gradient w.r.t. cos:
        grad_cos = sum over heads of (grad_output * x)
    
    Gradient w.r.t. sin:
        grad_sin = sum over heads of (grad_output * rotate_half(x))
    """
    half_head_dim = 64
    unsqueeze_dim = 1
    
    # Unsqueeze cos and sin for broadcasting with q and k
    # cos, sin: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
    sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)
    
    # Helper function to rotate half
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half_head_dim]
        x2 = x[..., half_head_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
    # Compute gradient w.r.t. q
    # grad_q = grad_q_embed * cos - rotate_half(grad_q_embed) * sin
    grad_q = (grad_q_embed * cos_unsqueezed) - (rotate_half(grad_q_embed) * sin_unsqueezed)

    # Compute gradient w.r.t. k
    # grad_k = grad_k_embed * cos - rotate_half(grad_k_embed) * sin
    grad_k = (grad_k_embed * cos_unsqueezed) - (rotate_half(grad_k_embed) * sin_unsqueezed)
    
    # Compute gradient w.r.t. cos
    # grad_cos = (grad_q_embed * q) + (grad_k_embed * k), summed over heads
    grad_cos_from_q = grad_q_embed * q
    grad_cos_from_k = grad_k_embed * k
    # Sum over the head dimension (dim=1) to match original cos shape
    grad_cos = grad_cos_from_q.sum(dim=unsqueeze_dim) + grad_cos_from_k.sum(dim=unsqueeze_dim)
    
    # Compute gradient w.r.t. sin
    # grad_sin = (grad_q_embed * rotate_half(q)) + (grad_k_embed * rotate_half(k)), summed over heads
    q_rotated = rotate_half(q)
    k_rotated = rotate_half(k)
    grad_sin_from_q = grad_q_embed * q_rotated
    grad_sin_from_k = grad_k_embed * k_rotated
    # Sum over the head dimension (dim=1) to match original sin shape
    grad_sin = grad_sin_from_q.sum(dim=unsqueeze_dim) + grad_sin_from_k.sum(dim=unsqueeze_dim)
    
    return grad_q, grad_k, grad_cos, grad_sin
