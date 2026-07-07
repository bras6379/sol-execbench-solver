import torch

@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    embeddings: torch.Tensor,
):
    """
    Backward pass for rotary embedding application.
    
    Forward was:
        emb = cat(embeddings, embeddings, dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        q_rotated = rotate_half(q)  # [x1, x2] -> [-x2, x1]
        k_rotated = rotate_half(k)
        q_embed = q * cos + q_rotated * sin
        k_embed = k * cos + k_rotated * sin
    
    Backward computes gradients w.r.t. q, k, and embeddings.
    """
    seq_len, num_heads, head_dim = q.shape
    
    # Recompute forward intermediates
    emb = torch.cat((embeddings, embeddings), dim=-1)  # (seq_len, head_dim * 2)
    cos_full = emb.cos()  # (seq_len, head_dim * 2)
    sin_full = emb.sin()  # (seq_len, head_dim * 2)
    
    # Slice to head_dim and add head dimension
    cos = cos_full[..., :head_dim].unsqueeze(-2)  # (seq_len, 1, head_dim)
    sin = sin_full[..., :head_dim].unsqueeze(-2)  # (seq_len, 1, head_dim)
    
    # rotate_half helper
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # rotate_half_backward: adjoint of rotate_half
    # rotate_half: [x1, x2] -> [-x2, x1]
    # backward: [g1, g2] -> [g2, -g1]
    def rotate_half_backward(grad_output):
        half_dim = grad_output.shape[-1] // 2
        g1 = grad_output[..., :half_dim]
        g2 = grad_output[..., half_dim:]
        return torch.cat((g2, -g1), dim=-1)
    
    # Recompute rotated versions
    q_rotated = rotate_half(q)  # (seq_len, num_heads, head_dim)
    k_rotated = rotate_half(k)  # (seq_len, num_heads, head_dim)
    
    # Gradient w.r.t. q
    # q_embed = q * cos + rotate_half(q) * sin
    # d(q_embed)/d(q) involves both direct term and rotate_half term
    # grad_q = grad_q_embed * cos + rotate_half_backward(grad_q_embed * sin)
    grad_q = grad_q_embed * cos + rotate_half_backward(grad_q_embed * sin)
    
    # Gradient w.r.t. k (same pattern)
    grad_k = grad_k_embed * cos + rotate_half_backward(grad_k_embed * sin)
    
    # Gradient w.r.t. embeddings
    # q_embed = q * cos(emb) + q_rotated * sin(emb)
    # d(q_embed)/d(cos) = q
    # d(q_embed)/d(sin) = q_rotated
    # d(cos(emb))/d(emb) = -sin(emb)
    # d(sin(emb))/d(emb) = cos(emb)
    
    # Gradient w.r.t. cos and sin (before trig backward)
    grad_cos_q = grad_q_embed * q  # (seq_len, num_heads, head_dim)
    grad_sin_q = grad_q_embed * q_rotated
    
    grad_cos_k = grad_k_embed * k
    grad_sin_k = grad_k_embed * k_rotated
    
    # Sum contributions from q and k
    grad_cos = grad_cos_q + grad_cos_k  # (seq_len, num_heads, head_dim)
    grad_sin = grad_sin_q + grad_sin_k
    
    # Sum over num_heads dimension
    grad_cos = grad_cos.sum(dim=-2)  # (seq_len, head_dim)
    grad_sin = grad_sin.sum(dim=-2)
    
    # Backprop through cos and sin functions
    # d(cos(emb))/d(emb) = -sin(emb)
    # d(sin(emb))/d(emb) = cos(emb)
    emb_sin = sin_full[..., :head_dim]  # (seq_len, head_dim)
    emb_cos = cos_full[..., :head_dim]  # (seq_len, head_dim)
    
    grad_emb_from_cos = grad_cos * (-emb_sin)
    grad_emb_from_sin = grad_sin * emb_cos
    
    grad_embeddings = grad_emb_from_cos + grad_emb_from_sin  # (seq_len, head_dim)
    
    return grad_q, grad_k, grad_embeddings
