import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    total_patches = axes_and_scalars['total_patches']
    head_dim = axes_and_scalars['head_dim']
    head_dim_quarter = axes_and_scalars['head_dim_quarter']
    max_grid_size = axes_and_scalars['max_grid_size']
    
    # Gradients from upstream
    grad_cos = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    grad_sin = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    
    # Position IDs - random valid indices into frequency table
    pos_ids = torch.randint(0, max_grid_size, (total_patches, 2), device=device, dtype=torch.int64)
    
    # Inverse frequencies (learnable parameter)
    inv_freq = torch.randn(head_dim_quarter, device=device, dtype=torch.float32)
    
    # Saved embedding from forward pass (random values representing actual embeddings)
    emb = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    
    return {
        'grad_cos': grad_cos,
        'grad_sin': grad_sin,
        'pos_ids': pos_ids,
        'inv_freq': inv_freq,
        'emb': emb
    }

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor
) -> torch.Tensor:
    """Backward pass for vision rotary position embedding generation.
    
    Computes gradient w.r.t. inverse frequencies through:
    1. Gradient through cos/sin: grad_emb = -grad_cos * sin(emb) + grad_sin * cos(emb)
    2. Gradient through concatenation: split and sum
    3. Gradient through indexing: scatter via index_add_
    4. Gradient through outer product: matrix multiply with seq
    
    Args:
        grad_cos: [total_patches, head_dim] gradient w.r.t. cosine output
        grad_sin: [total_patches, head_dim] gradient w.r.t. sine output
        pos_ids: [total_patches, 2] position indices for h and w
        inv_freq: [head_dim//4] inverse frequency tensor
        emb: [total_patches, head_dim] saved embedding from forward
        
    Returns:
        grad_inv_freq: [head_dim//4] gradient w.r.t. inverse frequencies
    """
    total_patches = grad_cos.shape[0]
    head_dim = grad_cos.shape[1]
    head_dim_quarter = inv_freq.shape[0]
    max_grid_size = pos_ids.max().item() + 1
    
    # Step 1: Gradient through cos and sin operations
    # d(cos(x))/dx = -sin(x), d(sin(x))/dx = cos(x)
    grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()
    
    # Step 2: Gradient through concatenation
    # Forward: emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    # Backward: split and sum
    grad_rotary_pos_emb = grad_emb[:, :head_dim//2] + grad_emb[:, head_dim//2:]
    
    # Step 3: Gradient through flatten operation
    # Forward: rotary_pos_emb = freqs[pos_ids].flatten(1)
    # Backward: unflatten
    grad_freqs_indexed = grad_rotary_pos_emb.reshape(total_patches, 2, head_dim_quarter)
    
    # Step 4: Gradient through indexing operation
    # Scatter gradients back to full freqs tensor
    grad_freqs = torch.zeros(
        max_grid_size, head_dim_quarter,
        device=inv_freq.device,
        dtype=inv_freq.dtype
    )
    
    # Scatter gradients for height positions
    grad_freqs.index_add_(0, pos_ids[:, 0], grad_freqs_indexed[:, 0, :])
    
    # Scatter gradients for width positions
    grad_freqs.index_add_(0, pos_ids[:, 1], grad_freqs_indexed[:, 1, :])
    
    # Step 5: Gradient through outer product
    # Forward: freqs = torch.outer(seq, inv_freq)
    # Backward: grad_inv_freq = seq^T @ grad_freqs
    seq = torch.arange(
        max_grid_size,
        device=inv_freq.device,
        dtype=inv_freq.dtype
    )
    grad_inv_freq = torch.matmul(seq, grad_freqs)
    
    return grad_inv_freq
