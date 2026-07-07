import torch

@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    """
    Multi-axis RoPE frequency computation.
    
    Args:
        ids: Position IDs of shape [seq_len, 3] containing [time, height, width] positions
        theta: Base frequency for RoPE computation
    
    Returns:
        Tuple of (freqs_cos, freqs_sin) each of shape [seq_len, total_dim]
        where total_dim = 16 + 56 + 56 = 128
    """
    # Fixed axis dimensions for Flux
    axes_dim = [16, 56, 56]
    n_axes = 3
    
    seq_len = ids.shape[0]
    device = ids.device
    
    freqs_dtype = torch.float32
    
    cos_list = []
    sin_list = []
    
    # Convert ids to float for computation
    pos = ids.float()
    
    # Process each axis independently
    for axis_idx in range(n_axes):
        dim = axes_dim[axis_idx]
        
        # Get positions for this axis: [seq_len]
        axis_pos = pos[:, axis_idx]
        
        # Compute frequency bands for this dimension
        # freq_bands shape: [dim // 2]
        half_dim = dim // 2
        freq_exponents = torch.arange(half_dim, dtype=freqs_dtype, device=device)
        freq_bands = 1.0 / (theta ** (freq_exponents / half_dim))
        
        # Compute angles: [seq_len, dim // 2]
        # Outer product of positions and frequency bands
        angles = axis_pos.unsqueeze(-1).to(freqs_dtype) * freq_bands.unsqueeze(0)
        
        # Compute cos and sin
        cos_angles = torch.cos(angles)  # [seq_len, dim // 2]
        sin_angles = torch.sin(angles)  # [seq_len, dim // 2]
        
        # Repeat interleave to match real RoPE format
        # Each frequency is repeated twice: [f1, f1, f2, f2, ...]
        cos_interleaved = torch.repeat_interleave(cos_angles, 2, dim=-1)  # [seq_len, dim]
        sin_interleaved = torch.repeat_interleave(sin_angles, 2, dim=-1)  # [seq_len, dim]
        
        cos_list.append(cos_interleaved)
        sin_list.append(sin_interleaved)
    
    # Concatenate all axes: [seq_len, total_dim]
    freqs_cos = torch.cat(cos_list, dim=-1)
    freqs_sin = torch.cat(sin_list, dim=-1)
    
    return freqs_cos, freqs_sin
