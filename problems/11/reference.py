import torch
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs including precomputed llama3-scaled inverse frequencies."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    head_dim = axes_and_scalars["head_dim"]
    
    # Llama3 RoPE constants
    rope_theta = 500000.0
    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    original_max_position_embeddings = 8192
    
    # Compute inverse frequencies with llama3 scaling
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (rope_theta ** (dim_indices / head_dim))
    
    # Compute wavelengths and scaling factors
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    
    # Wavelength for each frequency component
    wavelens = 2 * math.pi / inv_freq
    
    # Smooth interpolation factor
    smooth_factor = (original_max_position_embeddings / wavelens - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
    
    # Apply scaled frequencies
    scaled_inv_freq = inv_freq / factor
    inv_freq = (1 - smooth_factor) * inv_freq + smooth_factor * scaled_inv_freq
    
    # Generate position_ids (sequential positions for each batch)
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    
    return {
        "position_ids": position_ids,
        "inv_freq": inv_freq,
        "attention_scaling": 1.0
    }


@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float
) -> torch.Tensor:
    """
    Compute rotary position embeddings.
    
    Args:
        position_ids: Position indices (batch_size, seq_len)
        inv_freq: Precomputed inverse frequencies with llama3 scaling (head_dim/2,)
        attention_scaling: Attention scaling factor
    
    Returns:
        cos_sin: Concatenated cos and sin embeddings (batch_size, seq_len, head_dim, 2)
    """
    batch_size = position_ids.shape[0]
    
    # Expand inv_freq for batch dimension: (batch_size, head_dim/2, 1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        batch_size, -1, 1
    )
    
    # Expand position_ids: (batch_size, 1, seq_len)
    position_ids_expanded = position_ids[:, None, :].float()
    
    # Compute frequencies: (batch_size, head_dim/2, seq_len) -> (batch_size, seq_len, head_dim/2)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    
    # Duplicate frequencies for complex number representation
    # (batch_size, seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Compute cos and sin with attention scaling
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    
    # Stack cos and sin along last dimension: (batch_size, seq_len, head_dim, 2)
    cos_sin = torch.stack([cos, sin], dim=-1)
    
    # Convert to bfloat16
    return cos_sin.to(dtype=torch.bfloat16)
