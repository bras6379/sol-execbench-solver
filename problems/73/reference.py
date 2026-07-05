import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    """
    Fused encoder final RMSNorm with cross-attention K/V projection.
    
    Args:
        encoder_hidden_states: (batch_size, encoder_seq_len, 1024)
        norm_weight: (1024,) RMSNorm weight
        k_proj_weight: (128, 1024) Key projection weight
        v_proj_weight: (128, 1024) Value projection weight
        eps: RMSNorm epsilon
    
    Returns:
        keys: (batch_size, 2, encoder_seq_len, 64)
        values: (batch_size, 2, encoder_seq_len, 64)
    """
    batch_size, seq_len, hidden_size = encoder_hidden_states.shape
    num_kv_heads = 2
    head_dim = 64
    
    # RMSNorm computation
    input_dtype = encoder_hidden_states.dtype
    hidden_states = encoder_hidden_states.to(torch.float32)
    
    # Compute variance and normalize
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    normalized = (norm_weight * hidden_states).to(input_dtype)
    
    # K/V projections
    # normalized: (batch, seq_len, 1024)
    # weights: (128, 1024)
    # output: (batch, seq_len, 128)
    keys_flat = F.linear(normalized, k_proj_weight, bias=None)
    values_flat = F.linear(normalized, v_proj_weight, bias=None)
    
    # Reshape to multi-head format
    # (batch, seq_len, 128) -> (batch, seq_len, 2, 64) -> (batch, 2, seq_len, 64)
    keys = keys_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    
    values = values_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    values = values.transpose(1, 2).contiguous()
    
    return keys, values
