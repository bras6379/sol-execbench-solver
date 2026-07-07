import torch

@torch.no_grad()
def run(
    batch_size_scalar: int,
    seq_length_scalar: int,
    past_key_values_length_scalar: int,
):
    """
    Hybrid attention mask preparation for models with mixed full and sliding window attention.
    
    Creates two types of causal masks:
    1. Full causal mask: Standard lower-triangular mask for full attention layers
    2. Sliding window causal mask: Banded diagonal mask for sliding window attention layers
    """
    # Constants
    num_attention_heads = 64
    swa_num_attention_heads = 64
    sliding_window = 128
    dtype = torch.bool
    device = torch.device('cuda')
    
    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past_key_values_length = int(past_key_values_length_scalar)
    
    target_length = seq_length
    source_length = seq_length + past_key_values_length
    
    # Create full causal mask
    full_mask = torch.ones(
        (target_length, source_length),
        dtype=dtype,
        device=device,
    )
    
    # Make it causal (lower triangular)
    target_indices = torch.arange(target_length, device=device)[:, None]
    source_indices = torch.arange(source_length, device=device)[None, :]
    causal_cond = target_indices >= (source_indices - past_key_values_length)
    full_mask = full_mask.masked_fill(causal_cond, False)
    
    # Expand to [batch_size, num_heads, seq_length, source_length]
    full_attention_mask = full_mask[None, None, :, :].expand(
        batch_size, num_attention_heads, target_length, source_length
    ).contiguous()
    
    # Create sliding window causal mask
    swa_mask = torch.zeros(
        (target_length, source_length),
        dtype=dtype,
        device=device,
    )
    
    # Sliding window condition: within window size
    window_cond = (source_indices - past_key_values_length) >= (target_indices - sliding_window)
    
    # Combine conditions: must be both causal and within window
    valid_positions = causal_cond & window_cond
    swa_mask = swa_mask.masked_fill(valid_positions, False)
    
    # Expand to [batch_size, num_heads, seq_length, source_length]
    sliding_window_attention_mask = swa_mask[None, None, :, :].expand(
        batch_size, swa_num_attention_heads, target_length, source_length
    ).contiguous()
    
    return full_attention_mask, sliding_window_attention_mask
