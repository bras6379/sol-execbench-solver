import torch

@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    """
    Fused cos/sin embedding generation from frequencies for RoPE.
    
    Args:
        freqs: Frequency tensor of shape [batch_size, seq_len, head_dim // 2]
               Result of (inv_freq @ position_ids).transpose(1, 2)
        attention_scaling: Scaling factor for attention
    
    Returns:
        Tuple of (cos, sin) embeddings, each of shape [batch_size, seq_len, head_dim]
    """
    # Concatenate frequencies with themselves along last dimension
    # This doubles the head_dim dimension: [batch, seq, head_dim//2] -> [batch, seq, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Compute cos and sin with scaling
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    
    # Cast to target dtype (bfloat16)
    cos = cos.to(torch.bfloat16)
    sin = sin.to(torch.bfloat16)
    
    return cos, sin
