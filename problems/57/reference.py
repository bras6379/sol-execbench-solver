import torch


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MTP shifted embedding fusion."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 4096
    vocab_size = 157184
    hidden_size_x2 = 8192
    rms_norm_eps = 1e-6
    
    # Input IDs - random token indices in valid vocab range
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device
    )
    
    # Hidden states from previous layer
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device
    )
    
    # Word embedding table
    word_embeddings = torch.randn(
        vocab_size, hidden_size, dtype=torch.bfloat16, device=device
    )
    
    # RMSNorm weights (initialized to ones-like with small perturbation)
    enorm_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device) + \
                   torch.randn(hidden_size, dtype=torch.bfloat16, device=device) * 0.01
    hnorm_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device) + \
                   torch.randn(hidden_size, dtype=torch.bfloat16, device=device) * 0.01
    
    # Projection weight
    eh_proj_weight = torch.randn(
        hidden_size, hidden_size_x2, dtype=torch.bfloat16, device=device
    ) * 0.02  # Small init for projection
    
    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "word_embeddings": word_embeddings,
        "enorm_weight": enorm_weight,
        "hnorm_weight": hnorm_weight,
        "eh_proj_weight": eh_proj_weight,
        "rms_norm_eps": rms_norm_eps,
    }


@torch.no_grad()
def run(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    word_embeddings: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eh_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """
    MTP Shifted Embedding with Dual RMSNorm Fusion.
    
    1. Roll input_ids by -1 (shift left for next token prediction)
    2. Embed shifted input ids
    3. Apply RMSNorm to embeddings (enorm)
    4. Apply RMSNorm to hidden states (hnorm)
    5. Concatenate normalized embeddings and hidden states
    6. Project concatenated tensor back to hidden_size
    
    Args:
        input_ids: (batch_size, seq_len) token ids
        hidden_states: (batch_size, seq_len, hidden_size) current hidden states
        word_embeddings: (vocab_size, hidden_size) embedding table
        enorm_weight: (hidden_size,) RMSNorm weight for embeddings
        hnorm_weight: (hidden_size,) RMSNorm weight for hidden states
        eh_proj_weight: (hidden_size, hidden_size*2) projection weight
        rms_norm_eps: epsilon for numerical stability
    
    Returns:
        fused_hidden_states: (batch_size, seq_len, hidden_size)
    """
    # Step 1: Roll input_ids by -1 (shift left)
    # Last position gets filled with 0
    shifted_input_ids = torch.roll(input_ids, shifts=-1, dims=-1)
    shifted_input_ids[:, -1] = 0
    
    # Step 2: Embed shifted input ids
    # F.embedding equivalent: index into embedding table
    input_embeds = word_embeddings[shifted_input_ids]  # (batch_size, seq_len, hidden_size)
    
    # Step 3: Apply RMSNorm to embeddings
    # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    input_embeds_fp32 = input_embeds.to(torch.float32)
    embed_variance = input_embeds_fp32.pow(2).mean(-1, keepdim=True)
    input_embeds_normed = input_embeds_fp32 * torch.rsqrt(embed_variance + rms_norm_eps)
    input_embeds_normed = (enorm_weight * input_embeds_normed.to(enorm_weight.dtype))
    
    # Step 4: Apply RMSNorm to hidden states
    hidden_states_fp32 = hidden_states.to(torch.float32)
    hidden_variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    hidden_states_normed = hidden_states_fp32 * torch.rsqrt(hidden_variance + rms_norm_eps)
    hidden_states_normed = (hnorm_weight * hidden_states_normed.to(hnorm_weight.dtype))
    
    # Step 5: Concatenate along feature dimension
    # Shape: (batch_size, seq_len, hidden_size * 2)
    concatenated = torch.cat([input_embeds_normed, hidden_states_normed], dim=-1)
    
    # Step 6: Project back to hidden_size
    # Linear: x @ weight.T (weight is hidden_size x hidden_size*2)
    # So we do: (batch_size, seq_len, hidden_size*2) @ (hidden_size*2, hidden_size)
    fused_hidden_states = torch.matmul(concatenated, eh_proj_weight.t())
    
    return fused_hidden_states
