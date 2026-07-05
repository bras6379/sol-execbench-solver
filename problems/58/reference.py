import torch


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices in range [0, num_experts-1]."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Generate random expert indices in valid range [0, num_experts-1]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    
    return {"topk_idx": topk_idx}


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    """
    MoE token sorting using counting sort with prefix sum.
    
    Args:
        topk_idx: Expert indices (batch_size, seq_len, num_experts_per_tok)
                 with values in [0, num_experts-1]
    
    Returns:
        sorted_token_indices: Token indices sorted by expert (num_tokens,)
        expert_offsets: Cumulative offsets (num_experts+1,)
    """
    num_experts = 256
    flat = topk_idx.reshape(-1)

    # Stable sort on expert IDs = counting sort permutation
    _, sorted_token_indices = flat.sort(stable=True)

    # Expert offsets via histogram + prefix sum
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_offsets[1:] = torch.bincount(flat.long(), minlength=num_experts).cumsum(0).to(torch.int32)

    return sorted_token_indices.to(torch.int32), expert_offsets
