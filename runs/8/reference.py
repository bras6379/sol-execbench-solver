import torch


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the reference forward pass. Required method."""
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_size * seq_len * num_experts_per_tok

    # Initialize accumulation buffer with random values (not zeros) to detect no-op
    final_hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert outputs (weighted outputs from expert computation)
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Token indices (which token position each expert output belongs to)
    # These should be in range [0, batch_seq_len)
    # Simulate scattered indices with potential duplicates (for top_k > 1)
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.long, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs atomic accumulation of expert outputs back to token positions.
    
    Args:
        final_hidden_states: Accumulation buffer for all tokens (batch_seq_len, hidden_size)
        expert_outputs: Weighted outputs from expert computation (num_selected_tokens, hidden_size)
        token_indices: Original token positions (num_selected_tokens,)
        
    Returns:
        Updated final_hidden_states with expert contributions added
    """
    # Clone to avoid modifying input in-place for reference correctness
    output = final_hidden_states.clone()
    
    # Critical atomic accumulation operation
    # This performs: output[token_indices[i]] += expert_outputs[i]
    # for all i in parallel with atomic semantics
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)

    return output
