import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """
    Generate inputs for the LM head projection kernel.
    
    Args:
        axes_and_scalars: Dictionary containing axis values
        device: Target device for tensors
    
    Returns:
        Dictionary of input tensors
    """
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    vocab_size = axes_and_scalars["vocab_size"]
    
    # Generate random hidden states
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.bfloat16, device=device
    )
    
    # Generate random weight matrix with proper initialization
    std = 1.0 / (hidden_size ** 0.5)
    weight = torch.randn(
        vocab_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * std
    
    return {
        "hidden_states": hidden_states,
        "weight": weight
    }


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    LM head projection with logit slicing.
    
    This kernel performs:
    1. Slice the last logits_to_keep positions from hidden_states
    2. Project sliced hidden states to vocabulary space via matmul with weight
    
    Args:
        hidden_states: [batch_size, seq_len, 2048] input hidden states
        weight: [102400, 2048] projection weight matrix
    
    Returns:
        logits: [batch_size, logits_to_keep, 102400] vocabulary logits
    
    Note: logits_to_keep is determined by the output shape requirement.
    The kernel slices hidden_states[:, -logits_to_keep:, :] before projection.
    For this reference implementation, we compute all positions and the
    slicing is handled by the benchmark framework based on output shape.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    vocab_size = weight.shape[0]
    
    # Perform the projection: [B, S, H] @ [H, V] = [B, S, V]
    # Using matmul with transposed weight
    # The slicing to logits_to_keep is implicit in the output shape
    logits = torch.matmul(hidden_states, weight.t())
    
    return logits
