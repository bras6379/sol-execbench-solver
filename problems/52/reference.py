import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
):
    """
    Collapses multiple AltUp prediction streams into a single hidden state.
    
    Args:
        hidden_states: [altup_num_inputs, batch_size, seq_len, hidden_size]
        unembed_proj_1: [hidden_size, hidden_size] - projection for stream 1
        unembed_proj_2: [hidden_size, hidden_size] - projection for stream 2
        epsilon: small value for numerical stability
    
    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    # Extract first stream as reference (no projection needed)
    first_stream = hidden_states[0]  # [batch, seq_len, hidden_size]
    
    # Compute target magnitude from first stream
    # Shape: [batch, seq_len, 1]
    target_magnitude = torch.sqrt(
        torch.mean(first_stream.to(torch.float32) ** 2, dim=-1, keepdim=True)
    )
    
    # Initialize list to collect all streams
    collapsed_streams = [first_stream.to(torch.float32)]
    
    # Process stream 1
    # Project through unembed matrix: [batch, seq_len, hidden_size] @ [hidden_size, hidden_size].T
    projected_stream_1 = torch.matmul(
        hidden_states[1].to(torch.float32),
        unembed_proj_1.to(torch.float32).t()
    )
    
    # Compute current magnitude
    current_magnitude_1 = torch.sqrt(
        torch.maximum(
            torch.mean(projected_stream_1 ** 2, dim=-1, keepdim=True),
            torch.tensor(epsilon, dtype=torch.float32, device=hidden_states.device)
        )
    )
    
    # Normalize to match target magnitude
    normalized_stream_1 = projected_stream_1 * (target_magnitude / current_magnitude_1)
    collapsed_streams.append(normalized_stream_1)
    
    # Process stream 2
    projected_stream_2 = torch.matmul(
        hidden_states[2].to(torch.float32),
        unembed_proj_2.to(torch.float32).t()
    )
    
    # Compute current magnitude
    current_magnitude_2 = torch.sqrt(
        torch.maximum(
            torch.mean(projected_stream_2 ** 2, dim=-1, keepdim=True),
            torch.tensor(epsilon, dtype=torch.float32, device=hidden_states.device)
        )
    )
    
    # Normalize to match target magnitude
    normalized_stream_2 = projected_stream_2 * (target_magnitude / current_magnitude_2)
    collapsed_streams.append(normalized_stream_2)
    
    # Stack all streams and compute mean
    stacked_streams = torch.stack(collapsed_streams, dim=0)  # [3, batch, seq_len, hidden_size]
    
    # Average across streams (dim=0)
    output = torch.mean(stacked_streams, dim=0)  # [batch, seq_len, hidden_size]
    
    return output.to(torch.bfloat16)
