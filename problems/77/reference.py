import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Whisper decoder output projection: projects hidden states to vocabulary logits.
    
    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, d_model=1280)
        weight: Tensor of shape (vocab_size=51866, d_model=1280)
    
    Returns:
        logits: Tensor of shape (batch_size, seq_len, vocab_size=51866)
    """
    # Linear projection without bias: output = input @ weight.T
    # hidden_states: (batch_size, seq_len, d_model)
    # weight: (vocab_size, d_model)
    # logits: (batch_size, seq_len, vocab_size)
    logits = torch.matmul(hidden_states, weight.t())
    return logits
