import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    """
    Fused value projection with reshape and transpose for GQA attention.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, 5120]
        v_proj_weight: Weight matrix of shape [1024, 5120]
        
    Returns:
        value_states: Reshaped and transposed tensor of shape [batch_size, 8, seq_len, 128]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_kv_heads = 8
    head_dim = 128
    
    # Project to value space: [batch, seq_len, 5120] @ [5120, 1024] -> [batch, seq_len, 1024]
    # F.linear computes input @ weight.T, so with weight [1024, 5120], we get [batch, seq_len, 1024]
    value_proj = torch.nn.functional.linear(hidden_states, v_proj_weight)
    
    # Reshape to separate heads: [batch, seq_len, 1024] -> [batch, seq_len, 8, 128]
    value_states = value_proj.view(batch_size, seq_len, num_kv_heads, head_dim)
    
    # Transpose to attention format: [batch, seq_len, 8, 128] -> [batch, 8, seq_len, 128]
    value_states = value_states.transpose(1, 2).contiguous()
    
    return value_states
