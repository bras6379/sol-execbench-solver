import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
):
    """
    Fused QKV projection with bias and reshape for Gemma3 attention.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, 640)
        q_weight: Query projection weight (1024, 640)
        q_bias: Query projection bias (1024,)
        k_weight: Key projection weight (256, 640)
        k_bias: Key projection bias (256,)
        v_weight: Value projection weight (256, 640)
        v_bias: Value projection bias (256,)
        
    Returns:
        Tuple of (query_states, key_states, value_states) where:
            query_states: (batch_size, seq_len, 16, 128)
            key_states: (batch_size, seq_len, 2, 128)
            value_states: (batch_size, seq_len, 2, 128)
    """
    batch_size, seq_len, _ = hidden_states.shape
    
    # Constants
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256
    
    # Q projection: (batch, seq, 640) @ (1024, 640).T + bias -> (batch, seq, 1024)
    query_states = torch.matmul(hidden_states, q_weight.t()) + q_bias
    
    # K projection: (batch, seq, 640) @ (256, 640).T + bias -> (batch, seq, 256)
    key_states = torch.matmul(hidden_states, k_weight.t()) + k_bias
    
    # V projection: (batch, seq, 640) @ (256, 640).T + bias -> (batch, seq, 256)
    value_states = torch.matmul(hidden_states, v_weight.t()) + v_bias
    
    # Reshape for multi-head attention
    # Query: (batch, seq_len, 1024) -> (batch, seq_len, 4, 256)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    
    # Key: (batch, seq_len, 256) -> (batch, seq_len, 1, 256)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Value: (batch, seq_len, 256) -> (batch, seq_len, 1, 256)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    return query_states, key_states, value_states
