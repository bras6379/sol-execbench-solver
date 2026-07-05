import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
):
    """
    Fused QKV projection with GQA reshaping.
    
    Args:
        hidden_states: [batch_size, seq_len, 2048]
        q_weight: [2048, 2048] - Query projection weight
        k_weight: [512, 2048] - Key projection weight
        v_weight: [512, 2048] - Value projection weight
    
    Returns:
        query_states: [batch_size, 16, seq_len, 128]
        key_states: [batch_size, 4, seq_len, 128]
        value_states: [batch_size, 4, seq_len, 128]
    """
    # Constants
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    
    bsz, q_len, _ = hidden_states.size()
    
    # Three linear projections (no bias)
    # Q: [bsz, q_len, 2048] @ [2048, 2048].T -> [bsz, q_len, 2048]
    query_states = torch.matmul(hidden_states, q_weight.t())
    
    # K: [bsz, q_len, 2048] @ [512, 2048].T -> [bsz, q_len, 512]
    key_states = torch.matmul(hidden_states, k_weight.t())
    
    # V: [bsz, q_len, 2048] @ [512, 2048].T -> [bsz, q_len, 512]
    value_states = torch.matmul(hidden_states, v_weight.t())
    
    # Reshape and transpose for attention computation
    # Query: [bsz, q_len, 2048] -> [bsz, q_len, 16, 128] -> [bsz, 16, q_len, 128]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    
    # Key: [bsz, q_len, 512] -> [bsz, q_len, 4, 128] -> [bsz, 4, q_len, 128]
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Value: [bsz, q_len, 512] -> [bsz, q_len, 4, 128] -> [bsz, 4, q_len, 128]
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    return query_states, key_states, value_states
