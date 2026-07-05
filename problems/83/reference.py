import torch

@torch.no_grad()
def run(
    attention_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fused attention output: attention_weights @ value with transpose and reshape.
    
    Args:
        attention_weights: Softmax-normalized attention scores [B, H, Q, K]
        value: Value matrix [B, H, K, D]
        
    Returns:
        output: Attention output [B, Q, H*D] ready for output projection
    """
    batch_size = attention_weights.shape[0]
    seq_len_q = attention_weights.shape[2]
    num_heads = 20
    head_dim = 64
    hidden_size = num_heads * head_dim
    
    # Core attention matmul: [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
    attn_output = torch.matmul(attention_weights, value)
    
    # Transpose: [B, H, Q, D] -> [B, Q, H, D]
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # Reshape: [B, Q, H, D] -> [B, Q, H*D]
    attn_output = attn_output.reshape(batch_size, seq_len_q, hidden_size)
    
    return attn_output
