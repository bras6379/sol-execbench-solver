import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for GQA with RoPE.
    
    Computes gradients through:
    1. Output projection
    2. Attention output (softmax @ V)
    3. Attention scores (Q @ K^T)
    4. Repeat KV aggregation
    5. RoPE
    6. Q/K/V projections
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    kv_seq_len = key_states.shape[2]
    
    # 1. Gradient through output projection
    # output = attn_output @ o_weight^T
    # grad_attn_output = grad_output @ o_weight
    # grad_o_weight = grad_output^T @ attn_output
    grad_attn_output = torch.matmul(grad_output, o_weight)
    grad_o_weight = torch.matmul(
        grad_output.reshape(-1, grad_output.shape[-1]).t(),
        attn_output.reshape(-1, attn_output.shape[-1])
    )
    
    # 2. Gradient through reshape and transpose
    # [batch, seq_len, 4096] -> [batch, seq_len, 32, 128] -> [batch, 32, seq_len, 128]
    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, num_attention_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(1, 2)
    
    # 3. Gradient through attention: attn_output = attn_weights @ value_states
    # grad_attn_weights = grad_attn_output @ value_states^T
    # grad_value_states = attn_weights^T @ grad_attn_output
    grad_attn_weights = torch.matmul(grad_attn_output, value_states.transpose(2, 3))
    grad_value_states = torch.matmul(attn_weights.transpose(2, 3), grad_attn_output)
    
    # 4. Gradient through softmax
    # For softmax: grad_input = softmax * (grad_output - sum(grad_output * softmax))
    attn_weights_fp32 = attn_weights.to(torch.float32)
    grad_attn_weights_fp32 = grad_attn_weights.to(torch.float32)
    sum_grad = (grad_attn_weights_fp32 * attn_weights_fp32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights_fp32 * (grad_attn_weights_fp32 - sum_grad)
    grad_attn_scores = grad_attn_scores.to(query_states.dtype)
    
    # 5. Gradient through attention scores: attn_scores = (Q @ K^T) * scaling
    grad_attn_scores = grad_attn_scores * scaling
    
    # 6. Gradient through Q @ K^T
    # grad_Q = grad_attn_scores @ K
    # grad_K = grad_attn_scores^T @ Q
    grad_query_states = torch.matmul(grad_attn_scores, key_states)
    grad_key_states = torch.matmul(grad_attn_scores.transpose(2, 3), query_states)
    
    # 7. Gradient through repeat_kv (aggregate gradients for GQA)
    # Sum over the repeated dimension
    if num_key_value_groups != 1:
        grad_key_states = grad_key_states.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        ).sum(dim=2)
        
        grad_value_states = grad_value_states.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        ).sum(dim=2)
    
    # 8. Gradient through RoPE for queries
    # query_states = query_pre_rope * cos + rotate_half(query_pre_rope) * sin
    # grad_query_pre_rope = grad_query_states * cos + rotate_half_inverse(grad_query_states * sin)
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)
    
    # Query gradient through RoPE
    grad_q_cos = grad_query_states * cos_expanded
    grad_q_sin = grad_query_states * sin_expanded
    
    # Rotate grad_q_sin back (rotate_half_inverse)
    grad_q_sin_1 = grad_q_sin[..., : head_dim // 2]
    grad_q_sin_2 = grad_q_sin[..., head_dim // 2 :]
    grad_q_sin_rotated = torch.cat((grad_q_sin_2, -grad_q_sin_1), dim=-1)
    
    grad_query_states_pre_rope = grad_q_cos + grad_q_sin_rotated
    
    # Key gradient through RoPE
    grad_k_cos = grad_key_states * cos_expanded
    grad_k_sin = grad_key_states * sin_expanded
    
    grad_k_sin_1 = grad_k_sin[..., : head_dim // 2]
    grad_k_sin_2 = grad_k_sin[..., head_dim // 2 :]
    grad_k_sin_rotated = torch.cat((grad_k_sin_2, -grad_k_sin_1), dim=-1)
    
    grad_key_states_pre_rope = grad_k_cos + grad_k_sin_rotated
    
    # Value gradient (no RoPE applied to values)
    grad_value_states_pre_rope = grad_value_states
    
    # 9. Gradient through transpose and reshape for Q/K/V
    # [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim] -> [batch, seq_len, proj_size]
    grad_query_states_pre_rope = grad_query_states_pre_rope.transpose(1, 2).contiguous()
    grad_query_proj = grad_query_states_pre_rope.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    grad_key_states_pre_rope = grad_key_states_pre_rope.transpose(1, 2).contiguous()
    grad_key_proj = grad_key_states_pre_rope.reshape(batch_size, seq_len, num_key_value_heads * head_dim)
    
    grad_value_states_pre_rope = grad_value_states_pre_rope.transpose(1, 2).contiguous()
    grad_value_proj = grad_value_states_pre_rope.reshape(batch_size, seq_len, num_key_value_heads * head_dim)
    
    # 10. Gradient through Q/K/V projections
    # Q = hidden_states @ q_weight^T
    # grad_hidden_states_q = grad_Q @ q_weight
    # grad_q_weight = grad_Q^T @ hidden_states
    grad_hidden_states_q = torch.matmul(grad_query_proj, q_weight)
    grad_q_weight = torch.matmul(
        grad_query_proj.reshape(-1, grad_query_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    
    grad_hidden_states_k = torch.matmul(grad_key_proj, k_weight)
    grad_k_weight = torch.matmul(
        grad_key_proj.reshape(-1, grad_key_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    
    grad_hidden_states_v = torch.matmul(grad_value_proj, v_weight)
    grad_v_weight = torch.matmul(
        grad_value_proj.reshape(-1, grad_value_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    
    # Sum gradients for hidden_states from all three projection branches
    grad_hidden_states = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v
    
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_o_weight.to(torch.bfloat16),
    )
