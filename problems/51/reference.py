import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_transposed: torch.Tensor,
    key_transposed: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    rms_norm_eps: float,
):
    # Constants
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256
    hidden_size = 640
    
    batch_size, seq_len, _ = hidden_states.shape
    
    # ========== Backward through Q normalization ==========
    grad_query_float = grad_query.float()
    q_scale = 1.0 + q_norm_weight.float()  # (head_dim,)
    
    # Gradient w.r.t. q_norm_weight: sum over batch, num_heads, seq_len
    grad_q_norm_weight = (grad_query_float * q_normed.float()).sum(dim=(0, 1, 2))  # (head_dim,)
    
    # Gradient w.r.t. q_normed
    grad_q_normed = grad_query_float * q_scale  # (batch, num_heads, seq_len, head_dim)
    grad_q_normed_bf16 = grad_q_normed.to(torch.bfloat16)
    
    # Backward through RMSNorm for Q
    q_mean_term = (grad_q_normed_bf16.float() * q_normed.float()).mean(dim=-1, keepdim=True)
    grad_q_transposed = (q_rstd * (grad_q_normed_bf16.float() - q_mean_term * q_normed.float())).to(torch.bfloat16)
    
    # ========== Backward through K normalization ==========
    grad_key_float = grad_key.float()
    k_scale = 1.0 + k_norm_weight.float()  # (head_dim,)
    
    # Gradient w.r.t. k_norm_weight
    grad_k_norm_weight = (grad_key_float * k_normed.float()).sum(dim=(0, 1, 2))  # (head_dim,)
    
    # Gradient w.r.t. k_normed
    grad_k_normed = grad_key_float * k_scale
    grad_k_normed_bf16 = grad_k_normed.to(torch.bfloat16)
    
    # Backward through RMSNorm for K
    k_mean_term = (grad_k_normed_bf16.float() * k_normed.float()).mean(dim=-1, keepdim=True)
    grad_k_transposed = (k_rstd * (grad_k_normed_bf16.float() - k_mean_term * k_normed.float())).to(torch.bfloat16)
    
    # ========== Backward through V (no normalization) ==========
    grad_v_transposed = grad_value
    
    # ========== Backward through transpose ==========
    grad_q_reshaped = grad_q_transposed.transpose(1, 2).contiguous()
    grad_k_reshaped = grad_k_transposed.transpose(1, 2).contiguous()
    grad_v_reshaped = grad_v_transposed.transpose(1, 2).contiguous()
    
    # ========== Backward through reshape ==========
    grad_query_proj = grad_q_reshaped.view(batch_size, seq_len, num_attention_heads * head_dim)
    grad_key_proj = grad_k_reshaped.view(batch_size, seq_len, num_key_value_heads * head_dim)
    grad_value_proj = grad_v_reshaped.view(batch_size, seq_len, num_key_value_heads * head_dim)
    
    # ========== Backward through linear projections ==========
    # Gradient w.r.t. hidden_states (accumulate from all three projections)
    # For linear layer: y = x @ W^T, grad_x = grad_y @ W
    grad_hidden_states = torch.matmul(grad_query_proj, q_weight)
    grad_hidden_states = grad_hidden_states + torch.matmul(grad_key_proj, k_weight)
    grad_hidden_states = grad_hidden_states + torch.matmul(grad_value_proj, v_weight)
    
    # Gradient w.r.t. weights: grad_W = grad_y^T @ x
    grad_query_proj_2d = grad_query_proj.reshape(-1, num_attention_heads * head_dim)
    grad_key_proj_2d = grad_key_proj.reshape(-1, num_key_value_heads * head_dim)
    grad_value_proj_2d = grad_value_proj.reshape(-1, num_key_value_heads * head_dim)
    hidden_states_2d = hidden_states.reshape(-1, hidden_size)
    
    grad_q_weight = torch.matmul(grad_query_proj_2d.t(), hidden_states_2d)
    grad_k_weight = torch.matmul(grad_key_proj_2d.t(), hidden_states_2d)
    grad_v_weight = torch.matmul(grad_value_proj_2d.t(), hidden_states_2d)
    
    return (
        grad_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )
