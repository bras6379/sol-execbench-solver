import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars['batch_size']
    num_kv_heads = axes_and_scalars['num_kv_heads']
    new_seq_len = axes_and_scalars['new_seq_len']
    head_dim = axes_and_scalars['head_dim']
    max_seq_len = axes_and_scalars['max_seq_len']
    
    grad_key_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    grad_value_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key_states = torch.randn(batch_size, num_kv_heads, new_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    cos = torch.randn(batch_size, new_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(batch_size, new_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    cache_position = torch.arange(new_seq_len, dtype=torch.int64, device=device)
    
    return {
        'grad_key_cache': grad_key_cache,
        'grad_value_cache': grad_value_cache,
        'key_states': key_states,
        'cos': cos,
        'sin': sin,
        'cache_position': cache_position,
    }

@torch.no_grad()
def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    half_dim = key_states.shape[-1] // 2
    
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated_half = torch.cat((-k2, k1), dim=-1)
    
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    grad_key_states_rotated = grad_key_cache[:, :, cache_position]
    grad_value_states = grad_value_cache[:, :, cache_position]
    
    grad_from_cos_term = grad_key_states_rotated * cos_expanded
    grad_k_rotated_half = grad_key_states_rotated * sin_expanded
    
    grad_k_rotated_half_1 = grad_k_rotated_half[..., :half_dim]
    grad_k_rotated_half_2 = grad_k_rotated_half[..., half_dim:]
    
    grad_k2_from_rotate = -grad_k_rotated_half_1
    grad_k1_from_rotate = grad_k_rotated_half_2
    
    grad_k1_total = grad_from_cos_term[..., :half_dim] + grad_k1_from_rotate
    grad_k2_total = grad_from_cos_term[..., half_dim:] + grad_k2_from_rotate
    
    grad_key_states = torch.cat([grad_k1_total, grad_k2_total], dim=-1)
    
    grad_cos_expanded = grad_key_states_rotated * key_states
    grad_cos = grad_cos_expanded.sum(dim=1)
    
    grad_sin_expanded = grad_key_states_rotated * k_rotated_half
    grad_sin = grad_sin_expanded.sum(dim=1)
    
    grad_key_cache_input = grad_key_cache.clone()
    grad_key_cache_input[:, :, cache_position] = 0
    
    grad_value_cache_input = grad_value_cache.clone()
    grad_value_cache_input[:, :, cache_position] = 0
    
    return (
        grad_key_states.to(torch.bfloat16),
        grad_value_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_key_cache_input.to(torch.bfloat16),
        grad_value_cache_input.to(torch.bfloat16),
    )
