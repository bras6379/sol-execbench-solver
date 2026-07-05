import torch

@torch.no_grad()
def run(
    compressed_kv: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    eps: float,
):
    """
    DeepSeek-V3 latent KV expansion with split.
    
    1. RMSNorm on compressed KV
    2. Linear projection to expanded space
    3. Reshape and transpose
    4. Split into K_nope and V
    
    Args:
        compressed_kv: [batch_size, seq_len, kv_lora_rank=512]
        kv_a_layernorm_weight: [kv_lora_rank=512]
        kv_b_proj_weight: [kv_expanded_dim=32768, kv_lora_rank=512]
        eps: RMSNorm epsilon
    
    Returns:
        k_nope: [batch_size, num_heads=128, seq_len, qk_nope_head_dim=128]
        value_states: [batch_size, num_heads=128, seq_len, v_head_dim=128]
    """
    # Constants
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128
    
    bsz, seq_len, _ = compressed_kv.shape
    
    # Step 1: RMSNorm on compressed KV
    # Convert to float32 for numerical stability
    input_dtype = compressed_kv.dtype
    hidden_states = compressed_kv.to(torch.float32)
    
    # Compute variance and normalize
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    
    # Apply weight and convert back to original dtype
    normalized_kv = (kv_a_layernorm_weight.to(torch.float32) * hidden_states).to(input_dtype)
    
    # Step 2: Linear projection
    # [batch, seq_len, 512] @ [512, 32768] -> [batch, seq_len, 32768]
    expanded_kv = torch.matmul(normalized_kv, kv_b_proj_weight.t())
    
    # Step 3: Reshape and transpose
    # [batch, seq_len, 32768] -> [batch, seq_len, 128, 256]
    kv = expanded_kv.view(bsz, seq_len, num_heads, qk_nope_head_dim + v_head_dim)
    
    # [batch, seq_len, 128, 256] -> [batch, 128, seq_len, 256]
    kv = kv.transpose(1, 2)
    
    # Step 4: Split into K_nope and V along the last dimension
    # k_nope: [batch, 128, seq_len, 128]
    # value_states: [batch, 128, seq_len, 128]
    k_nope = kv[:, :, :, :qk_nope_head_dim].contiguous()
    value_states = kv[:, :, :, qk_nope_head_dim:].contiguous()
    
    return k_nope, value_states
