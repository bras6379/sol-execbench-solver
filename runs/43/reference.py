import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """
    Fused MLA QKV projection with RoPE split.
    
    Query path: hidden -> q_a_proj -> RMSNorm -> q_b_proj -> split(q_nope, q_pe)
    KV path: hidden -> kv_a_proj -> split(compressed_kv, k_pe)
    """
    # Constants
    num_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    q_head_dim = 192  # qk_nope_head_dim + qk_rope_head_dim
    kv_lora_rank = 512
    
    bsz, seq_len, _ = hidden_states.shape
    
    # Query pathway: hidden -> q_a_proj -> layernorm -> q_b_proj -> split
    # q_a_proj: Linear(7168, 1536)
    q_latent = F.linear(hidden_states, q_a_proj_weight)  # (bsz, seq_len, 1536)
    
    # RMSNorm on q_latent
    input_dtype = q_latent.dtype
    q_latent_fp32 = q_latent.to(torch.float32)
    variance = q_latent_fp32.pow(2).mean(-1, keepdim=True)
    q_latent_normed = q_latent_fp32 * torch.rsqrt(variance + rms_norm_eps)
    q_latent = (q_a_layernorm_weight * q_latent_normed.to(input_dtype))
    
    # q_b_proj: Linear(1536, 24576)
    q = F.linear(q_latent, q_b_proj_weight)  # (bsz, seq_len, 24576)
    q = q.view(bsz, seq_len, num_heads, q_head_dim)  # (bsz, seq_len, 128, 192)
    
    # Split query into nope and pe components
    q_nope = q[..., :qk_nope_head_dim].contiguous()  # (bsz, seq_len, 128, 128)
    q_pe = q[..., qk_nope_head_dim:].contiguous()  # (bsz, seq_len, 128, 64)
    
    # KV pathway: hidden -> kv_a_proj -> split
    # kv_a_proj: Linear(7168, 576)
    kv_combined = F.linear(hidden_states, kv_a_proj_weight)  # (bsz, seq_len, 576)
    
    # Split into compressed_kv and k_pe
    compressed_kv = kv_combined[..., :kv_lora_rank].contiguous()  # (bsz, seq_len, 512)
    k_pe = kv_combined[..., kv_lora_rank:].contiguous()  # (bsz, seq_len, 64)
    k_pe = k_pe.view(bsz, seq_len, 1, qk_rope_head_dim)  # (bsz, seq_len, 1, 64)
    
    return q_nope, q_pe, compressed_kv, k_pe
