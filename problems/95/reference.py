import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    x_mean: torch.Tensor,
    x_var: torch.Tensor,
    x_norm: torch.Tensor,
    ln_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    out_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
):
    """
    Backward pass for fused vision multi-head attention with norms.
    
    Computes gradients through:
    1. Residual connection
    2. Output projection
    3. Attention mechanism (softmax, matmul)
    4. Head split/merge
    5. QKV projection
    6. LayerNorm
    """
    batch_size, seq_len, embed_dim = x.shape
    num_heads = 16
    head_dim = 64
    
    # Gradient through residual connection
    # output = output_before_residual + residual
    grad_output_before_residual = grad_output
    grad_residual = grad_output
    
    # Gradient through output projection
    # output = F.linear(attn_output, out_weight, out_bias)
    grad_attn_output = F.linear(grad_output_before_residual, out_weight.t())
    grad_out_weight = torch.matmul(
        grad_output_before_residual.reshape(-1, embed_dim).t(),
        attn_output.reshape(-1, embed_dim)
    )
    grad_out_bias = grad_output_before_residual.sum(dim=(0, 1))
    
    # Gradient through head merging
    grad_attn_output_heads = grad_attn_output.view(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)
    
    # Gradient through attention application: attn_output = attn_weights @ v
    grad_v = torch.matmul(
        attn_weights.transpose(-2, -1),
        grad_attn_output_heads
    )
    
    grad_attn_weights = torch.matmul(
        grad_attn_output_heads,
        v.transpose(-2, -1)
    )
    
    # Gradient through softmax
    sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)
    
    # Gradient through scaled dot-product: attn_scores = (q * scale) @ k^T
    grad_q = torch.matmul(grad_attn_scores, k) * scale
    grad_k = torch.matmul(
        grad_attn_scores.transpose(-2, -1),
        q * scale
    )
    
    # Gradient through head splitting
    grad_q = grad_q.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_k = grad_k.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_v = grad_v.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    
    # Gradient through QKV chunking
    grad_qkv = torch.cat([grad_q, grad_k, grad_v], dim=-1)
    
    # Gradient through QKV projection
    grad_x_norm = F.linear(grad_qkv, qkv_weight.t())
    grad_qkv_weight = torch.matmul(
        grad_qkv.reshape(-1, 3 * embed_dim).t(),
        x_norm.reshape(-1, embed_dim)
    )
    grad_qkv_bias = grad_qkv.sum(dim=(0, 1))
    
    # Gradient through LayerNorm
    x_normalized = (x - x_mean) / torch.sqrt(x_var + norm_eps)
    grad_ln_weight = (grad_x_norm * x_normalized).sum(dim=(0, 1))
    grad_ln_bias = grad_x_norm.sum(dim=(0, 1))
    
    # Gradient through normalization
    grad_x_normalized = grad_x_norm * ln_weight
    
    # Gradient through standardization
    std = torch.sqrt(x_var + norm_eps)
    x_centered = x - x_mean
    
    grad_x_from_norm = grad_x_normalized / std
    mean_grad = grad_x_from_norm.mean(dim=-1, keepdim=True)
    mean_grad_x_centered = (grad_x_normalized * x_centered).mean(dim=-1, keepdim=True) / (std ** 2)
    
    grad_x_from_norm = grad_x_from_norm - mean_grad - x_centered * mean_grad_x_centered
    
    # Combine gradients from residual and normalization paths
    grad_x = grad_residual + grad_x_from_norm
    
    return (
        grad_x,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
