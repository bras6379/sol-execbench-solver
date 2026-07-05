import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    video_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    scale: float,
):
    """
    Backward pass for cross-attention text-video conditioning.
    
    Computes gradients through:
    1. Output projection
    2. Attention-value multiplication
    3. Softmax
    4. Attention score computation (Q @ K^T)
    5. Q, K, V linear projections
    """
    batch_size, num_video_tokens, hidden_size = video_latents.shape
    num_text_tokens = text_embeddings.shape[1]
    num_heads = 16
    head_dim = 64
    
    # Recompute forward pass intermediates
    # Project queries from video latents
    queries = F.linear(video_latents, query_weight, query_bias)  # [B, N_v, D]
    queries = queries.view(batch_size, num_video_tokens, num_heads, head_dim)
    queries = queries.transpose(1, 2)  # [B, H, N_v, d]
    
    # Project keys and values from text embeddings
    keys = F.linear(text_embeddings, key_weight, key_bias)  # [B, N_t, D]
    keys = keys.view(batch_size, num_text_tokens, num_heads, head_dim)
    keys = keys.transpose(1, 2)  # [B, H, N_t, d]
    
    values = F.linear(text_embeddings, value_weight, value_bias)  # [B, N_t, D]
    values = values.view(batch_size, num_text_tokens, num_heads, head_dim)
    values = values.transpose(1, 2)  # [B, H, N_t, d]
    
    # Compute attention scores
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # [B, H, N_v, N_t]
    
    # Compute attention probabilities
    attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)  # [B, H, N_v, N_t]
    
    # Apply attention to values
    context = torch.matmul(attention_probs, values)  # [B, H, N_v, d]
    
    # Reshape back to [B, N_v, D]
    context = context.transpose(1, 2).contiguous()  # [B, N_v, H, d]
    context = context.view(batch_size, num_video_tokens, hidden_size)  # [B, N_v, D]
    
    # ========================================
    # Backward through output projection
    # ========================================
    grad_context = torch.matmul(grad_output, output_weight)  # [B, N_v, D]
    
    # grad_output_weight = grad_output^T @ context (summed over batch)
    grad_output_weight = torch.einsum('bnd,bnk->dk', grad_output, context)  # [D, D]
    grad_output_bias = grad_output.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Backward through reshape from attention
    # ========================================
    grad_context_heads = grad_context.view(batch_size, num_video_tokens, num_heads, head_dim)
    grad_context_heads = grad_context_heads.transpose(1, 2)  # [B, H, N_v, d]
    
    # ========================================
    # Backward through attention @ values
    # ========================================
    grad_attention_probs = torch.matmul(grad_context_heads, values.transpose(-2, -1))  # [B, H, N_v, N_t]
    grad_values = torch.matmul(attention_probs.transpose(-2, -1), grad_context_heads)  # [B, H, N_t, d]
    
    # ========================================
    # Backward through softmax
    # ========================================
    sum_grad_probs = (grad_attention_probs * attention_probs).sum(dim=-1, keepdim=True)  # [B, H, N_v, 1]
    grad_attention_scores = attention_probs * (grad_attention_probs - sum_grad_probs)  # [B, H, N_v, N_t]
    
    # ========================================
    # Backward through scaling
    # ========================================
    grad_attention_scores = grad_attention_scores * scale
    
    # ========================================
    # Backward through Q @ K^T
    # ========================================
    grad_queries = torch.matmul(grad_attention_scores, keys)  # [B, H, N_v, d]
    grad_keys = torch.matmul(grad_attention_scores.transpose(-2, -1), queries)  # [B, H, N_t, d]
    
    # ========================================
    # Backward through reshape for Q, K, V
    # ========================================
    grad_queries = grad_queries.transpose(1, 2).contiguous()  # [B, N_v, H, d]
    grad_queries = grad_queries.view(batch_size, num_video_tokens, hidden_size)  # [B, N_v, D]
    
    grad_keys = grad_keys.transpose(1, 2).contiguous()  # [B, N_t, H, d]
    grad_keys = grad_keys.view(batch_size, num_text_tokens, hidden_size)  # [B, N_t, D]
    
    grad_values = grad_values.transpose(1, 2).contiguous()  # [B, N_t, H, d]
    grad_values = grad_values.view(batch_size, num_text_tokens, hidden_size)  # [B, N_t, D]
    
    # ========================================
    # Backward through Q projection
    # ========================================
    grad_video_latents = torch.matmul(grad_queries, query_weight)  # [B, N_v, D]
    grad_query_weight = torch.einsum('bnd,bnk->dk', grad_queries, video_latents)  # [D, D]
    grad_query_bias = grad_queries.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Backward through K projection
    # ========================================
    grad_text_from_keys = torch.matmul(grad_keys, key_weight)  # [B, N_t, D]
    grad_key_weight = torch.einsum('bnd,bnk->dk', grad_keys, text_embeddings)  # [D, D]
    grad_key_bias = grad_keys.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Backward through V projection
    # ========================================
    grad_text_from_values = torch.matmul(grad_values, value_weight)  # [B, N_t, D]
    grad_value_weight = torch.einsum('bnd,bnk->dk', grad_values, text_embeddings)  # [D, D]
    grad_value_bias = grad_values.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Accumulate gradients for text_embeddings
    # ========================================
    grad_text_embeddings = grad_text_from_keys + grad_text_from_values
    
    return (
        grad_video_latents,
        grad_text_embeddings,
        grad_query_weight,
        grad_query_bias,
        grad_key_weight,
        grad_key_bias,
        grad_value_weight,
        grad_value_bias,
        grad_output_weight,
        grad_output_bias,
    )
