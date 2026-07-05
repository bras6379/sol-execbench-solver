import torch
import torch.nn.functional as F

@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Apply Gemma3's softcapping transformation followed by softmax.
    
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30]
    
    Args:
        attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        
    Returns:
        Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    SOFTCAP = 30.0
    
    # Apply softcapping transformation
    # Step 1: Divide by softcap
    scaled = attn_weights / SOFTCAP
    
    # Step 2: Apply tanh to clamp to [-1, 1]
    clamped = torch.tanh(scaled)
    
    # Step 3: Multiply by softcap to restore scale (now in [-30, 30])
    softcapped = clamped * SOFTCAP
    
    # Apply softmax normalization along the key dimension
    # Upcast to float32 for numerical stability, then cast back
    output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    
    return output
