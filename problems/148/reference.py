import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_state: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    input_layernorm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    post_attention_layernorm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    gate_attn: torch.Tensor,
    gate_ffn: torch.Tensor,
    norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_state.shape
    num_heads = 16
    head_dim = hidden_size // num_heads
    scaling = head_dim ** -0.5
    
    # Self-attention block with gated residual
    residual = hidden_state
    
    # Input layer norm
    hidden_state_fp32 = hidden_state.to(torch.float32)
    mean = hidden_state_fp32.mean(dim=-1, keepdim=True)
    var = ((hidden_state_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_state_normed = (hidden_state_fp32 - mean) / torch.sqrt(var + norm_eps)
    hidden_state_normed = hidden_state_normed.to(hidden_state.dtype)
    hidden_state = hidden_state_normed * input_layernorm_weight + input_layernorm_bias
    
    # Compute Q, K, V
    query = torch.matmul(hidden_state, q_proj_weight.t())
    key = torch.matmul(hidden_state, k_proj_weight.t())
    value = torch.matmul(hidden_state, v_proj_weight.t())
    
    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    
    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, o_proj_weight.t())
    
    # Gated residual connection for attention
    hidden_state = residual + torch.tanh(gate_attn) * attn_output
    
    # MLP block with gated residual
    residual = hidden_state
    
    # Post-attention layer norm
    hidden_state_fp32 = hidden_state.to(torch.float32)
    mean = hidden_state_fp32.mean(dim=-1, keepdim=True)
    var = ((hidden_state_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_state_normed = (hidden_state_fp32 - mean) / torch.sqrt(var + norm_eps)
    hidden_state_normed = hidden_state_normed.to(hidden_state.dtype)
    hidden_state = hidden_state_normed * post_attention_layernorm_weight + post_attention_layernorm_bias
    
    # MLP with GELU activation
    hidden_state = torch.matmul(hidden_state, fc1_weight.t()) + fc1_bias
    hidden_state = F.gelu(hidden_state)
    hidden_state = torch.matmul(hidden_state, fc2_weight.t()) + fc2_bias
    
    # Gated residual connection for MLP
    output = residual + torch.tanh(gate_ffn) * hidden_state
    
    return output
