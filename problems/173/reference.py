import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 12
    head_dim = hidden_size // num_heads
    scale = head_dim ** -0.5
    
    # ===== Attention Block =====
    residual = hidden_states
    
    # LayerNorm1
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm1_weight + layer_norm1_bias
    
    # Q, K, V projections
    queries = torch.matmul(hidden_states, q_proj_weight.t()) + q_proj_bias
    keys = torch.matmul(hidden_states, k_proj_weight.t()) + k_proj_bias
    values = torch.matmul(hidden_states, v_proj_weight.t()) + v_proj_bias
    
    # Reshape for multi-head attention
    queries = queries.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    keys = keys.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    values = values.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention computation
    attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    
    attn_output = torch.matmul(attn_weights, values)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # Output projection
    attn_output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    hidden_states = residual + attn_output
    
    # ===== MLP Block =====
    residual = hidden_states
    
    # LayerNorm2
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm2_weight + layer_norm2_bias
    
    # MLP: fc1 -> quick_gelu -> fc2
    hidden_states = torch.matmul(hidden_states, fc1_weight.t()) + fc1_bias
    # Quick GELU: x * sigmoid(1.702 * x)
    hidden_states = hidden_states * torch.sigmoid(1.702 * hidden_states)
    hidden_states = torch.matmul(hidden_states, fc2_weight.t()) + fc2_bias
    
    # Residual connection
    output = residual + hidden_states
    
    return output
