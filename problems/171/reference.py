import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    mlp_lin1_weight: torch.Tensor,
    mlp_lin1_bias: torch.Tensor,
    mlp_lin2_weight: torch.Tensor,
    mlp_lin2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    # Constants
    window_size = 14
    num_attention_heads = 12
    head_dim = 64
    scale = head_dim ** -0.5
    
    batch_size, height, width, channels = hidden_states.shape
    
    # Store for residual
    residual = hidden_states
    
    # Layer norm 1
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm1_weight + layer_norm1_bias
    
    # Window partition with padding
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
    
    pad_height = height + pad_h
    pad_width = width + pad_w
    
    # Reshape into windows
    hidden_states = hidden_states.reshape(
        batch_size,
        pad_height // window_size,
        window_size,
        pad_width // window_size,
        window_size,
        channels
    )
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(-1, window_size, window_size, channels)
    
    batch_windows = windows.shape[0]
    window_h = window_size
    window_w = window_size
    
    # QKV projection
    qkv = F.linear(windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(batch_windows, window_h * window_w, 3, num_attention_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_windows, num_heads, seq_len, head_dim)
    
    query, key, value = qkv[0], qkv[1], qkv[2]
    
    # Attention scores
    attn_weights = (query @ key.transpose(-2, -1)) * scale
    
    # Compute relative position bias
    coords_h = torch.arange(window_h, device=rel_pos_h.device)
    coords_w = torch.arange(window_w, device=rel_pos_w.device)
    
    rel_coords_h = coords_h[:, None] - coords_h[None, :]
    rel_coords_w = coords_w[:, None] - coords_w[None, :]
    
    rel_coords_h = rel_coords_h + window_h - 1
    rel_coords_w = rel_coords_w + window_w - 1
    
    # rel_pos_h_emb shape: [window_h, window_h, head_dim] where first dim is query pos, second is key pos
    rel_pos_h_emb = rel_pos_h[rel_coords_h.flatten()].reshape(
        window_h, window_h, head_dim
    )
    # rel_pos_w_emb shape: [window_w, window_w, head_dim]
    rel_pos_w_emb = rel_pos_w[rel_coords_w.flatten()].reshape(
        window_w, window_w, head_dim
    )
    
    # Reshape query for bias computation
    # query shape: [batch_windows, num_heads, window_h * window_w, head_dim]
    query_for_bias = query.reshape(
        batch_windows, num_attention_heads, window_h, window_w, head_dim
    )
    
    # Compute height bias: query_for_bias[b,n,qh,qw,c] * rel_pos_h_emb[qh,kh,c] -> [b,n,qh,qw,kh]
    # Using einsum with distinct subscripts
    rel_h = torch.einsum('bnijc,ikc->bnijk', query_for_bias, rel_pos_h_emb)
    # Compute width bias: query_for_bias[b,n,qh,qw,c] * rel_pos_w_emb[qw,kw,c] -> [b,n,qh,qw,kw]
    rel_w = torch.einsum('bnijc,jkc->bnijk', query_for_bias, rel_pos_w_emb)
    
    # Combine biases: [b,n,qh,qw,kh] + [b,n,qh,qw,kw] -> [b,n,qh,qw,kh,kw]
    rel_pos_bias = rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :]
    rel_pos_bias = rel_pos_bias.reshape(
        batch_windows, num_attention_heads, window_h * window_w, window_h * window_w
    )
    
    attn_weights = attn_weights + rel_pos_bias
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # Apply attention to values
    attn_output = attn_weights @ value
    attn_output = attn_output.transpose(1, 2).reshape(batch_windows, window_h, window_w, channels)
    
    # Output projection
    attn_output = F.linear(attn_output, proj_weight, proj_bias)
    
    # Window unpartition
    num_windows_h = pad_height // window_size
    num_windows_w = pad_width // window_size
    
    attn_output = attn_output.reshape(
        batch_size,
        num_windows_h,
        num_windows_w,
        window_size,
        window_size,
        -1
    )
    attn_output = attn_output.permute(0, 1, 3, 2, 4, 5).contiguous()
    attn_output = attn_output.reshape(batch_size, pad_height, pad_width, -1)
    
    # Remove padding
    attn_output = attn_output[:, :height, :width, :].contiguous()
    
    # First residual connection
    hidden_states = residual + attn_output
    
    # MLP block with second residual
    residual = hidden_states
    
    # Layer norm 2
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm2_weight + layer_norm2_bias
    
    # MLP
    hidden_states = F.linear(hidden_states, mlp_lin1_weight, mlp_lin1_bias)
    hidden_states = F.gelu(hidden_states)
    hidden_states = F.linear(hidden_states, mlp_lin2_weight, mlp_lin2_bias)
    
    # Second residual connection
    output = residual + hidden_states
    
    return output
