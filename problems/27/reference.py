import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars['batch_size']
    num_frames = axes_and_scalars['num_frames']
    num_patches_per_frame = axes_and_scalars['num_patches_per_frame']
    hidden_size = 1024
    head_dim = 64
    
    hidden_states = torch.randn(batch_size, num_frames, num_patches_per_frame, hidden_size, device=device, dtype=torch.float32)
    qkv_weight = torch.randn(3 * hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    qkv_bias = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
    out_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    out_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    rope_theta = 10000.0
    temporal_freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    spatial_freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    scale = 1.0 / math.sqrt(head_dim)
    
    return {
        'hidden_states': hidden_states,
        'qkv_weight': qkv_weight,
        'qkv_bias': qkv_bias,
        'out_weight': out_weight,
        'out_bias': out_bias,
        'temporal_freqs': temporal_freqs,
        'spatial_freqs': spatial_freqs,
        'scale': scale,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    temporal_freqs: torch.Tensor,
    spatial_freqs: torch.Tensor,
    scale: float,
):
    batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
    seq_len = num_frames * num_patches
    num_attention_heads = 16
    head_dim = 64
    
    # Generate position indices
    device = hidden_states.device
    frame_positions = torch.arange(num_frames, device=device).unsqueeze(1).expand(num_frames, num_patches).reshape(-1).float()
    
    # Assume square patches for height/width positions
    patches_per_side = int(math.sqrt(num_patches))
    if patches_per_side * patches_per_side != num_patches:
        patches_per_side = int(math.ceil(math.sqrt(num_patches)))
    
    height_idx = torch.arange(num_patches, device=device) // patches_per_side
    width_idx = torch.arange(num_patches, device=device) % patches_per_side
    height_positions = height_idx.unsqueeze(0).expand(num_frames, num_patches).reshape(-1).float()
    width_positions = width_idx.unsqueeze(0).expand(num_frames, num_patches).reshape(-1).float()
    
    # Reshape to (batch, seq_len, hidden_size)
    hidden_states_flat = hidden_states.reshape(batch_size, seq_len, hidden_size)
    
    # QKV projection: (batch, seq_len, 3 * hidden_size)
    qkv = F.linear(hidden_states_flat, qkv_weight, qkv_bias)
    
    # Reshape and split into Q, K, V
    qkv = qkv.reshape(batch_size, seq_len, 3, num_attention_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply 3D RoPE to Q and K
    def apply_rope_3d(x, frame_pos, height_pos, width_pos, temporal_freqs, spatial_freqs):
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        temporal_dim = head_dim // 3
        spatial_dim = (head_dim - temporal_dim) // 2
        remaining_dim = head_dim - temporal_dim - 2 * spatial_dim
        
        x_temporal = x[..., :temporal_dim]
        x_height = x[..., temporal_dim:temporal_dim + spatial_dim]
        x_width = x[..., temporal_dim + spatial_dim:temporal_dim + 2 * spatial_dim]
        x_remaining = x[..., temporal_dim + 2 * spatial_dim:]
        
        def apply_rope_rotation(x_part, positions, freqs):
            dim = x_part.shape[-1]
            half_dim = dim // 2
            if half_dim == 0:
                return x_part
            angles = positions.unsqueeze(-1) * freqs[:half_dim]
            cos_vals = torch.cos(angles)
            sin_vals = torch.sin(angles)
            
            x1 = x_part[..., 0::2]
            x2 = x_part[..., 1::2]
            
            min_dim = min(x1.shape[-1], cos_vals.shape[-1])
            x1 = x1[..., :min_dim]
            x2 = x2[..., :min_dim]
            cos_vals = cos_vals[..., :min_dim]
            sin_vals = sin_vals[..., :min_dim]
            
            x_rotated_1 = x1 * cos_vals.unsqueeze(0).unsqueeze(0) - x2 * sin_vals.unsqueeze(0).unsqueeze(0)
            x_rotated_2 = x1 * sin_vals.unsqueeze(0).unsqueeze(0) + x2 * cos_vals.unsqueeze(0).unsqueeze(0)
            
            x_out = torch.stack([x_rotated_1, x_rotated_2], dim=-1)
            x_out = x_out.flatten(-2)
            
            if dim % 2 == 1:
                x_out = torch.cat([x_out, x_part[..., -1:]], dim=-1)
            return x_out
        
        x_temporal = apply_rope_rotation(x_temporal, frame_pos, temporal_freqs)
        x_height = apply_rope_rotation(x_height, height_pos, spatial_freqs)
        x_width = apply_rope_rotation(x_width, width_pos, spatial_freqs)
        
        return torch.cat([x_temporal, x_height, x_width, x_remaining], dim=-1)
    
    q = apply_rope_3d(q, frame_positions, height_positions, width_positions, temporal_freqs, spatial_freqs)
    k = apply_rope_3d(k, frame_positions, height_positions, width_positions, temporal_freqs, spatial_freqs)
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Softmax
    attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    
    # Apply attention to values
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    
    # Output projection
    output = F.linear(attn_output, out_weight, out_bias)
    
    # Reshape back to original shape
    output = output.reshape(batch_size, num_frames, num_patches, hidden_size)
    
    return output
