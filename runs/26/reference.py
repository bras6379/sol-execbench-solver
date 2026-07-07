import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    video: torch.Tensor,
    patch_projection_weight: torch.Tensor,
    patch_projection_bias: torch.Tensor,
    spatial_pos_embedding: torch.Tensor,
    temporal_pos_embedding: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
):
    """
    Video patch embedding projection.
    
    Args:
        video: (batch, frames, channels, height, width)
        patch_projection_weight: (hidden_size, in_channels, 1, patch_size, patch_size)
        patch_projection_bias: (hidden_size,)
        spatial_pos_embedding: (1, num_spatial_patches, hidden_size)
        temporal_pos_embedding: (1, num_frames, 1, hidden_size)
        norm_weight: (hidden_size,)
        norm_bias: (hidden_size,)
        eps: layer norm epsilon
    
    Returns:
        output: (batch, total_patches, hidden_size)
    """
    batch_size, frames, channels, height, width = video.shape
    hidden_size = patch_projection_weight.shape[0]
    patch_size = patch_projection_weight.shape[3]
    
    # Rearrange to (batch, channels, frames, height, width) for Conv3D
    video = video.permute(0, 2, 1, 3, 4)
    
    # Apply patch projection using Conv3D
    # (B, C, F, H, W) -> (B, hidden_size, F, H', W')
    patches = F.conv3d(
        video,
        patch_projection_weight,
        patch_projection_bias,
        stride=(1, patch_size, patch_size),
        padding=(0, 0, 0)
    )
    
    # Rearrange to (batch, frames, hidden_size, num_patches_h, num_patches_w)
    patches = patches.permute(0, 2, 1, 3, 4)
    
    # Get spatial dimensions
    batch_size, frames, hidden_size, num_h, num_w = patches.shape
    
    # Flatten spatial dimensions: (B, F, hidden_size, H', W') -> (B, F, H'*W', hidden_size)
    patches = patches.reshape(batch_size, frames, hidden_size, num_h * num_w)
    patches = patches.permute(0, 1, 3, 2)  # (B, F, num_spatial_patches, hidden_size)
    
    # Add spatial positional embeddings (broadcast across frames)
    # spatial_pos_embedding: (1, num_spatial_patches, hidden_size) -> broadcast to (B, F, S, H)
    patches = patches + spatial_pos_embedding.unsqueeze(1)
    
    # Add temporal positional embeddings (broadcast across spatial patches)
    # temporal_pos_embedding: (1, num_frames, 1, hidden_size) -> broadcast to (B, F, S, H)
    patches = patches + temporal_pos_embedding
    
    # Flatten temporal and spatial dimensions: (B, F, S, H) -> (B, F*S, H)
    patches = patches.reshape(batch_size, frames * num_h * num_w, hidden_size)
    
    # Apply layer normalization
    # Compute mean and variance along last dimension
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True, unbiased=False)
    patches_normalized = (patches - mean) / torch.sqrt(var + eps)
    output = patches_normalized * norm_weight + norm_bias
    
    return output
