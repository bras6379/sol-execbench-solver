import torch

@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    """
    Joint attention context projection.
    
    1. Concatenate image and context attention outputs along sequence dimension
    2. Duplicate along feature dimension (creating [features, features] pattern)
    3. Project through linear layer
    4. Split back into image and context streams
    
    Args:
        image_attention_output: (batch_size, image_seq_len, inner_dim)
        context_attention_output: (batch_size, context_seq_len, inner_dim)
        to_out_weight: (inner_dim, 2*inner_dim)
        to_out_bias: (inner_dim,)
    
    Returns:
        projected_image: (batch_size, image_seq_len, inner_dim)
        projected_context: (batch_size, context_seq_len, inner_dim)
    """
    batch_size = image_attention_output.shape[0]
    image_seq_len = image_attention_output.shape[1]
    context_seq_len = context_attention_output.shape[1]
    
    # Concatenate image and context attention outputs along sequence dimension
    # Shape: (batch_size, image_seq_len + context_seq_len, inner_dim)
    combined_attention = torch.cat(
        [image_attention_output, context_attention_output],
        dim=1
    )
    
    # Duplicate along feature dimension for joint processing
    # Shape: (batch_size, image_seq_len + context_seq_len, 2 * inner_dim)
    combined_features = torch.cat(
        [combined_attention, combined_attention],
        dim=-1
    )
    
    # Project through output layer: linear(x) = x @ W^T + b
    # Shape: (batch_size, image_seq_len + context_seq_len, inner_dim)
    projected_output = torch.matmul(combined_features, to_out_weight.t()) + to_out_bias
    
    # Split back into image and context streams
    projected_image = projected_output[:, :image_seq_len, :]
    projected_context = projected_output[:, image_seq_len:, :]
    
    return projected_image, projected_context
