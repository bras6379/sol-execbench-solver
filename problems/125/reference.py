import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    inner_dim = axes_and_scalars["inner_dim"]
    pooled_projection_dim = axes_and_scalars["pooled_projection_dim"]
    time_embed_dim = axes_and_scalars["time_embed_dim"]
    half_time_embed_dim = axes_and_scalars["half_time_embed_dim"]
    
    # Timestep values in [0, 1] range
    timestep = torch.rand(batch_size, dtype=torch.float32, device=device)
    
    # Pooled text projections
    pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=torch.float32, device=device)
    
    # Precomputed frequency tensor for sinusoidal embeddings
    # Frequencies: 10000^(-2i/d) for i in [0, d/2)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, time_embed_dim, 2, dtype=torch.float32, device=device) / time_embed_dim
    )
    
    # Timestep MLP weights
    timestep_linear1_weight = torch.randn(inner_dim, time_embed_dim, dtype=torch.float32, device=device) * 0.02
    timestep_linear1_bias = torch.zeros(inner_dim, dtype=torch.float32, device=device)
    timestep_linear2_weight = torch.randn(inner_dim, inner_dim, dtype=torch.float32, device=device) * 0.02
    timestep_linear2_bias = torch.zeros(inner_dim, dtype=torch.float32, device=device)
    
    # Text embedder weights
    text_embedder_weight = torch.randn(inner_dim, pooled_projection_dim, dtype=torch.float32, device=device) * 0.02
    text_embedder_bias = torch.zeros(inner_dim, dtype=torch.float32, device=device)
    
    return {
        "timestep": timestep,
        "pooled_projections": pooled_projections,
        "freqs": freqs,
        "timestep_linear1_weight": timestep_linear1_weight,
        "timestep_linear1_bias": timestep_linear1_bias,
        "timestep_linear2_weight": timestep_linear2_weight,
        "timestep_linear2_bias": timestep_linear2_bias,
        "text_embedder_weight": text_embedder_weight,
        "text_embedder_bias": text_embedder_bias,
    }

@torch.no_grad()
def run(
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    freqs: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
):
    # Scale timesteps by 1000 (FLUX convention)
    timestep_scaled = timestep * 1000.0
    
    # Generate sinusoidal embeddings
    # timestep_scaled shape: (batch_size,)
    # freqs shape: (half_time_embed_dim,)
    # Compute arguments: timestep_scaled[:, None] * freqs[None, :]
    args = timestep_scaled[:, None] * freqs[None, :]  # (batch_size, half_time_embed_dim)
    
    # Compute sin and cos
    sin_embed = torch.sin(args)
    cos_embed = torch.cos(args)
    
    # Concatenate cos and sin: [cos, sin]
    timestep_embed = torch.cat([cos_embed, sin_embed], dim=-1)  # (batch_size, time_embed_dim)
    
    # Timestep MLP: Linear -> SiLU -> Linear
    # First linear layer
    x = torch.nn.functional.linear(timestep_embed, timestep_linear1_weight, timestep_linear1_bias)
    
    # SiLU activation: x * sigmoid(x)
    x = x * torch.sigmoid(x)
    
    # Second linear layer
    timestep_embed = torch.nn.functional.linear(x, timestep_linear2_weight, timestep_linear2_bias)
    
    # Project pooled text embeddings
    text_embed = torch.nn.functional.linear(pooled_projections, text_embedder_weight, text_embedder_bias)
    
    # Combine timestep and text embeddings
    conditioning = timestep_embed + text_embed
    
    return conditioning
