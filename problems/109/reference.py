import math

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time_dim = axes_and_scalars["time_dim"]
    d_model = 1024
    max_source_positions = 1500
    downsample_hidden_size = 384
    conv_out_dim = 3840  # 384 * 10
    kernel_size = 3
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return {
        "input_features": torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype),
        # Conv weights — Kaiming init
        "conv2d1_weight": kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size),
        "conv2d1_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d2_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d2_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d3_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d3_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        # Linear projection weight
        "conv_out_weight": xavier(d_model, conv_out_dim),
        # Sinusoidal positional embedding
        "positional_embedding": pe.to(dtype),
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
    b, c, f, t = x.size()
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    
    # Linear projection to d_model (no bias)
    x = F.linear(x, conv_out_weight)
    
    # Scale embeddings
    x = x * embed_scale
    
    # Add positional embeddings
    seq_len = x.shape[1]
    pos_embed = positional_embedding[:seq_len, :].unsqueeze(0)
    x = x + pos_embed
    
    return x
