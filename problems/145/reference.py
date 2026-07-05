import torch
import torch.nn.functional as F
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    d_model = 256
    d_inner = 1024
    order = 2
    l_max = 32768
    short_filter_order = 3
    filter_order = 64
    emb_dim = 5
    inner_width = d_model * (order + 1)
    
    hidden_states = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32, device=device)
    norm1_weight = torch.ones(d_model, dtype=torch.float32, device=device)
    norm1_bias = torch.zeros(d_model, dtype=torch.float32, device=device)
    norm2_weight = torch.ones(d_model, dtype=torch.float32, device=device)
    norm2_bias = torch.zeros(d_model, dtype=torch.float32, device=device)
    in_proj_weight = torch.randn(inner_width, d_model, dtype=torch.float32, device=device) * 0.02
    in_proj_bias = torch.randn(inner_width, dtype=torch.float32, device=device) * 0.02
    short_conv_weight = torch.randn(inner_width, 1, short_filter_order, dtype=torch.float32, device=device) * 0.02
    short_conv_bias = torch.randn(inner_width, dtype=torch.float32, device=device) * 0.02
    filter_linear1_weight = torch.randn(filter_order, emb_dim, dtype=torch.float32, device=device) * 0.02
    filter_linear1_bias = torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    sin_freq = torch.ones(1, filter_order, dtype=torch.float32, device=device)
    filter_linear2_weight = torch.randn(filter_order, filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear2_bias = torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear3_weight = torch.randn(filter_order, filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear3_bias = torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear_final_weight = torch.randn(d_model, filter_order, dtype=torch.float32, device=device) * 0.02
    filter_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02
    max_decay = math.log(0.01) / 0.3
    min_decay = math.log(0.01) / 1.5
    deltas = torch.linspace(min_decay, max_decay, d_model, device=device)[None, None, :]
    exp_mod_deltas = deltas.to(torch.float32)
    out_proj_weight = torch.randn(d_model, d_model, dtype=torch.float32, device=device) * 0.02
    out_proj_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02
    mlp_fc1_weight = torch.randn(d_inner, d_model, dtype=torch.float32, device=device) * 0.02
    mlp_fc1_bias = torch.randn(d_inner, dtype=torch.float32, device=device) * 0.02
    mlp_fc2_weight = torch.randn(d_model, d_inner, dtype=torch.float32, device=device) * 0.02
    mlp_fc2_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "norm1_weight": norm1_weight,
        "norm1_bias": norm1_bias,
        "norm2_weight": norm2_weight,
        "norm2_bias": norm2_bias,
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "short_conv_weight": short_conv_weight,
        "short_conv_bias": short_conv_bias,
        "filter_linear1_weight": filter_linear1_weight,
        "filter_linear1_bias": filter_linear1_bias,
        "sin_freq": sin_freq,
        "filter_linear2_weight": filter_linear2_weight,
        "filter_linear2_bias": filter_linear2_bias,
        "filter_linear3_weight": filter_linear3_weight,
        "filter_linear3_bias": filter_linear3_bias,
        "filter_linear_final_weight": filter_linear_final_weight,
        "filter_bias": filter_bias,
        "exp_mod_deltas": exp_mod_deltas,
        "out_proj_weight": out_proj_weight,
        "out_proj_bias": out_proj_bias,
        "mlp_fc1_weight": mlp_fc1_weight,
        "mlp_fc1_bias": mlp_fc1_bias,
        "mlp_fc2_weight": mlp_fc2_weight,
        "mlp_fc2_bias": mlp_fc2_bias,
        "layer_norm_eps": 1e-5,
        "exp_mod_shift": 0.05
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    short_conv_weight: torch.Tensor,
    short_conv_bias: torch.Tensor,
    filter_linear1_weight: torch.Tensor,
    filter_linear1_bias: torch.Tensor,
    sin_freq: torch.Tensor,
    filter_linear2_weight: torch.Tensor,
    filter_linear2_bias: torch.Tensor,
    filter_linear3_weight: torch.Tensor,
    filter_linear3_bias: torch.Tensor,
    filter_linear_final_weight: torch.Tensor,
    filter_bias: torch.Tensor,
    exp_mod_deltas: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    layer_norm_eps: float,
    exp_mod_shift: float,
):
    d_model = 256
    order = 2
    l_max = 32768
    inner_width = d_model * (order + 1)
    
    batch_size, seq_len, _ = hidden_states.shape
    l_filter = min(seq_len, l_max)
    device = hidden_states.device
    
    # First Residual + LayerNorm
    residual = hidden_states.to(torch.float32)
    mean = residual.mean(dim=-1, keepdim=True)
    var = residual.var(dim=-1, keepdim=True, unbiased=False)
    normed = (residual - mean) / torch.sqrt(var + layer_norm_eps)
    normed = normed * norm1_weight + norm1_bias
    
    # Hyena Operator - Input projection
    u = F.linear(normed, in_proj_weight, in_proj_bias)
    u = u.transpose(1, 2)
    
    # Short depthwise convolution
    u_padded = F.pad(u, (2, 2))
    uc = F.conv1d(u_padded, short_conv_weight, short_conv_bias, groups=inner_width)
    uc = uc[..., :l_filter]
    
    # Split into x and v
    splits = uc.split(d_model, dim=1)
    x = splits[:-1]
    v = splits[-1]
    
    # Implicit Filter Generation - Positional embeddings
    t = torch.linspace(0, 1, l_filter, device=device)[None, :, None]
    bands = 2
    t_rescaled = torch.linspace(0, l_filter - 1, l_filter, device=device)[None, :, None]
    w = 2 * math.pi * t_rescaled / l_filter
    f = torch.linspace(1e-4, bands - 1, bands, device=device)[None, None]
    z = torch.cat([t, torch.cos(-f * w), torch.sin(-f * w)], dim=-1)
    
    # Filter MLP
    h = F.linear(z, filter_linear1_weight, filter_linear1_bias)
    h = torch.sin(sin_freq * h)
    h = F.linear(h, filter_linear2_weight, filter_linear2_bias)
    h = torch.sin(sin_freq * h)
    h = F.linear(h, filter_linear3_weight, filter_linear3_bias)
    h = torch.sin(sin_freq * h)
    h = F.linear(h, filter_linear_final_weight, None)
    
    # Exponential modulation
    decay = torch.exp(-t * exp_mod_deltas.abs())
    h = h * (decay + exp_mod_shift)
    
    # Add bias and reshape for FFT convolution
    h = h + filter_bias.view(1, 1, d_model)
    k = h.transpose(0, 1).reshape(1, d_model, l_filter)
    bias_reshaped = filter_bias.reshape(1, d_model)
    
    # Gating and FFT Convolution (order=2, one iteration)
    for o, x_i in enumerate(reversed(x[1:])):
        v = v * x_i
        fft_size = 2 * l_filter
        k_f = torch.fft.rfft(k[o].to(torch.float32), n=fft_size) / fft_size
        v_f = torch.fft.rfft(v.to(torch.float32), n=fft_size)
        y = torch.fft.irfft(v_f * k_f, n=fft_size, norm='forward')[..., :l_filter]
        v = y + v * bias_reshaped[o].unsqueeze(-1)
    
    # Final gating with x[0]
    y = (v * x[0]).transpose(1, 2)
    
    # Pad back to original seq_len if needed
    if l_filter < seq_len:
        y = F.pad(y, (0, 0, 0, seq_len - l_filter))
    
    # Output projection
    hyena_out = F.linear(y, out_proj_weight, out_proj_bias)
    
    # First Residual Addition
    residual = hyena_out + residual
    
    # Second LayerNorm
    residual_float = residual.to(torch.float32)
    mean = residual_float.mean(dim=-1, keepdim=True)
    var = residual_float.var(dim=-1, keepdim=True, unbiased=False)
    normed = (residual_float - mean) / torch.sqrt(var + layer_norm_eps)
    normed = normed * norm2_weight + norm2_bias
    
    # MLP
    mlp_out = F.linear(normed, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = F.gelu(mlp_out, approximate="tanh")
    mlp_out = F.linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)
    
    # Final Residual Addition
    output = mlp_out + residual_float
    
    return output
