import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
):
    """
    Fused Mamba conv1d with gating.
    
    Args:
        hidden_states: (batch_size, seq_len, 8192)
        attention_mask: (batch_size, seq_len)
        in_proj_weight: (32768, 8192)
        in_proj_bias: (32768,)
        conv1d_weight: (16384, 1, 4) - depthwise conv
        conv1d_bias: (16384,)
    
    Returns:
        output_hidden_states: (batch_size, 16384, seq_len)
        gate: (batch_size, 16384, seq_len)
    """
    batch_size, seq_len, _ = hidden_states.shape
    intermediate_size = 16384
    conv_kernel_size = 4
    
    # 1. Gated linear projection: (B, L, 8192) -> (B, L, 32768)
    projected_states = F.linear(hidden_states, in_proj_weight, in_proj_bias)
    
    # 2. Transpose for conv1d: (B, L, 32768) -> (B, 32768, L)
    projected_states = projected_states.transpose(1, 2)
    
    # 3. Split into hidden states and gate: (B, 32768, L) -> 2x (B, 16384, L)
    hidden_states_conv, gate = projected_states.chunk(2, dim=1)
    
    # 4. Apply attention mask before convolution
    hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
    
    # 5. Causal 1D convolution with grouped convolution (depthwise)
    # Pad on the left for causal convolution
    hidden_states_padded = F.pad(hidden_states_conv, (conv_kernel_size - 1, 0))
    
    # Depthwise conv1d: groups=intermediate_size
    hidden_states_conv = F.conv1d(
        hidden_states_padded,
        conv1d_weight,
        conv1d_bias,
        groups=intermediate_size
    )
    
    # 6. Apply SiLU activation: silu(x) = x * sigmoid(x)
    hidden_states_conv = hidden_states_conv * torch.sigmoid(hidden_states_conv)
    
    # 7. Apply attention mask after convolution
    hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
    
    return hidden_states_conv, gate
