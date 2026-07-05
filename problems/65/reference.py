import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Fused Conv1d projection and split.
    
    Args:
        x: Input tensor of shape [batch_size, in_channels, time]
        weight: Conv1d weight of shape [out_channels*2, in_channels, kernel_size]
        bias: Conv1d bias of shape [out_channels*2]
    
    Returns:
        mean: Mean statistics of shape [batch_size, out_channels, time]
        logs: Log-variance statistics of shape [batch_size, out_channels, time]
    """
    # Perform Conv1d projection to 2x channels
    # For kernel_size=1, no padding needed
    # Shape: [batch_size, out_channels*2, time]
    stats = F.conv1d(x, weight, bias, padding=0)
    
    # Split along channel dimension into mean and logs
    out_channels = weight.shape[0] // 2
    mean, logs = torch.split(stats, out_channels, dim=1)
    
    return mean, logs
