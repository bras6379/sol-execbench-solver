import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Strided 2D convolution for spatial downsampling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Convolution weights of shape (out_channels, in_channels, kernel_size, kernel_size)
        bias: Bias tensor of shape (out_channels,)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_height, out_width)
        where out_height = (height + 2*padding - kernel_size) // stride + 1
        and out_width = (width + 2*padding - kernel_size) // stride + 1
    """
    # Fixed constants for NAFNet downsampling
    stride = 2
    padding = 1
    
    # Perform strided convolution
    output = F.conv2d(x, weight, bias, stride=stride, padding=padding)
    
    return output
