import torch

@torch.no_grad()
def run(x: torch.Tensor):
    """
    Fused FFT size padding and real FFT computation for Hyena convolution.
    
    Args:
        x: Input tensor of shape (batch, channels, seqlen)
        
    Returns:
        x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1)
        x_freq_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1)
    """
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    
    # Cast to float32 for FFT numerical stability (already float32 but explicit)
    x_f32 = x.to(torch.float32)
    
    # Perform real FFT with implicit zero-padding to fft_size
    # Output shape: (batch, channels, seqlen+1) complex
    x_freq = torch.fft.rfft(x_f32, n=fft_size)
    
    # Normalize by fft_size
    x_freq = x_freq / fft_size
    
    # Return real and imaginary parts separately since we can't have complex dtype in outputs
    x_freq_real = x_freq.real.contiguous()
    x_freq_imag = x_freq.imag.contiguous()
    
    return x_freq_real, x_freq_imag
