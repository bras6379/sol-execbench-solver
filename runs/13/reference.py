import torch

@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, normalized: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor):
    # Convert grad_output to float32 for computation
    grad_output_f32 = grad_output.to(torch.float32)
    
    # Gradient with respect to weight
    # dy/dweight = normalized (summed over batch and sequence dimensions)
    grad_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))
    
    # Gradient with respect to normalized output
    # dy/dnormalized = weight
    grad_normalized = grad_output_f32 * weight
    
    # Gradient with respect to x (before normalization)
    # For RMSNorm: norm = x * rstd where rstd = 1/sqrt(mean(x^2) + eps)
    # dnorm/dx = rstd * (I - norm * x^T / hidden_size)
    # Therefore: grad_x = rstd * (grad_normalized - mean(grad_normalized * normalized) * normalized)
    
    # Compute mean(grad_normalized * normalized) over hidden dimension
    mean_grad_norm = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
    
    # Gradient through normalization
    grad_x = rstd * (grad_normalized - mean_grad_norm * normalized)
    
    # Convert back to input dtype (bfloat16)
    grad_x_bf16 = grad_x.to(torch.bfloat16)
    
    # Both hidden_states and residual receive the same gradient
    grad_hidden_states = grad_x_bf16
    grad_residual = grad_x_bf16.clone()
    
    return grad_hidden_states, grad_residual, grad_weight
