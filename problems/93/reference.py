import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    N = axes_and_scalars["N"]
    hidden_size = 5120
    n_routed_experts = 160
    top_k = 8
    routed_scaling_factor = 2.5
    
    # Gradient from upstream
    grad_topk_weights = torch.randn(N, top_k, dtype=torch.bfloat16, device=device)
    
    # Saved tensors from forward pass
    hidden_states = torch.randn(N, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n_routed_experts, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    
    # Scores are sigmoid outputs, so in (0, 1)
    router_logits = torch.randn(N, n_routed_experts, dtype=torch.bfloat16, device=device)
    scores = torch.sigmoid(router_logits)
    
    # topk_indices: each row has top_k unique indices in [0, n_routed_experts)
    topk_indices = torch.stack([
        torch.randperm(n_routed_experts, device=device)[:top_k]
        for _ in range(N)
    ]).to(torch.int64)
    
    # topk_weights: gathered from scores
    topk_weights = scores.gather(1, topk_indices)
    
    # Normalization
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights_normalized = topk_weights / denominator
    
    return {
        "grad_topk_weights": grad_topk_weights,
        "hidden_states": hidden_states,
        "weight": weight,
        "scores": scores,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "topk_weights_normalized": topk_weights_normalized,
        "denominator": denominator,
        "routed_scaling_factor": routed_scaling_factor,
    }


@torch.no_grad()
def run(
    grad_topk_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_weights_normalized: torch.Tensor,
    denominator: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    Backward pass for grouped top-k MoE routing.
    
    Gradient flow:
    1. grad through scaling: grad_normalized = grad_output * scaling_factor
    2. grad through normalization: uses quotient rule
    3. grad through gather (scatter): scatter gradients to full scores tensor
    4. grad through sigmoid: grad * scores * (1 - scores)
    5. grad through linear: matmul for both hidden_states and weight gradients
    """
    N = hidden_states.shape[0]
    n_routed_experts = scores.shape[1]
    
    # Step 1: Gradient through scaling
    # topk_weights_scaled = topk_weights_normalized * routed_scaling_factor
    grad_topk_weights_normalized = grad_topk_weights * routed_scaling_factor  # (N, 8)
    
    # Step 2: Gradient through normalization
    # topk_weights_normalized = topk_weights / denominator
    # Using quotient rule for gradient
    grad_sum = (grad_topk_weights_normalized * topk_weights_normalized).sum(dim=-1, keepdim=True)  # (N, 1)
    grad_topk_weights_unnorm = (grad_topk_weights_normalized - grad_sum) / denominator  # (N, 8)
    
    # Step 3: Gradient through gather operation (scatter in backward)
    # topk_weights = scores.gather(1, topk_indices)
    grad_scores = torch.zeros(N, n_routed_experts, dtype=grad_topk_weights_unnorm.dtype,
                              device=grad_topk_weights_unnorm.device)  # (N, 160)
    grad_scores.scatter_add_(1, topk_indices, grad_topk_weights_unnorm)  # (N, 160)
    
    # Step 4: Gradient through sigmoid
    # scores = sigmoid(router_logits)
    # d_sigmoid/d_x = sigmoid(x) * (1 - sigmoid(x)) = scores * (1 - scores)
    grad_router_logits = grad_scores * scores * (1.0 - scores)  # (N, 160)
    
    # Step 5: Gradient through linear projection
    # router_logits = hidden_states @ weight.T
    
    # grad_hidden_states = grad_router_logits @ weight
    grad_hidden_states = torch.matmul(
        grad_router_logits,  # (N, 160)
        weight  # (160, 5120)
    )  # (N, 5120)
    
    # grad_weight = grad_router_logits.T @ hidden_states
    grad_weight = torch.matmul(
        grad_router_logits.t(),  # (160, N)
        hidden_states  # (N, 5120)
    )  # (160, 5120)
    
    return grad_hidden_states, grad_weight
