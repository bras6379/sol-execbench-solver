import torch


def get_inputs(
    axes_and_scalars: dict[str, int], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE routing backward pass."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_experts = axes_and_scalars["num_experts"]
    top_k = axes_and_scalars["top_k"]
    
    # Hidden states from forward pass
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Gate weight
    gate_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    
    # Router logits from forward pass
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
    
    # Routing probs (softmax of router_logits)
    routing_probs = torch.softmax(router_logits.float(), dim=1)
    
    # Selected experts (top-k indices)
    _, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
    
    # Routing weights sum (for normalization backward)
    routing_weights, _ = torch.topk(routing_probs, top_k, dim=-1)
    routing_weights_sum = routing_weights.sum(dim=-1, keepdim=True)
    
    # Gradients
    grad_routing_weights = torch.randn(num_tokens, top_k, dtype=torch.bfloat16, device=device) * 0.1
    grad_expert_mask = torch.randn(num_experts, top_k, num_tokens, dtype=torch.bfloat16, device=device) * 0.01
    grad_router_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device) * 0.1
    
    return {
        "hidden_states": hidden_states,
        "gate_weight": gate_weight,
        "router_logits": router_logits,
        "routing_probs": routing_probs,
        "selected_experts": selected_experts,
        "routing_weights_sum": routing_weights_sum,
        "grad_routing_weights": grad_routing_weights,
        "grad_expert_mask": grad_expert_mask,
        "grad_router_logits": grad_router_logits,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits: torch.Tensor,
    routing_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights_sum: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for MoE sparse routing.
    
    Computes gradients through:
    1. Normalization (quotient rule)
    2. Top-k selection (straight-through estimator)
    3. Softmax
    4. Linear projection
    """
    num_tokens = hidden_states.shape[0]
    num_experts = gate_weight.shape[0]
    top_k = selected_experts.shape[1]
    input_dtype = hidden_states.dtype
    
    # Initialize gradient accumulator for routing probabilities
    grad_routing_probs = torch.zeros_like(routing_probs)  # float32
    
    # Process grad_routing_weights through normalization backward
    grad_routing_weights_f32 = grad_routing_weights.float()
    
    # Gradient through normalization: d(w_i / sum(w)) / d(w_i)
    # Gather the routing weights before normalization
    routing_weights_unnorm = torch.gather(routing_probs, dim=1, index=selected_experts)  # [num_tokens, top_k]
    
    # Compute gradient w.r.t. unnormalized weights using quotient rule
    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm / routing_weights_sum).sum(
        dim=-1, keepdim=True
    )
    grad_routing_weights_unnorm = (
        grad_routing_weights_f32 / routing_weights_sum - 
        grad_sum / routing_weights_sum
    )
    
    # Gradient through top-k selection (straight-through estimator)
    # Scatter gradients back to selected expert positions
    grad_routing_probs.scatter_(
        dim=1,
        index=selected_experts,
        src=grad_routing_weights_unnorm
    )
    
    # Process grad_expert_mask (auxiliary gradient path)
    # grad_expert_mask shape: (num_experts, top_k, num_tokens)
    # Permute: (num_experts, top_k, num_tokens) -> (num_tokens, top_k, num_experts)
    grad_expert_mask_permuted = grad_expert_mask.permute(2, 1, 0).float()
    
    # Gather gradients for selected experts
    grad_from_mask = torch.gather(
        grad_expert_mask_permuted,
        dim=2,
        index=selected_experts.unsqueeze(2)
    ).squeeze(2)
    
    # Scatter back to routing_probs space
    grad_routing_probs.scatter_add_(
        dim=1,
        index=selected_experts,
        src=grad_from_mask
    )
    
    # Process grad_router_logits (direct gradient path)
    grad_routing_probs = grad_routing_probs + grad_router_logits.float()
    
    # Gradient through softmax
    # d(softmax)/d(logits) = softmax * (grad - sum(grad * softmax))
    dot_product = (grad_routing_probs * routing_probs).sum(dim=1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    
    # Convert back to input dtype
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)
    
    # Gradient through linear projection
    # grad_hidden_states = grad_router_logits @ gate_weight
    # Shape: (num_tokens, num_experts) @ (num_experts, hidden_size)
    grad_hidden_states = torch.matmul(
        grad_router_logits_computed,
        gate_weight
    )
    
    # grad_gate_weight = grad_router_logits.T @ hidden_states
    # Shape: (num_experts, num_tokens) @ (num_tokens, hidden_size)
    grad_gate_weight = torch.matmul(
        grad_router_logits_computed.t(),
        hidden_states
    )
    
    return grad_hidden_states, grad_gate_weight
