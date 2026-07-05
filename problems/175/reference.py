import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    Sparse MoE expert computation with top-k routing.
    
    1. Compute routing scores via sigmoid activation
    2. Apply group-wise expert selection (n_group=1, topk_group=1 simplifies to global topk)
    3. Select top-8 experts per token
    4. Compute SwiGLU for each selected expert
    5. Combine with shared expert output
    """
    # Constants
    n_routed_experts = 128
    num_experts_per_tok = 8
    n_group = 1
    topk_group = 1
    norm_topk_prob = True
    
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    
    # Router logits: [num_tokens, n_routed_experts]
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    
    # Sigmoid activation for scores
    scores = torch.sigmoid(router_logits)
    
    # Apply score correction bias
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    
    # Group-wise expert selection
    experts_per_group = n_routed_experts // n_group
    group_scores = scores_for_choice.view(-1, n_group, experts_per_group)
    
    # Select top-2 experts per group and sum their scores
    group_scores_agg = group_scores.topk(2, dim=-1)[0].sum(dim=-1)  # [num_tokens, n_group]
    
    # Select top groups
    group_idx = torch.topk(group_scores_agg, k=topk_group, dim=-1, sorted=False)[1]
    
    # Create mask for selected groups
    group_mask = torch.zeros_like(group_scores_agg)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand mask to all experts
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, experts_per_group)
        .reshape(-1, n_routed_experts)
    )
    
    # Mask out non-selected groups
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    
    # Select top-k experts
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    
    # Normalize weights if required
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    
    # Apply routing scaling factor
    topk_weights = topk_weights * routed_scaling_factor
    topk_weights = topk_weights.to(hidden_states.dtype)
    
    # Compute routed expert outputs
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    
    # Create expert mask: [n_routed_experts, num_tokens, num_experts_per_tok]
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)  # [n_routed_experts, num_tokens, num_experts_per_tok]
    
    # Process each expert
    for expert_idx in range(n_routed_experts):
        mask = expert_mask[expert_idx]  # [num_tokens, num_experts_per_tok]
        token_indices, weight_indices = torch.where(mask)
        
        if token_indices.numel() > 0:
            # Get routing weights for this expert
            expert_weights = topk_weights[token_indices, weight_indices]
            
            # Get input tokens for this expert
            expert_input = hidden_states[token_indices]
            
            # Expert MLP computation: SwiGLU activation
            gate_output = F.silu(
                F.linear(expert_input, expert_gate_weights[expert_idx])
            )
            up_output = F.linear(expert_input, expert_up_weights[expert_idx])
            intermediate = gate_output * up_output
            expert_output = F.linear(intermediate, expert_down_weights[expert_idx])
            
            # Apply routing weights
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Accumulate to final output
            final_hidden_states.index_add_(0, token_indices, weighted_output)
    
    final_hidden_states = final_hidden_states.to(hidden_states.dtype)
    
    # Compute shared expert output
    shared_gate_output = F.silu(F.linear(hidden_states, shared_gate_weight))
    shared_up_output = F.linear(hidden_states, shared_up_weight)
    shared_intermediate = shared_gate_output * shared_up_output
    shared_output = F.linear(shared_intermediate, shared_down_weight)
    
    # Combine routed and shared experts
    output = final_hidden_states + shared_output
    
    return output
