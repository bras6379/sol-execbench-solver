import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    Group-limited top-k expert routing.
    
    Two-stage selection:
    1. Organize 256 experts into 8 groups of 32
    2. Compute group scores by summing top-2 expert scores within each group
    3. Select top-4 groups based on group scores
    4. Mask out experts from non-selected groups
    5. Select top-8 experts from remaining candidates
    6. Normalize weights and apply scaling factor
    """
    # Constants
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group  # 32
    
    num_tokens = hidden_states.shape[0]
    
    # Compute routing logits: [num_tokens, 256]
    logits = F.linear(
        hidden_states.to(torch.float32),
        weight.to(torch.float32)
    )
    
    # Apply sigmoid activation to get routing scores
    scores = torch.sigmoid(logits)  # [num_tokens, 256]
    
    # Add learned expert bias for routing adjustment
    scores_for_routing = scores + expert_bias.to(torch.float32)
    
    # Step 1: Reshape scores into groups [num_tokens, 8, 32]
    group_scores_reshaped = scores_for_routing.view(num_tokens, n_group, experts_per_group)
    
    # Step 2: Get top-2 scores within each group and sum them
    # [num_tokens, 8, 2] -> [num_tokens, 8]
    top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=-1)  # [num_tokens, 8]
    
    # Step 3: Select top-4 groups based on aggregated scores
    # [num_tokens, 4]
    _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)
    
    # Step 4: Create group mask [num_tokens, 8]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    
    # Step 5: Expand mask to expert level [num_tokens, 256]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )
    
    # Step 6: Mask out experts from non-selected groups
    neg_inf = torch.finfo(torch.float32).min
    masked_scores = scores_for_routing.masked_fill(score_mask == 0, neg_inf)
    
    # Step 7: Select top-8 experts from masked scores
    _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)
    
    # Gather selected expert scores (use original scores without bias)
    selected_scores = torch.gather(scores, dim=1, index=topk_idx)
    
    # Normalize routing weights
    topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
    
    # Apply routing scaling factor
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight
