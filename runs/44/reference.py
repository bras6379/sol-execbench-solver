import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE expert computation."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    # Hidden states - random bfloat16
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    # Expert IDs - each token selects num_experts_per_tok unique experts
    expert_ids = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        expert_ids[i] = perm

    # Expert weights - normalized per token
    expert_weights_raw = torch.rand(
        num_tokens, num_experts_per_tok, dtype=torch.bfloat16, device=device
    ) + 0.1
    expert_weights = expert_weights_raw / expert_weights_raw.sum(dim=-1, keepdim=True)

    # Gate projection weights for all experts
    gate_proj_weights = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    # Up projection weights for all experts
    up_proj_weights = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    # Down projection weights for all experts
    down_proj_weights = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    return {
        "hidden_states": hidden_states,
        "expert_ids": expert_ids,
        "expert_weights": expert_weights,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
) -> torch.Tensor:
    """
    MoE expert computation with SwiGLU activation and multi-expert routing.

    For each token, routes to top-k assigned experts and computes:
    output = sum_k(weight_k * down_proj(silu(gate_proj(x)) * up_proj(x)))

    Args:
        hidden_states: (num_tokens, 7168) input hidden states
        expert_ids: (num_tokens, 8) expert assignments per token
        expert_weights: (num_tokens, 8) routing weights per token
        gate_proj_weights: (256, 2048, 7168) gate projection weights
        up_proj_weights: (256, 2048, 7168) up projection weights
        down_proj_weights: (256, 7168, 2048) down projection weights

    Returns:
        output: (num_tokens, 7168) weighted expert outputs
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = gate_proj_weights.shape[0]

    # Initialize output
    output = torch.zeros_like(hidden_states)

    # Create expert mask from expert_ids: [num_experts, num_experts_per_tok, num_tokens]
    expert_mask = F.one_hot(expert_ids, num_classes=num_experts).permute(2, 1, 0)

    # Process each expert's tokens
    for expert_id in range(num_experts):
        # Find which tokens are assigned to this expert and which slot
        idx, top_x = torch.where(expert_mask[expert_id])

        if top_x.numel() == 0:
            continue

        # Get tokens for this expert
        expert_tokens = hidden_states[top_x]  # (num_expert_tokens, hidden_size)

        # Get weights for this expert
        gate_w = gate_proj_weights[expert_id]  # (intermediate_size, hidden_size)
        up_w = up_proj_weights[expert_id]      # (intermediate_size, hidden_size)
        down_w = down_proj_weights[expert_id]  # (hidden_size, intermediate_size)

        # Gate projection
        gate_output = torch.matmul(expert_tokens, gate_w.t())

        # SiLU activation
        gate_output = F.silu(gate_output)

        # Up projection
        up_output = torch.matmul(expert_tokens, up_w.t())

        # Element-wise multiply (SwiGLU)
        intermediate = gate_output * up_output

        # Down projection
        expert_output = torch.matmul(intermediate, down_w.t())

        # Apply routing weight and accumulate
        weights = expert_weights[top_x, idx].unsqueeze(-1)
        output.index_add_(0, top_x, (expert_output * weights).to(torch.bfloat16))

    return output
