import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    batch_seq_len = batch_size * seq_len
    
    # Hidden states
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.bfloat16, device=device
    )
    
    # Routing weights - normalized per token
    routing_weights = torch.rand(
        batch_seq_len, num_experts_per_tok,
        dtype=torch.float32, device=device
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    
    # Selected experts - random expert indices in [0, num_experts)
    # Each token selects num_experts_per_tok different experts
    selected_experts = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(batch_seq_len)
    ]).to(torch.int64)
    
    # Expert weights
    gate_proj_weights = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02
    
    up_proj_weights = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02
    
    down_proj_weights = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "selected_experts": selected_experts,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    """
    MoE expert parallel execution with weighted aggregation.
    
    For each expert:
    1. Find tokens assigned to this expert
    2. Gather those tokens
    3. Execute gated MLP: down(SiLU(gate(x)) * up(x))
    4. Weight by routing weights
    5. Scatter-add back to output positions
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = selected_experts.shape[1]
    
    # Flatten batch and sequence dimensions
    hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden]
    
    # Create expert mask: one-hot encoding of expert assignments
    # [batch*seq, top_k, num_experts] -> [num_experts, top_k, batch*seq]
    expert_mask = F.one_hot(
        selected_experts,
        num_classes=num_experts
    ).permute(2, 1, 0)  # [num_experts, top_k, batch*seq]
    
    # Initialize output accumulator
    final_hidden_states = torch.zeros(
        (batch_size * seq_len, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )
    
    # Process each expert
    for expert_idx in range(num_experts):
        # Find which tokens are assigned to this expert
        # idx: which top-k position, top_x: which token indices
        idx, top_x = torch.where(expert_mask[expert_idx])
        
        # Skip if no tokens assigned to this expert
        if top_x.shape[0] == 0:
            continue
        
        # Gather tokens assigned to this expert
        current_state = hidden_states_flat[top_x]  # [num_tokens, hidden]
        
        # Get expert weights
        gate_w = gate_proj_weights[expert_idx]  # [intermediate, hidden]
        up_w = up_proj_weights[expert_idx]      # [intermediate, hidden]
        down_w = down_proj_weights[expert_idx]  # [hidden, intermediate]
        
        # Execute gated MLP: down(SiLU(gate(x)) * up(x))
        # Convert to float32 for computation
        current_state_f32 = current_state.to(torch.float32)
        gate_w_f32 = gate_w.to(torch.float32)
        up_w_f32 = up_w.to(torch.float32)
        down_w_f32 = down_w.to(torch.float32)
        
        # gate_proj: [num_tokens, hidden] @ [hidden, intermediate] = [num_tokens, intermediate]
        gate_out = current_state_f32 @ gate_w_f32.t()
        
        # up_proj: [num_tokens, hidden] @ [hidden, intermediate] = [num_tokens, intermediate]
        up_out = current_state_f32 @ up_w_f32.t()
        
        # SiLU activation on gate output
        gate_activated = F.silu(gate_out)
        
        # Element-wise multiply
        intermediate = gate_activated * up_out
        
        # down_proj: [num_tokens, intermediate] @ [intermediate, hidden] = [num_tokens, hidden]
        expert_output = intermediate @ down_w_f32.t()
        
        # Weight expert output by routing weights
        # routing_weights[top_x, idx] gives the weight for each token-expert pair
        token_weights = routing_weights[top_x, idx].unsqueeze(1)  # [num_tokens, 1]
        weighted_output = expert_output * token_weights  # [num_tokens, hidden]
        
        # Scatter weighted results back to original positions
        final_hidden_states.index_add_(
            0,
            top_x,
            weighted_output.to(torch.bfloat16)
        )
    
    # Reshape back to original dimensions
    final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    
    return final_hidden_states
