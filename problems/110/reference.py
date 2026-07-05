import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj_weights: torch.Tensor,
    expert_up_proj_weights: torch.Tensor,
    expert_down_proj_weights: torch.Tensor,
):
    """
    Complete MoE expert computation with routing and expert MLPs.
    
    Args:
        hidden_states: (batch_size, seq_len, 4096)
        gate_weight: (64, 4096) - gate routing weights
        expert_gate_proj_weights: (64, 2560, 4096) - gate proj for all experts
        expert_up_proj_weights: (64, 2560, 4096) - up proj for all experts
        expert_down_proj_weights: (64, 4096, 2560) - down proj for all experts
    
    Returns:
        final_hidden_states: (batch_size, seq_len, 4096)
        router_logits: (batch_size * seq_len, 64)
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_experts = 64
    top_k = 8
    
    # Flatten for routing: (batch_size * sequence_length, 4096)
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    
    # Gate routing: (batch_size * sequence_length, 64)
    router_logits = F.linear(hidden_states_flat, gate_weight)
    
    # Softmax routing weights: (batch_size * sequence_length, 64)
    routing_weights = F.softmax(router_logits.float(), dim=1, dtype=torch.float32)
    
    # Top-k expert selection: (batch_size * sequence_length, 8)
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Normalize top-k probabilities
    routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
    
    # Cast back to input dtype
    routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)
    
    # Initialize output accumulator
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # One-hot encode selected experts: (64, 8, batch_size * sequence_length)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Process each expert (sparse computation)
    for expert_idx in range(num_experts):
        # Find tokens routed to this expert
        expert_mask_flat = expert_mask[expert_idx]  # (8, batch_size * sequence_length)
        idx, top_x = torch.where(expert_mask_flat)
        
        if top_x.numel() == 0:
            continue  # Skip unused experts
        
        # Get expert weights
        gate_proj_w = expert_gate_proj_weights[expert_idx]  # (2560, 4096)
        up_proj_w = expert_up_proj_weights[expert_idx]  # (2560, 4096)
        down_proj_w = expert_down_proj_weights[expert_idx]  # (4096, 2560)
        
        # Index tokens for this expert
        current_state = hidden_states_flat[top_x]  # (num_tokens_for_expert, 4096)
        
        # SwiGLU: gate_proj(x) * silu(up_proj(x)) -> down_proj
        gate_out = F.linear(current_state, gate_proj_w)  # (num_tokens, 2560)
        up_out = F.linear(current_state, up_proj_w)  # (num_tokens, 2560)
        
        # SiLU activation on gate output, multiply with up output
        silu_gate = gate_out / (1.0 + torch.exp(-gate_out.float())).to(gate_out.dtype)
        intermediate = silu_gate * up_out  # (num_tokens, 2560)
        
        # Down projection
        current_hidden_states = F.linear(intermediate, down_proj_w)  # (num_tokens, 4096)
        
        # Weight by routing probabilities
        current_hidden_states = current_hidden_states * routing_weights_topk[top_x, idx, None]
        
        # Accumulate into output (in-place addition)
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    
    # Reshape back to original shape
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    
    return final_hidden_states, router_logits.to(hidden_states.dtype)
