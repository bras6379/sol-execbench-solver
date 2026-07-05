import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    norm_topk_prob: bool,
):
    """
    Sparse MoE routing and expert dispatch with top-k gating.
    
    1. Gate network computes routing logits for all 128 experts
    2. Top-8 experts are selected per token via softmax + topk
    3. Routing weights are normalized across selected experts
    4. Tokens are dynamically dispatched to experts
    5. Expert outputs are weighted and aggregated
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_experts = gate_weight.shape[0]
    top_k = 8
    
    # Flatten batch and sequence dimensions for routing
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    num_tokens = hidden_states_flat.shape[0]
    
    # Step 1: Compute routing logits via gating network
    # (num_tokens, hidden_size) @ (hidden_size, num_experts) -> (num_tokens, num_experts)
    router_logits = torch.matmul(hidden_states_flat, gate_weight.t())
    
    # Step 2: Compute routing weights via softmax
    routing_weights = F.softmax(router_logits.float(), dim=1).to(hidden_states.dtype)
    
    # Step 3: Select top-k experts per token
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Step 4: Normalize routing weights across selected experts (if enabled)
    if norm_topk_prob:
        routing_weights_topk = routing_weights_topk / (routing_weights_topk.sum(dim=-1, keepdim=True) + 1e-9)
    
    # Step 5: Initialize output accumulator
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Step 6: Create expert mask for efficient dispatching
    # One-hot encode selected experts: (num_tokens, top_k, num_experts)
    # Then permute to: (num_experts, top_k, num_tokens)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Step 7: Loop over experts and compute outputs for assigned tokens
    for expert_idx in range(num_experts):
        # Find which tokens are routed to this expert and at which top-k position
        idx, top_x = torch.where(expert_mask[expert_idx])
        
        if top_x.numel() == 0:
            continue
        
        # Gather hidden states for tokens assigned to this expert
        current_state = hidden_states_flat[top_x]
        
        # Expert MLP forward: SiLU(gate_proj(x)) * up_proj(x), then down_proj
        gate_out = torch.matmul(current_state, expert_gate_proj[expert_idx].t())
        up_out = torch.matmul(current_state, expert_up_proj[expert_idx].t())
        
        # SiLU activation: x * sigmoid(x)
        silu_gate = gate_out * torch.sigmoid(gate_out)
        intermediate = silu_gate * up_out
        
        expert_out = torch.matmul(intermediate, expert_down_proj[expert_idx].t())
        
        # Weight by routing probability
        current_hidden_states = expert_out * routing_weights_topk[top_x, idx, None]
        
        # Scatter-add expert outputs back to final output
        final_hidden_states.index_add_(0, top_x, current_hidden_states)
    
    # Step 8: Reshape output back to original dimensions
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    
    return final_hidden_states
