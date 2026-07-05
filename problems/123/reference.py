import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    norm_min: float,
):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_experts = 64
    top_k = 8
    
    # Flatten batch and sequence dimensions
    hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq_len, hidden_size]
    
    # === Shared Experts (process all tokens) ===
    shared_gate_out = torch.matmul(hidden_states_flat, shared_gate_proj.t())  # [batch*seq_len, 7168]
    shared_silu = F.silu(shared_gate_out)
    shared_up_out = torch.matmul(hidden_states_flat, shared_up_proj.t())  # [batch*seq_len, 7168]
    shared_intermediate = shared_silu * shared_up_out
    shared_output = torch.matmul(shared_intermediate, shared_down_proj.t())  # [batch*seq_len, hidden_size]
    
    # === Gating and Routing (float32 for stability) ===
    # Compute router logits
    router_logits = torch.matmul(hidden_states_flat.float(), gate_weight.t())  # [batch*seq_len, num_experts]
    
    # Apply softmax to get routing probabilities
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)  # [batch*seq_len, num_experts]
    
    # Apply correction bias
    routing_weights = routing_weights + e_score_correction_bias  # [batch*seq_len, num_experts]
    
    # Select top-k experts per token
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [batch*seq_len, top_k]
    
    # Gather routing weights for selected experts
    routing_weights_selected = torch.gather(routing_weights, dim=-1, index=selected_experts)  # [batch*seq_len, top_k]
    
    # Normalize routing weights with clamping
    routing_weights_normalized = routing_weights_selected / torch.clamp(
        routing_weights_selected.sum(dim=-1, keepdim=True), 
        min=norm_min
    )  # [batch*seq_len, top_k]
    
    # Convert back to bfloat16 for computation
    routing_weights_normalized = routing_weights_normalized.to(hidden_states.dtype)
    
    # === Token Dispatch and Expert Computation ===
    final_hidden_states = torch.zeros(
        (batch_size * seq_len, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Create expert mask for efficient dispatch
    # Shape: [num_experts, top_k, batch*seq_len]
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Process each expert that has at least one token assigned
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).squeeze(-1)
    
    for expert_idx in expert_hit:
        expert_idx_int = expert_idx.item()
        
        # Find which tokens are routed to this expert
        idx, top_x = torch.where(expert_mask[expert_idx_int].squeeze(0))  # idx: position in top_k, top_x: token indices
        
        if top_x.numel() == 0:
            continue
        
        # Gather tokens for this expert
        current_state = hidden_states_flat[top_x]  # [num_tokens_for_expert, hidden_size]
        
        # Expert MLP: gate_proj + silu + up_proj + down_proj
        gate_out = torch.matmul(current_state, expert_gate_proj[expert_idx_int].t())  # [num_tokens, intermediate_size]
        silu_out = F.silu(gate_out)
        up_out = torch.matmul(current_state, expert_up_proj[expert_idx_int].t())  # [num_tokens, intermediate_size]
        intermediate = silu_out * up_out
        expert_output = torch.matmul(intermediate, expert_down_proj[expert_idx_int].t())  # [num_tokens, hidden_size]
        
        # Apply routing weights
        weighted_output = expert_output * routing_weights_normalized[top_x, idx, None]  # [num_tokens, hidden_size]
        
        # Accumulate to final output (scatter-add operation)
        final_hidden_states.index_add_(0, top_x, weighted_output)
    
    # === Add Shared Expert Output ===
    final_hidden_states = final_hidden_states + shared_output
    
    # Reshape back to [batch_size, seq_len, hidden_size]
    output = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    
    return output, router_logits
