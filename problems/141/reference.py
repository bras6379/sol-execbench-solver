import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE training token repeat and expert computation."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Hidden states - random input tokens
    hidden_states = torch.randn(batch_seq_len, hidden_size, device=device, dtype=torch.float16)
    
    # TopK indices - each token selects num_experts_per_tok unique experts
    topk_idx = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(batch_seq_len)
    ]).to(torch.int64)
    
    # TopK weights - normalized routing weights (softmax-like)
    topk_weight_raw = torch.rand(batch_seq_len, num_experts_per_tok, device=device, dtype=torch.float16)
    topk_weight = topk_weight_raw / topk_weight_raw.sum(dim=-1, keepdim=True)
    
    # Expert weights - small initialization
    expert_gate_projs = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) * 0.02
    expert_up_projs = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) * 0.02
    expert_down_projs = torch.randn(
        num_experts, hidden_size, moe_intermediate_size, device=device, dtype=torch.float16
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "topk_idx": topk_idx,
        "topk_weight": topk_weight,
        "expert_gate_projs": expert_gate_projs,
        "expert_up_projs": expert_up_projs,
        "expert_down_projs": expert_down_projs,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
) -> torch.Tensor:
    """
    MoE training forward pass with token repetition and masked expert computation.
    
    Steps:
    1. Repeat each token num_experts_per_tok times
    2. Process each expert using masked indexing (SwiGLU FFN)
    3. Reshape and apply routing weights
    4. Sum across experts
    
    Args:
        hidden_states: [batch_seq_len, hidden_size]
        topk_idx: [batch_seq_len, num_experts_per_tok] - selected expert indices
        topk_weight: [batch_seq_len, num_experts_per_tok] - routing weights
        expert_gate_projs: [num_experts, moe_intermediate_size, hidden_size]
        expert_up_projs: [num_experts, moe_intermediate_size, hidden_size]
        expert_down_projs: [num_experts, hidden_size, moe_intermediate_size]
    
    Returns:
        output: [batch_seq_len, hidden_size]
    """
    batch_seq_len, hidden_size = hidden_states.shape
    num_experts = expert_gate_projs.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    
    # Step 1: Repeat tokens for each selected expert
    # [batch_seq_len, hidden_size] -> [batch_seq_len * num_experts_per_tok, hidden_size]
    hidden_states_repeated = hidden_states.repeat_interleave(num_experts_per_tok, dim=0)
    
    # Flatten expert indices for masking
    # [batch_seq_len, num_experts_per_tok] -> [batch_seq_len * num_experts_per_tok]
    flat_topk_idx = topk_idx.view(-1)
    
    # Step 2: Initialize output tensor
    y = torch.zeros_like(hidden_states_repeated)
    
    # Step 3: Process each expert using masked indexing
    for expert_idx in range(num_experts):
        # Create boolean mask for tokens assigned to this expert
        expert_mask = (flat_topk_idx == expert_idx)
        
        if expert_mask.any():
            # Get tokens for this expert
            expert_input = hidden_states_repeated[expert_mask]
            
            # Expert computation: SwiGLU FFN
            # gate_proj(x) * silu(up_proj(x)) -> down_proj
            gate_output = F.linear(expert_input, expert_gate_projs[expert_idx])
            up_output = F.linear(expert_input, expert_up_projs[expert_idx])
            
            # SwiGLU: silu(gate) * up
            intermediate = F.silu(gate_output) * up_output
            
            expert_output = F.linear(intermediate, expert_down_projs[expert_idx])
            
            # Write back to output using mask
            y[expert_mask] = expert_output
    
    # Step 4: Reshape and apply routing weights
    # [batch_seq_len * num_experts_per_tok, hidden_size] ->
    # [batch_seq_len, num_experts_per_tok, hidden_size]
    y = y.view(batch_seq_len, num_experts_per_tok, hidden_size)
    
    # Apply routing weights and sum across experts
    # [batch_seq_len, num_experts_per_tok, 1] * [batch_seq_len, num_experts_per_tok, hidden_size]
    # -> [batch_seq_len, hidden_size]
    output = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
    
    return output
