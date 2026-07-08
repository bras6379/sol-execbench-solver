import torch


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    n_routed_experts = axes_and_scalars["n_routed_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Gradient inputs
    grad_tokens_per_expert = torch.randn(n_routed_experts, dtype=torch.float32, device=device)
    grad_expert_mask = torch.randn(batch_seq_len, n_routed_experts, dtype=torch.float32, device=device)
    grad_load_balance_loss = torch.randn(1, dtype=torch.float32, device=device)
    
    # Saved tensors from forward pass
    topk_idx = torch.randint(0, n_routed_experts, (batch_seq_len, num_experts_per_tok), dtype=torch.int64, device=device)
    
    # Create expert_mask from topk_idx (as done in forward)
    expert_mask = torch.zeros((batch_seq_len, n_routed_experts), dtype=torch.int64, device=device)
    expert_mask.scatter_(1, topk_idx, 1)
    
    # Compute tokens_per_expert from expert_mask
    tokens_per_expert = expert_mask.sum(dim=0).to(torch.int32)
    
    # Training flag as 1-element bool tensor
    training = torch.tensor([True], dtype=torch.bool, device=device)
    
    return {
        "grad_tokens_per_expert": grad_tokens_per_expert,
        "grad_expert_mask": grad_expert_mask,
        "grad_load_balance_loss": grad_load_balance_loss,
        "topk_idx": topk_idx,
        "expert_mask": expert_mask,
        "tokens_per_expert": tokens_per_expert,
        "training": training,
    }


@torch.no_grad()
def run(
    grad_tokens_per_expert: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_load_balance_loss: torch.Tensor,
    topk_idx: torch.Tensor,
    expert_mask: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    training: torch.Tensor,
):
    """
    Backward pass for MoE expert load balancing.
    
    Computes gradients through:
    1. load_balance_loss computation
    2. sum reduction (tokens_per_expert = expert_mask.sum(dim=0))
    3. scatter operation (discrete, no gradient for topk_idx)
    
    Returns accumulated gradient for expert_mask.
    """
    batch_seq_len = topk_idx.shape[0]
    n_routed_experts = 256
    num_experts_per_tok = 8
    
    # Extract scalar values from tensors
    grad_loss_val = grad_load_balance_loss.item()
    is_training = training.item()
    
    # Initialize output gradient
    grad_expert_mask_out = grad_expert_mask.clone()
    
    # Gradient from load_balance_loss
    # load_balance_loss = n_routed_experts * sum(expert_fraction * uniform_prob)
    # where expert_fraction = tokens_per_expert / (batch_seq_len * num_experts_per_tok)
    # and uniform_prob = 1.0 / n_routed_experts
    #
    # d(load_balance_loss)/d(tokens_per_expert) = n_routed_experts * uniform_prob / (batch_seq_len * num_experts_per_tok)
    #                                            = 1 / (batch_seq_len * num_experts_per_tok)
    
    grad_tpe_accumulated = grad_tokens_per_expert.clone()
    
    if is_training:
        uniform_prob = 1.0 / n_routed_experts
        grad_from_loss = grad_loss_val * n_routed_experts * uniform_prob / (
            batch_seq_len * num_experts_per_tok
        )
        grad_tpe_accumulated = grad_tpe_accumulated + grad_from_loss
    
    # Gradient through sum operation
    # tokens_per_expert = expert_mask.sum(dim=0)
    # d(loss)/d(expert_mask) = d(loss)/d(tokens_per_expert).unsqueeze(0).expand_as(expert_mask)
    
    # Broadcast gradient from [n_routed_experts] to [batch_seq_len, n_routed_experts]
    grad_expert_mask_from_sum = grad_tpe_accumulated.unsqueeze(0).expand(
        batch_seq_len, n_routed_experts
    ).float()
    
    grad_expert_mask_out = grad_expert_mask_out + grad_expert_mask_from_sum
    
    return grad_expert_mask_out


if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={
            "batch_seq_len": 512,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        },
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
