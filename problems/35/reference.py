import torch

@torch.no_grad()
def run(emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    AdaLayerNormZero modulation parameter extraction.
    
    Performs a linear projection followed by chunking into 6 modulation parameters.
    
    Args:
        emb: Timestep embeddings of shape [batch_size, inner_dim]
        weight: Linear weight matrix of shape [6 * inner_dim, inner_dim]
        bias: Linear bias of shape [6 * inner_dim]
        
    Returns:
        Tuple of 6 tensors, each of shape [batch_size, inner_dim]:
        - shift_msa, scale_msa, gate_msa: for attention path
        - shift_mlp, scale_mlp, gate_mlp: for MLP path
    """
    # Linear projection: [batch_size, inner_dim] @ [inner_dim, 6*inner_dim] + bias
    # = [batch_size, 6*inner_dim]
    emb_out = torch.matmul(emb, weight.t()) + bias
    
    # Chunk into 6 equal parts along the last dimension
    # Each chunk has shape [batch_size, inner_dim]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=1)
    
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
