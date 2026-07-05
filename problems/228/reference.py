import torch
import math


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or page_beg >= page_end:
            # No queries or KV for this batch element
            continue

        kv_len = page_end - page_beg
        pages = kv_indices[page_beg:page_end]

        # Since page_size=1, pages are token indices
        tok_idx = pages[:kv_len].to(torch.long)
        Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

        q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        q_pe_batch = q_pe[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_kpe]

        q_len = q_end - q_start

        for i in range(q_len):
            qn = q_nope_batch[i]  # [num_heads, head_dim_ckv]
            qp = q_pe_batch[i]  # [num_heads, head_dim_kpe]

            logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_heads, kv_len]
            logits_scaled = logits * sm_scale

            # Apply causal mask
            prefix_len = kv_len - q_len  # Number of previously cached tokens
            query_abs_pos = prefix_len + i  # Absolute position of current query
            
            causal_mask = torch.arange(kv_len, device=logits_scaled.device) > query_abs_pos
            logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))

            # Compute 2-base LSE
            lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_heads, L_tokens]
            out = attn @ Kc  # [num_heads, head_dim_ckv]
            output[q_start + i] = out.to(torch.bfloat16)

    return output, lse