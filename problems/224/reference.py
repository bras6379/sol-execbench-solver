import torch
import math


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]
    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    # Flatten page dimension since page_size=1
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            # No queries or KV for this batch element
            continue

        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        
        # Number of KV tokens is equal to number of pages for page_size=1
        num_kv_tokens = page_ids.shape[0]
        k_batch = k_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]
        
        # Get queries for this sequence
        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
        num_q_tokens = q_batch.shape[0]

        # Delta for causal masking
        delta = num_kv_tokens - num_q_tokens

        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx

            # Apply causal mask
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue

            q_pos = q_batch[q_idx]  # [num_qo_heads, head_dim]

            for h in range(num_qo_heads):
                # Find corresponding KV head for GQA
                kv_head = h // gqa_ratio

                q_head = q_pos[h]  # [head_dim]
                k_head = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, head_dim]
                v_head = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, head_dim]

                logits = torch.matmul(q_head, k_head.T)  # [max_kv_idx]
                logits_scaled = logits * sm_scale

                # Compute 2-base LSE
                lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                attn = torch.softmax(logits_scaled, dim=-1)  # [max_kv_idx]
                out_head = torch.matmul(attn, v_head)  # [head_dim]
                output[global_q_idx, h] = out_head.to(torch.bfloat16)

    return output, lse