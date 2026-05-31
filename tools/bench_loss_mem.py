"""Isolate the top-K KL loss-path memory saving (issue #49, reviewer).

A successful training rerun supports but does NOT isolate the loss fix (ordering /
split / allocator differ). This measures peak CUDA memory of the OLD vs NEW
topk_kl_loss on the SAME realistic [N, vocab] grad tensor — no training confounds —
so the ~full-vocab-fp32-copy saving is shown directly, not inferred.

Run on the GPU box when free (needs a few GB). Tiny + fast.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _old_loss(student_logits, teacher_logits, k):
    topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
    t_denom = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
    teacher_logp = topk_vals.float() - t_denom
    teacher_p = teacher_logp.exp()
    s_denom = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    student_logq = student_logits.float().gather(-1, topk_idx) - s_denom
    return (teacher_p * (teacher_logp - student_logq)).sum(dim=-1).mean()


def _new_loss(student_logits, teacher_logits, k):
    topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
    t_denom = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    teacher_logp = (topk_vals - t_denom).float()
    teacher_p = teacher_logp.exp()
    s_denom = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    student_logq = (student_logits.gather(-1, topk_idx) - s_denom).float()
    return (teacher_p * (teacher_logp - student_logq)).sum(dim=-1).mean()


def _measure(fn, n, vocab, k):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    student = torch.randn(n, vocab, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(n, vocab, device="cuda", dtype=torch.bfloat16)
    before = torch.cuda.memory_allocated() / 1e9
    loss = fn(student, teacher, k)
    loss.backward()  # retains the graph tensors the training loop would
    peak = torch.cuda.max_memory_allocated() / 1e9
    val = float(loss.detach())
    del student, teacher, loss
    return {"loss": val, "peak_gb": peak, "base_tensors_gb": before}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(0)
    old = _measure(_old_loss, a.n, a.vocab, a.k)
    new = _measure(_new_loss, a.n, a.vocab, a.k)
    print(f"N={a.n} vocab={a.vocab} k={a.k}")
    print(f"OLD: peak={old['peak_gb']:.3f}GB  (base inputs {old['base_tensors_gb']:.3f}GB)")
    print(f"NEW: peak={new['peak_gb']:.3f}GB  (base inputs {new['base_tensors_gb']:.3f}GB)")
    print(f"SAVING: {old['peak_gb'] - new['peak_gb']:.3f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
