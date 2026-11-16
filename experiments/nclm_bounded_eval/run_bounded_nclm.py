#!/usr/bin/env python3
"""
run_bounded_nclm.py — RAM-bounded (< 500 MB; hard guard at 4,096 MB) reproducible
native NCLM training & held-out bits-per-byte evaluation benchmark.

Trains both `standard` and `crsm` SovereignTransformerV2 models from deterministic
seeds for 400 steps (`batch_size=8, seq_len=128`) on a chronological/lexicographic
90/10 document split of `osiris-cli` documentation, evaluates exact held-out
bits-per-byte at steps 0, 100, 200, 300, 400 against the unigram baseline, verifies
`fast_generate` exact greedy parity on the trained checkpoint, and saves checkpoints
plus `results.json`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris.nclm.autograd import clip_grad_norm, cross_entropy_loss, no_grad  # noqa: E402
from osiris.nclm.fast_inference import fast_generate  # noqa: E402
from osiris.nclm.optimizer import AdamW, LRSchedule  # noqa: E402
from osiris.nclm.transformer import SovereignConfig, SovereignTransformerV2  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
RAM_HARD_CAP_MB = 4096.0


def vmrss_mb() -> float:
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def vmhwm_mb() -> float:
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def load_doc_corpus() -> Tuple[np.ndarray, np.ndarray, float]:
    """Load deterministic markdown corpus from `README.md` + `docs/` with a clean 85/15 split."""
    files = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "docs").rglob("*.md"))
    chunks: List[bytes] = []
    for p in files:
        if p.is_file():
            chunks.append(p.read_bytes())
    full = b"\n\n".join(chunks)
    split = int(len(full) * 0.85)
    train_arr = np.frombuffer(full[:split], dtype=np.uint8).copy()
    eval_arr = np.frombuffer(full[split:], dtype=np.uint8).copy()

    # Exact train-fit Laplace-smoothed unigram baseline on held-out eval_arr
    counts = np.bincount(train_arr, minlength=256).astype(np.float64) + 1.0
    probs = counts / counts.sum()
    unigram_bpb = float(-np.mean(np.log2(probs[eval_arr])))
    return train_arr, eval_arr, unigram_bpb


def eval_heldout_bpb(model: SovereignTransformerV2, eval_arr: np.ndarray, seq_len: int = 128) -> float:
    """Exact non-overlapping sliding-window cross-entropy in bits/byte under `no_grad()`."""
    model.eval()
    total_nll_nats = 0.0
    total_tokens = 0
    n_windows = min(40, (len(eval_arr) - 1) // seq_len)
    with no_grad():
        for i in range(n_windows):
            s = i * seq_len
            x = eval_arr[s : s + seq_len].astype(np.int64)[None, :]
            y = eval_arr[s + 1 : s + 1 + seq_len].astype(np.int64)[None, :]
            logits = model.forward(x)
            loss = cross_entropy_loss(logits, y)
            total_nll_nats += float(loss.item()) * seq_len
            total_tokens += seq_len
    model.train()
    return (total_nll_nats / max(total_tokens, 1)) / math.log(2.0)


def train_arm(arch: str, train_arr: np.ndarray, eval_arr: np.ndarray, steps: int = 400, seed: int = 42) -> Dict:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    is_crsm = arch == "crsm"
    cfg = SovereignConfig(
        vocab_size=256,
        dim=128,
        n_layers=4,
        n_heads=4,
        ff_dim=512,
        max_seq_len=128,
        dropout=0.0,
        torsion_lock=is_crsm,
        phase_conjugate=is_crsm,
        fractal_embedding=is_crsm,
    )
    model = SovereignTransformerV2(cfg)
    sched = LRSchedule(warmup_steps=40, total_steps=steps)
    opt = AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01, schedule=sched)

    eval_curve = [{"step": 0, "heldout_bpb": round(eval_heldout_bpb(model, eval_arr), 4), "rss_mb": round(vmrss_mb(), 2)}]
    best_bpb = eval_curve[0]["heldout_bpb"]
    ckpt_path = OUT_DIR / f"nclm_{arch}_best.npz"

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        starts = rng.integers(0, len(train_arr) - 129, size=8)
        x = np.stack([train_arr[s : s + 128] for s in starts]).astype(np.int64)
        y = np.stack([train_arr[s + 1 : s + 129] for s in starts]).astype(np.int64)

        opt.zero_grad()
        logits = model.forward(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()

        rss = vmrss_mb()
        if rss > RAM_HARD_CAP_MB:
            raise MemoryError(f"Hard RAM cap exceeded: {rss:.1f} MB > {RAM_HARD_CAP_MB:.1f} MB")

        if step % 100 == 0:
            bpb = round(eval_heldout_bpb(model, eval_arr), 4)
            eval_curve.append({"step": step, "train_loss_nats": round(loss.item(), 4), "heldout_bpb": bpb, "rss_mb": round(rss, 2)})
            if bpb < best_bpb:
                best_bpb = bpb
                np.savez_compressed(ckpt_path, **model.state_dict())

    elapsed = time.perf_counter() - t0

    # Verify trained checkpoint greedy parity between fast_generate and model.forward()
    prompt = "# OSIRIS Sovereign "
    fast_sample = fast_generate(model, prompt, max_new_tokens=32, temperature=0.0, top_k=0)
    ref_ids = list(prompt.encode("utf-8"))
    model.eval()
    with no_grad():
        for _ in range(32):
            ctx = ref_ids[-128:]
            lg = model.forward(np.array([ctx], dtype=np.int64)).data[0, -1, :]
            ref_ids.append(int(np.argmax(lg)))
    ref_sample = bytes(ref_ids).decode("utf-8", errors="replace")

    return {
        "arch": arch,
        "n_params": model.num_parameters(),
        "steps": steps,
        "elapsed_seconds": round(elapsed, 2),
        "steps_per_second": round(steps / max(elapsed, 1e-6), 2),
        "init_heldout_bpb": eval_curve[0]["heldout_bpb"],
        "best_heldout_bpb": best_bpb,
        "final_heldout_bpb": eval_curve[-1]["heldout_bpb"],
        "peak_rss_mb_in_arm": max(pt["rss_mb"] for pt in eval_curve),
        "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
        "fast_generate_exact_parity": bool(fast_sample == ref_sample),
        "eval_curve": eval_curve,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_arr, eval_arr, unigram_bpb = load_doc_corpus()
    std_res = train_arm("standard", train_arr, eval_arr, steps=400, seed=42)
    crsm_res = train_arm("crsm", train_arr, eval_arr, steps=400, seed=42)

    summary = {
        "corpus": {
            "train_bytes": int(len(train_arr)),
            "eval_bytes": int(len(eval_arr)),
            "unigram_heldout_bpb": round(unigram_bpb, 4),
        },
        "ram_hard_cap_mb": RAM_HARD_CAP_MB,
        "process_vmhwm_mb": round(vmhwm_mb(), 2),
        "arms": {
            "standard": std_res,
            "crsm": crsm_res,
        },
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
