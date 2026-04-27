"""
Run a categorical-knowledge probe on the TAXI baseline-evaluation set.

For each row we have:
  query_fwd / query_rev: a query template with `<subj>` and `<answer>` slots
  fwd_choices: list of candidate answers (e.g. ['dog', 'cat', ...])
  answer_fwd:  the gold answer label
  rev_choices, answer_rev: analogous reverse direction

We construct a prompt by replacing `<subj>` with the row's `subj` and
substituting each candidate for `<answer>`, then score the resulting
sentences via length-normalised log-likelihood over the candidate token span.

Output: results/taxi/<model_name>.json with per-property breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(__file__))
from probe_utils import (
    acquired_bits,
    parameter_share,
    rank_predict,
    score_candidates,
    set_seed,
)

DATA_FILE = Path(__file__).parent.parent / "datasets" / "taxi" / "data" / "baseline-evaluation.json"
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "taxi"


def load_taxi_baseline() -> List[dict]:
    """Convert TAXI's column-oriented json into a list-of-rows."""
    with open(DATA_FILE) as f:
        d = json.load(f)
    keys = list(d.keys())
    n = len(d[keys[0]])
    # row keys can be string indices "0".."n-1"
    rows = []
    for i in range(n):
        sk = str(i)
        row = {k: d[k][sk] for k in keys}
        rows.append(row)
    return rows


def fill_query(query: str, subj: str, candidate: str) -> tuple[str, str]:
    """Return (prompt, completion) by splitting at `<answer>`.

    The prompt is everything before <answer>; the completion is the candidate
    (we drop everything *after* <answer> to make scoring tokenization-clean,
    same as BEAR template handling)."""
    q = query.replace("<subj>", subj)
    if "<answer>" not in q:
        return (q, candidate)
    prefix, _ = q.split("<answer>", 1)
    return (prefix.rstrip(), candidate)


def run_taxi_probe(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    direction: str = "fwd",
    max_rows: int | None = None,
) -> dict:
    set_seed(42)
    print(f"\n=== Loading {model_name} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    rows = load_taxi_baseline()
    if max_rows is not None:
        rows = rows[:max_rows]
    print(f"  rows: {len(rows)}", flush=True)

    per_property: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "answer_set_sizes": [], "category": ""})
    per_category: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "bits": 0.0})
    grand_correct = 0
    grand_total = 0
    grand_bits = 0.0

    t_start = time.time()
    for i, row in enumerate(tqdm(rows, desc="TAXI")):
        if direction == "fwd":
            query = row.get("query_fwd")
            choices = row.get("fwd_choices") or []
            gold = row.get("answer_fwd")
        else:
            query = row.get("query_rev")
            choices = row.get("rev_choices") or []
            gold = row.get("answer_rev")
        if not query or not choices or gold is None:
            continue
        if gold not in choices:
            continue
        gold_idx = choices.index(gold)

        # Build prompts: prompt = prefix-before-<answer>; candidates = choices
        # The prefix is identical across choices, so call score_candidates with
        # the same prefix and varying completions.
        prompt = query.replace("<subj>", row["subj"]).split("<answer>")[0].rstrip()

        # Candidates may be multi-word; we score each as a free-form completion.
        scores = score_candidates(
            model, tokenizer, prompt, choices,
            device=device, batch_size=24,
        )
        pred_idx = rank_predict(scores)
        ok = (pred_idx == gold_idx)

        prop = row.get("property", "?")
        cat = row.get("superordinate_category", "?")
        per_property[prop]["n"] += 1
        per_property[prop]["correct"] += int(ok)
        per_property[prop]["answer_set_sizes"].append(len(choices))
        per_property[prop]["category"] = cat
        per_category[cat]["n"] += 1
        per_category[cat]["correct"] += int(ok)

        # Per-fact bits acquired (if correct, using its own answer-set size)
        # We accumulate a per-property baseline correction at the end.
        grand_total += 1
        if ok:
            grand_correct += 1

    # Compute per-property acquired bits using majority answer-set size
    for prop, d in per_property.items():
        avg_a = float(sum(d["answer_set_sizes"]) / max(1, len(d["answer_set_sizes"])))
        bits = acquired_bits(d["correct"], d["n"], int(round(avg_a)) if avg_a >= 2 else 2)
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0
        d["avg_answer_set_size"] = avg_a
        d["acquired_bits"] = bits
        grand_bits += bits
        per_category[d["category"]]["bits"] += bits

    summary = {
        "model_name": model_name,
        "direction": direction,
        "n_params": n_params,
        "grand_correct": grand_correct,
        "grand_total": grand_total,
        "grand_accuracy": grand_correct / grand_total if grand_total else 0.0,
        "total_acquired_bits": grand_bits,
        "param_share_2bpp": parameter_share(grand_bits, n_params, 2.0),
        "param_share_3p6bpp": parameter_share(grand_bits, n_params, 3.6),
        "per_property": dict(per_property),
        "per_category": dict(per_category),
        "elapsed_seconds": time.time() - t_start,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--direction", default="fwd", choices=["fwd", "rev"])
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        slug = args.model.replace("/", "__")
        out = RESULTS_ROOT / f"{slug}__taxi_{args.direction}.json"
    else:
        out = Path(args.out)

    res = run_taxi_probe(args.model, direction=args.direction, max_rows=args.max_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {out}")
    print(
        f"Grand accuracy {res['grand_accuracy']:.3f}  bits {res['total_acquired_bits']:.1f}  "
        f"share@2bpp {res['param_share_2bpp']*100:.4f}%  share@3.6bpp {res['param_share_3p6bpp']*100:.4f}%"
    )


if __name__ == "__main__":
    main()
