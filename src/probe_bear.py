"""
Run the BEAR (small) log-likelihood probe on a HuggingFace causal LM.

For each (subject, relation, gold_answer_idx) triple, we score every label
in the relation's answer space using a length-normalised log-likelihood
ranking. Accuracy and acquired bits are aggregated per relation and across
the dataset.

Output: results/bear/<model_name>.json with per-relation breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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

DATA_ROOT = Path(__file__).parent.parent / "datasets" / "bear"
META_ROOT = Path(__file__).parent.parent / "code" / "BEAR" / "BEAR"
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "bear"


def load_relations() -> Dict[str, dict]:
    """Load relation -> {templates, answer_space_labels, answer_space_ids}."""
    with open(META_ROOT / "metadata_relations.json") as f:
        return json.load(f)


def fill_template(template: str, subject: str) -> str:
    """Replace [X] with subject and strip the trailing [Y]."""
    # Templates look like "The native language of [X] is [Y]."
    # We feed the model "The native language of <subject> is" and let it
    # complete with the answer label.
    # Strip everything from [Y] onwards (incl. trailing punctuation).
    if "[Y]" not in template:
        return template.replace("[X]", subject)
    prefix, _ = template.split("[Y]", 1)
    return prefix.replace("[X]", subject).rstrip()


def load_relation_facts(relation_id: str, subset: str = "small") -> List[dict]:
    """Load fact list for a relation from BEAR small or big."""
    folder = "BEAR_small" if subset == "small" else "BEAR_big"
    p = DATA_ROOT / folder / f"{relation_id}.jsonl"
    if not p.exists():
        return []
    facts = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            facts.append(json.loads(line))
    return facts


def run_bear_probe(
    model_name: str,
    subset: str = "small",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    max_facts_per_relation: int | None = None,
) -> dict:
    set_seed(42)
    print(f"\n=== Loading {model_name} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    relations = load_relations()
    per_relation = {}
    grand_correct = 0
    grand_total = 0
    grand_acquired = 0.0

    rel_iter = list(relations.items())

    t_start = time.time()
    for ri, (rel_id, rel_meta) in enumerate(rel_iter):
        templates = rel_meta.get("templates", [])
        answer_space = rel_meta.get("answer_space_labels", [])
        if not templates or not answer_space:
            continue
        # Use the first template for efficiency. (Templates are paraphrases;
        # one is sufficient for ranking accuracy and far cheaper.)
        template = templates[0]
        facts = load_relation_facts(rel_id, subset=subset)
        if not facts:
            continue
        if max_facts_per_relation is not None:
            facts = facts[:max_facts_per_relation]

        n_correct = 0
        per_fact = []
        for fact in tqdm(facts, desc=f"{rel_id} ({ri + 1}/{len(rel_iter)})", leave=False):
            sub = fact["sub_label"]
            gold = fact["answer_idx"]
            prompt = fill_template(template, sub)
            scores = score_candidates(
                model, tokenizer, prompt, answer_space,
                device=device, batch_size=24,
            )
            pred = rank_predict(scores)
            ok = (pred == gold)
            if ok:
                n_correct += 1
            per_fact.append({"sub_id": fact.get("sub_id", ""), "gold": gold, "pred": pred, "correct": ok})
        n_total = len(facts)
        a_bits = acquired_bits(n_correct, n_total, len(answer_space))
        per_relation[rel_id] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": n_correct / n_total if n_total else 0.0,
            "answer_space_size": len(answer_space),
            "acquired_bits": a_bits,
            "template": template,
        }
        grand_correct += n_correct
        grand_total += n_total
        grand_acquired += a_bits
        # Periodic log
        elapsed = time.time() - t_start
        print(
            f"  [{ri+1}/{len(rel_iter)}] {rel_id}: acc={per_relation[rel_id]['accuracy']:.3f} "
            f"({n_correct}/{n_total}, |A|={len(answer_space)}, +{a_bits:.1f} bits, total {elapsed:.0f}s)",
            flush=True,
        )

    summary = {
        "model_name": model_name,
        "subset": subset,
        "n_params": n_params,
        "grand_correct": grand_correct,
        "grand_total": grand_total,
        "grand_accuracy": grand_correct / grand_total if grand_total else 0.0,
        "total_acquired_bits": grand_acquired,
        "param_share_2bpp": parameter_share(grand_acquired, n_params, 2.0),
        "param_share_3p6bpp": parameter_share(grand_acquired, n_params, 3.6),
        "per_relation": per_relation,
        "elapsed_seconds": time.time() - t_start,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", default="small", choices=["small", "big"])
    parser.add_argument("--max_facts", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        slug = args.model.replace("/", "__")
        out = RESULTS_ROOT / f"{slug}__bear_{args.subset}.json"
    else:
        out = Path(args.out)

    res = run_bear_probe(args.model, subset=args.subset, max_facts_per_relation=args.max_facts)
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
