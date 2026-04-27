"""
Orchestrator: run BEAR-small + TAXI probes across a list of HF models on
designated GPUs. Designed to run sequentially within a single process so we
can fully release GPU memory between models.

Usage: python src/run_all.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from probe_bear import run_bear_probe
from probe_taxi import run_taxi_probe

ROOT = Path(__file__).parent.parent
RESULTS_ROOT = ROOT / "results"

MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
]


def slug(m: str) -> str:
    return m.replace("/", "__")


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Pin to GPU 0 by default (single-process orchestration).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    sweep_summary = {}
    for m in MODELS:
        m_slug = slug(m)
        print(f"\n{'#' * 60}\n# Running on {m}\n{'#' * 60}")
        # BEAR
        bear_out = RESULTS_ROOT / "bear" / f"{m_slug}__bear_small.json"
        if bear_out.exists():
            print(f"  BEAR result already exists at {bear_out}; loading.")
            with open(bear_out) as f:
                bear_res = json.load(f)
        else:
            t0 = time.time()
            bear_res = run_bear_probe(m, subset="small")
            bear_out.parent.mkdir(parents=True, exist_ok=True)
            with open(bear_out, "w") as f:
                json.dump(bear_res, f, indent=2)
            print(f"  BEAR done in {time.time()-t0:.0f}s -> {bear_out}")
        # GC
        gc.collect()
        torch.cuda.empty_cache()

        # TAXI
        taxi_out = RESULTS_ROOT / "taxi" / f"{m_slug}__taxi_fwd.json"
        if taxi_out.exists():
            print(f"  TAXI result already exists at {taxi_out}; loading.")
            with open(taxi_out) as f:
                taxi_res = json.load(f)
        else:
            t0 = time.time()
            taxi_res = run_taxi_probe(m, direction="fwd")
            taxi_out.parent.mkdir(parents=True, exist_ok=True)
            with open(taxi_out, "w") as f:
                json.dump(taxi_res, f, indent=2)
            print(f"  TAXI done in {time.time()-t0:.0f}s -> {taxi_out}")
        gc.collect()
        torch.cuda.empty_cache()

        sweep_summary[m] = {
            "n_params": bear_res["n_params"],
            "bear": {
                "accuracy": bear_res["grand_accuracy"],
                "bits": bear_res["total_acquired_bits"],
                "share_2bpp": bear_res["param_share_2bpp"],
                "share_3p6bpp": bear_res["param_share_3p6bpp"],
                "n_total": bear_res["grand_total"],
            },
            "taxi": {
                "accuracy": taxi_res["grand_accuracy"],
                "bits": taxi_res["total_acquired_bits"],
                "share_2bpp": taxi_res["param_share_2bpp"],
                "share_3p6bpp": taxi_res["param_share_3p6bpp"],
                "n_total": taxi_res["grand_total"],
            },
        }
        # Save running summary in case we get interrupted.
        with open(RESULTS_ROOT / "sweep_summary.json", "w") as f:
            json.dump(sweep_summary, f, indent=2)

    print(f"\nWrote {RESULTS_ROOT / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
