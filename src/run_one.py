"""Run BEAR + TAXI for a single model. Used for parallel orchestration."""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    m = args.model
    m_slug = m.replace("/", "__")

    bear_out = RESULTS_ROOT / "bear" / f"{m_slug}__bear_small.json"
    taxi_out = RESULTS_ROOT / "taxi" / f"{m_slug}__taxi_fwd.json"

    if not bear_out.exists():
        print(f"--- BEAR for {m} ---", flush=True)
        t0 = time.time()
        bear_res = run_bear_probe(m, subset="small")
        bear_out.parent.mkdir(parents=True, exist_ok=True)
        with open(bear_out, "w") as f:
            json.dump(bear_res, f, indent=2)
        print(f"BEAR done in {time.time()-t0:.0f}s -> {bear_out}", flush=True)
    else:
        print(f"BEAR cached -> {bear_out}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    if not taxi_out.exists():
        print(f"--- TAXI for {m} ---", flush=True)
        t0 = time.time()
        taxi_res = run_taxi_probe(m, direction="fwd")
        taxi_out.parent.mkdir(parents=True, exist_ok=True)
        with open(taxi_out, "w") as f:
            json.dump(taxi_res, f, indent=2)
        print(f"TAXI done in {time.time()-t0:.0f}s -> {taxi_out}", flush=True)
    else:
        print(f"TAXI cached -> {taxi_out}", flush=True)


if __name__ == "__main__":
    main()
