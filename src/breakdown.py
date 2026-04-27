"""
Compute per-relation BEAR breakdown and per-property TAXI breakdown into
'standard / common' vs 'long-tail' relations, plus bootstrap CIs on the
parameter share.

For BEAR we proxy "standard / common" knowledge by relations whose answer
space is a small set of culturally familiar entries (e.g. countries,
languages, sports). All BEAR relations are in fact standard relational
knowledge — the dataset was built that way — so this script reports per-
relation bits and a bootstrap CI on the *aggregate* share.

For TAXI everything is categorical/standard by construction; we report the
per-superordinate-category breakdown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"


def bootstrap_share(model_slug, n_resamples=1000, seed=42):
    """Bootstrap the BEAR + TAXI parameter share over resampled facts."""
    bear = json.load(open(RES / "bear" / f"{model_slug}__bear_small.json"))
    taxi = json.load(open(RES / "taxi" / f"{model_slug}__taxi_fwd.json"))
    n_params = bear["n_params"]

    # Build one entry per fact: (relation_size, correct?)
    bear_pool = []
    for rel_id, info in bear["per_relation"].items():
        a = info["answer_space_size"]
        n_correct = info["n_correct"]
        n_tot = info["n_total"]
        bear_pool.extend([(a, 1)] * n_correct + [(a, 0)] * (n_tot - n_correct))

    taxi_pool = []
    for prop, info in taxi["per_property"].items():
        a = max(2, int(round(info.get("avg_answer_set_size", 8))))
        c = info["correct"]
        n = info["n"]
        taxi_pool.extend([(a, 1)] * c + [(a, 0)] * (n - c))

    rng = np.random.default_rng(seed)

    def share_for_pool(pool, factor=1.0):
        if not pool:
            return 0.0
        # Acquired bits: per-relation we computed N * log2(|A|) * (acc - 1/|A|)
        # Easier here: each correct fact contributes log2(|A|) bits, each wrong
        # fact contributes 0; subtract baseline N/|A| * log2(|A|).
        # We accumulate sum_correct * log2(|A|) and sum_total * log2(|A|)/|A|.
        sum_c = 0.0
        sum_b = 0.0
        for a, c in pool:
            la = math.log2(a)
            sum_c += c * la
            sum_b += la / a
        return max(0.0, sum_c - sum_b)

    point = (share_for_pool(bear_pool) + share_for_pool(taxi_pool)) / (2.0 * n_params)

    # Bootstrap
    boots = []
    bear_arr = np.array(bear_pool, dtype=np.int64) if bear_pool else np.empty((0, 2), dtype=np.int64)
    taxi_arr = np.array(taxi_pool, dtype=np.int64) if taxi_pool else np.empty((0, 2), dtype=np.int64)
    for _ in range(n_resamples):
        if len(bear_arr):
            i = rng.integers(0, len(bear_arr), len(bear_arr))
            b_b = bear_arr[i]
            b_pool = list(map(tuple, b_b.tolist()))
        else:
            b_pool = []
        if len(taxi_arr):
            i = rng.integers(0, len(taxi_arr), len(taxi_arr))
            t_b = taxi_arr[i]
            t_pool = list(map(tuple, t_b.tolist()))
        else:
            t_pool = []
        s = (share_for_pool(b_pool) + share_for_pool(t_pool)) / (2.0 * n_params)
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * n_resamples)]
    hi = boots[int(0.975 * n_resamples)]
    return point, lo, hi


def bear_per_relation_table():
    sweep = json.load(open(RES / "sweep_summary.json"))
    rows = []
    for m in sorted(sweep.keys(), key=lambda k: sweep[k]["n_params"]):
        slug_m = m.replace("/", "__")
        try:
            bear = json.load(open(RES / "bear" / f"{slug_m}__bear_small.json"))
        except FileNotFoundError:
            continue
        for rel_id, info in bear["per_relation"].items():
            rows.append({
                "model": m.split("/")[-1],
                "relation": rel_id,
                "answer_space": info["answer_space_size"],
                "n_total": info["n_total"],
                "accuracy": info["accuracy"],
                "acquired_bits": info["acquired_bits"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "bear_per_relation.csv", index=False)
    return df


def main():
    sweep = json.load(open(RES / "sweep_summary.json"))
    print("Bootstrap CIs on combined parameter share:")
    boots = {}
    for m in sorted(sweep.keys(), key=lambda k: sweep[k]["n_params"]):
        slug_m = m.replace("/", "__")
        point, lo, hi = bootstrap_share(slug_m, n_resamples=400)
        boots[m] = {"point_pct": 100 * point, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi}
        print(f"  {m}: {100*point:.4f}% (95% CI {100*lo:.4f}–{100*hi:.4f})")
    with open(RES / "bootstrap_share.json", "w") as f:
        json.dump(boots, f, indent=2)

    df = bear_per_relation_table()
    print(f"\nWrote BEAR per-relation table with {len(df)} rows.")


if __name__ == "__main__":
    main()
