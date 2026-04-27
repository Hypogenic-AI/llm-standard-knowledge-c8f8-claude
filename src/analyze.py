"""
Analyse BEAR + TAXI probe outputs across model scales.

Generates:
  - figures/scaling_bits.png         (acquired bits vs. params, log-log)
  - figures/scaling_share.png        (parameter share vs. params)
  - figures/per_category.png         (TAXI per-category share)
  - figures/bear_vs_taxi_bits.png    (relational vs categorical bits)
  - results/summary_table.csv        (scaling table)
  - results/per_fact_footprint.json  (Knowledge-Neurons cross-check)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load_sweep():
    p = RES / "sweep_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run src/run_all.py first.")
    with open(p) as f:
        return json.load(f)


def load_full(model_slug: str):
    bear = json.load(open(RES / "bear" / f"{model_slug}__bear_small.json"))
    taxi = json.load(open(RES / "taxi" / f"{model_slug}__taxi_fwd.json"))
    return bear, taxi


def make_scaling_table(sweep):
    rows = []
    for model, d in sweep.items():
        rows.append({
            "model": model.split("/")[-1],
            "params": d["n_params"],
            "bear_acc": d["bear"]["accuracy"],
            "bear_bits": d["bear"]["bits"],
            "bear_share_2bpp_pct": 100 * d["bear"]["share_2bpp"],
            "bear_share_3p6bpp_pct": 100 * d["bear"]["share_3p6bpp"],
            "bear_n": d["bear"]["n_total"],
            "taxi_acc": d["taxi"]["accuracy"],
            "taxi_bits": d["taxi"]["bits"],
            "taxi_share_2bpp_pct": 100 * d["taxi"]["share_2bpp"],
            "taxi_share_3p6bpp_pct": 100 * d["taxi"]["share_3p6bpp"],
            "taxi_n": d["taxi"]["n_total"],
            "combined_bits": d["bear"]["bits"] + d["taxi"]["bits"],
            "combined_share_2bpp_pct": 100 * (d["bear"]["bits"] + d["taxi"]["bits"]) / (2.0 * d["n_params"]),
        })
    df = pd.DataFrame(rows).sort_values("params").reset_index(drop=True)
    df.to_csv(RES / "summary_table.csv", index=False)
    return df


def fig_scaling_bits(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(df["params"], df["bear_bits"], "o-", label="BEAR (relational)", color="tab:blue", lw=2, ms=8)
    ax.loglog(df["params"], df["taxi_bits"], "s-", label="TAXI (categorical)", color="tab:orange", lw=2, ms=8)
    ax.loglog(df["params"], df["combined_bits"], "^--", label="Combined", color="tab:green", lw=2, ms=8, alpha=0.8)
    # Capacity ceiling lines
    p_range = np.array([df["params"].min() * 0.7, df["params"].max() * 1.4])
    ax.loglog(p_range, 2.0 * p_range, ":", color="grey", label="2 bpp ceiling (Allen-Zhu)")
    ax.loglog(p_range, 3.6 * p_range, ":", color="darkgrey", label="3.6 bpp ceiling (Morris)", alpha=0.7)
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Acquired bits of standard knowledge")
    ax.set_title("Standard-knowledge bits vs. parameter count")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "scaling_bits.png", dpi=150)
    plt.close(fig)


def fig_scaling_share(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogx(df["params"], df["bear_share_2bpp_pct"], "o-", label="BEAR @ 2 bpp", color="tab:blue", lw=2, ms=8)
    ax.semilogx(df["params"], df["taxi_share_2bpp_pct"], "s-", label="TAXI @ 2 bpp", color="tab:orange", lw=2, ms=8)
    ax.semilogx(df["params"], df["combined_share_2bpp_pct"], "^--", label="Combined @ 2 bpp", color="tab:green", lw=2, ms=8)
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Parameter share (% of bit budget)")
    ax.set_title("Standard-knowledge parameter share vs. model size")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "scaling_share.png", dpi=150)
    plt.close(fig)


def fig_per_category(sweep):
    """Per-superordinate-category breakdown of TAXI bits & accuracy across models."""
    cats = ["animal", "plant", "vehicle", "instrument", "food", "drink"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.18

    models_sorted = sorted(sweep.keys(), key=lambda m: sweep[m]["n_params"])
    x = np.arange(len(cats))
    for i, m in enumerate(models_sorted):
        slug_m = m.replace("/", "__")
        try:
            taxi = json.load(open(RES / "taxi" / f"{slug_m}__taxi_fwd.json"))
        except FileNotFoundError:
            continue
        per_cat = taxi.get("per_category", {})
        n_params = taxi["n_params"]
        accs = []
        bits = []
        for c in cats:
            d = per_cat.get(c, {"correct": 0, "n": 1, "bits": 0.0})
            accs.append(d["correct"] / max(1, d["n"]))
            bits.append(d.get("bits", 0.0))
        axes[0].bar(x + i * width - 1.5 * width, accs, width, label=m.split("/")[-1])
        axes[1].bar(x + i * width - 1.5 * width, bits, width, label=m.split("/")[-1])

    axes[0].set_xticks(x); axes[0].set_xticklabels(cats, rotation=20)
    axes[0].set_ylabel("Accuracy"); axes[0].set_title("TAXI accuracy by category")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats, rotation=20)
    axes[1].set_ylabel("Acquired bits"); axes[1].set_title("TAXI acquired bits by category")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "per_category.png", dpi=150)
    plt.close(fig)


def fig_bear_vs_taxi(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(df))
    width = 0.4
    ax.bar(x - width/2, df["bear_bits"], width, label="BEAR (relational)", color="tab:blue")
    ax.bar(x + width/2, df["taxi_bits"], width, label="TAXI (categorical)", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("Acquired bits")
    ax.set_title("BEAR vs TAXI acquired bits by model")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "bear_vs_taxi_bits.png", dpi=150)
    plt.close(fig)


# Pythia hidden / intermediate dims for the per-fact footprint cross-check.
# (Source: Pythia config — d_model = hidden, d_inter = 4 * hidden.)
PYTHIA_DIMS = {
    "EleutherAI/pythia-160m": {"d_model": 768, "d_inter": 3072, "n_layers": 12},
    "EleutherAI/pythia-410m": {"d_model": 1024, "d_inter": 4096, "n_layers": 24},
    "EleutherAI/pythia-1b":   {"d_model": 2048, "d_inter": 8192, "n_layers": 16},
    "EleutherAI/pythia-2.8b": {"d_model": 2560, "d_inter": 10240, "n_layers": 32},
}


def per_fact_footprint(sweep):
    """Estimate per-fact parameter footprint two ways and aggregate to a
    total parameter share for the *known* facts.

    Approach A (Knowledge Neurons; Dai et al. 2021):
        ~4 FFN intermediate dims/fact, each contributing ~2*d_model parameters
        (one row of W_in plus one column of W_out) -> 8 * d_model / fact.

    Approach B (ROME; Meng et al. 2022):
        Full rank-1 update on one MLP-down layer: d_model + d_inter parameters
        (a left-vector + right-vector of the rank-1 update). This is generous
        because ROME's update conceptually changes one matrix slice; the
        functional support is rank-1 ≈ d_model + d_inter.
    """
    out = {}
    for model, d in sweep.items():
        if model not in PYTHIA_DIMS:
            continue
        dims = PYTHIA_DIMS[model]
        n_facts = int(d["bear"]["n_total"] * d["bear"]["accuracy"]) \
                + int(d["taxi"]["n_total"] * d["taxi"]["accuracy"])
        params_per_fact_kn = 4 * 2 * dims["d_model"]   # 8 * d_model
        params_per_fact_rome = dims["d_model"] + dims["d_inter"]
        kn_share = (n_facts * params_per_fact_kn) / d["n_params"]
        rome_share = (n_facts * params_per_fact_rome) / d["n_params"]
        out[model] = {
            "n_facts_known": n_facts,
            "params_per_fact_kn": params_per_fact_kn,
            "params_per_fact_rome": params_per_fact_rome,
            "kn_share": kn_share,
            "rome_share": rome_share,
        }
    with open(RES / "per_fact_footprint.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def scaling_regression(df):
    """Log-log regression of acquired bits vs. params; report slope ± CI."""
    out = {}
    for col, label in [("bear_bits", "BEAR"), ("taxi_bits", "TAXI"), ("combined_bits", "Combined")]:
        x = np.log10(df["params"].values)
        y = np.log10(np.maximum(df[col].values, 1e-9))
        if len(x) < 2:
            continue
        slope, intercept, r, p, se = stats.linregress(x, y)
        out[label] = {"slope": slope, "intercept": intercept, "r2": r ** 2, "p": p, "stderr": se}

    # Also regress *parameter share* on params (slope < 0 supports H2)
    x = np.log10(df["params"].values)
    y_share = df["combined_share_2bpp_pct"].values
    if len(x) >= 2:
        slope, intercept, r, p, se = stats.linregress(x, y_share)
        out["share_vs_params_linfit"] = {"slope_pct_per_decade": slope, "r2": r ** 2, "p": p}

    with open(RES / "scaling_regression.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    sweep = load_sweep()
    df = make_scaling_table(sweep)
    print(df.to_string(index=False))
    print()

    fig_scaling_bits(df)
    fig_scaling_share(df)
    fig_per_category(sweep)
    fig_bear_vs_taxi(df)

    pf = per_fact_footprint(sweep)
    print("Per-fact footprint (Knowledge-Neurons / ROME cross-check):")
    for m, v in pf.items():
        print(f"  {m}: {v['n_facts_known']} known facts;"
              f" KN-share {v['kn_share']*100:.4f}%, ROME-share {v['rome_share']*100:.4f}%")
    print()

    reg = scaling_regression(df)
    print("Scaling regression slopes (log10 bits ~ log10 params):")
    for k, v in reg.items():
        print(f"  {k}: {v}")

    print(f"\nFigures written to {FIG}")
    print(f"Tables written to {RES}")


if __name__ == "__main__":
    main()
