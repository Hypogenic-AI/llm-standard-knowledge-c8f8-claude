"""
Shared utilities for log-likelihood ranking probes (BEAR / TAXI style).

A probe scores each candidate completion (e.g. "Bengali") by computing the
length-normalised log-likelihood of the answer tokens given the prompt
("The native language of Ali Akbar Khan is "). The candidate with the highest
LL is the model's prediction.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ProbeResult:
    correct: bool
    pred_idx: int
    gold_idx: int
    scores: List[float]  # length-normalised LL for each candidate


def score_candidates(
    model,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
    device: str = "cuda",
    add_leading_space: bool = True,
    length_normalize: bool = True,
    batch_size: int = 16,
) -> List[float]:
    """Compute the (length-normalised) log-likelihood of each candidate
    appended after `prompt`.

    Returns a list of float LL scores (higher = more likely).
    """
    # Pre-tokenise prompt once. We compute LL of *only* the candidate tokens
    # (i.e. mask the prompt portion).
    if add_leading_space and not prompt.endswith(" "):
        prompt = prompt + " "

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    prompt_len = prompt_ids.shape[0]

    scores: List[float] = []

    # Build the full-sequence input ids per candidate
    encoded = []
    for cand in candidates:
        full_text = prompt + cand
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
        # answer ids = anything after the prompt
        # For BPE-style tokenisers, prompt+candidate may not start with the
        # exact prompt_ids prefix when the candidate boundary causes a
        # different tokenisation. We re-derive answer length by re-tokenising
        # both sides.
        answer_ids = full_ids[prompt_len:]
        # Edge case: if answer_ids is empty (prompt+candidate retokenised
        # so candidate merged into last prompt token), fall back to tokenising
        # candidate alone with a space prefix.
        if answer_ids.shape[0] == 0:
            answer_ids = tokenizer(" " + cand if add_leading_space else cand,
                                   return_tensors="pt", add_special_tokens=False).input_ids[0]
            full_ids = torch.cat([prompt_ids, answer_ids])
        encoded.append((full_ids, answer_ids.shape[0]))

    # Batch by sequences of similar length for efficiency
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Process in mini-batches
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        max_len = max(b[0].shape[0] for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        ans_lens = []
        for i, (full_ids, alen) in enumerate(batch):
            L = full_ids.shape[0]
            input_ids[i, :L] = full_ids
            attn_mask[i, :L] = 1
            ans_lens.append((L, alen))

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

        # logits[i, t] predicts token t+1, so answer log-probs are at positions
        # [L - alen - 1, ..., L - 2] predicting tokens [L - alen, ..., L - 1].
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        for i, (L, alen) in enumerate(ans_lens):
            if alen <= 0:
                scores.append(float("-inf"))
                continue
            # positions of predictors
            # we want log P(token_t | tokens_<t) for t in [L-alen, L-1]
            target_ids = input_ids[i, L - alen:L]
            pred_logits = log_probs[i, L - alen - 1:L - 1, :]
            tok_lls = pred_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            ll = tok_lls.sum().item()
            if length_normalize:
                ll = ll / alen
            scores.append(ll)

    return scores


def rank_predict(scores: Sequence[float]) -> int:
    """Return argmax index over scores."""
    return int(np.argmax(scores))


def acquired_bits(n_correct: int, n_total: int, answer_set_size: int) -> float:
    """Compute the *acquired* bits of information using a uniform-random
    baseline.

    bits = N_total * log2(|A|) * max(0, accuracy - 1/|A|)

    Rationale: a random predictor is expected to be right with prob 1/|A|,
    contributing 0 bits of information. The fraction *above* baseline is
    converted to bits at log2(|A|) per fact.
    """
    if n_total == 0 or answer_set_size <= 1:
        return 0.0
    accuracy = n_correct / n_total
    baseline = 1.0 / answer_set_size
    if accuracy <= baseline:
        return 0.0
    return n_total * math.log2(answer_set_size) * (accuracy - baseline)


def parameter_share(bits: float, n_params: int, bpp: float = 2.0) -> float:
    """bits / (bpp * n_params) — the fraction of the model's bit budget."""
    if n_params <= 0:
        return 0.0
    return bits / (bpp * n_params)


def bootstrap_ci(values, n_resamples: int = 1000, ci: float = 0.95, seed: int = 42):
    """Bootstrap CI over a binary (0/1) sample's mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, arr.size, arr.size)
        means.append(arr[idx].mean())
    means = np.sort(means)
    lo = means[int((1 - ci) / 2 * n_resamples)]
    hi = means[int((1 + ci) / 2 * n_resamples)]
    return float(arr.mean()), float(lo), float(hi)
