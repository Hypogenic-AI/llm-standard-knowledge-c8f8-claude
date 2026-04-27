# Research Plan — How much volume is *standard knowledge* in LLMs?

## Motivation & Novelty Assessment

### Why This Research Matters
Mechanistic interpretability and capacity-scaling work both touch the question
"how much of an LLM's parameters store *facts*?", but neither isolates the
specific category of knowledge most users care about: **standard categorical
and conditional knowledge** ("Peanuts are legumes", "Georgia is a US state and
a country"). Quantifying this fraction tells us (a) how much capacity is left
over for reasoning, style, and long-tail facts, and (b) at what model scale
diminishing returns set in for *common* knowledge — informing distillation,
quantisation, and edge-deployment decisions.

### Gap in Existing Work
- Allen-Zhu & Li (2024) measure ceilings of **2 bits per parameter (bpp)** on
  synthetic biographies; Morris et al. (2025) measure **3.6 bpp** on uniform
  random strings. Neither partitions stored bits by knowledge type.
- TAXI (Powell et al. 2024) probes *consistency* of categorical knowledge
  under editing but never converts probe accuracy into a parameter footprint.
- BEAR (Wiland et al. 2024) probes relational knowledge across LM types but
  again does not connect coverage to parameter share.
- ROME / Knowledge Neurons localise individual facts but stop at the *per-fact*
  level — no aggregate "X% of parameters are spent on standard knowledge."
- Hong et al. 2025 show specialisation rises with model generation, suggesting
  any answer is **architecture- and scale-dependent** and must be reported
  per-model.

### Our Novel Contribution
We *combine* probe-based knowledge accounting (BEAR + TAXI) with Allen-Zhu's
**bpp** ceiling to produce, for the first time:
1. A direct estimate of the **fraction of LLM parameters devoted to standard
   categorical and conditional knowledge**, reported across a family of model
   scales (Pythia 160M / 410M / 1B / 2.8B).
2. A breakdown by **knowledge type** (categorical vs. relational) and by
   **superordinate category** (animals, vehicles, foods, …).
3. A scaling analysis: does the parameter share *grow*, *stay flat*, or
   *shrink* as models get bigger?

### Experiment Justification
- **Experiment 1 (BEAR LL-rank probe across Pythia sizes):** measures the
  acquired bits of *relational* standard knowledge per model, comparable
  across scales. Justified because BEAR's log-likelihood ranking eliminates
  tokenizer artefacts that plague LAMA's cloze probe.
- **Experiment 2 (TAXI category probe):** isolates *taxonomic / categorical*
  knowledge (the headline form of "standard" knowledge). Justified because
  it's the only public benchmark with explicit category-membership and
  category-conditional property structure.
- **Experiment 3 (Bit-budget → parameter-share conversion):** converts
  acquired bits to fraction of model capacity using the Allen-Zhu 2 bpp
  ceiling and Morris 3.6 bpp upper bound. Justified because this is the
  step missing from prior work and is the deliverable of the project.
- **Experiment 4 (Per-fact footprint sanity check):** uses an analytical
  lower bound from Knowledge Neurons (≈4 FFN-intermediate dims/fact, each
  contributing 2·d_model parameters) to cross-check the bit-budget estimate.

---

## Research Question
**What fraction of an LLM's parameters is devoted to standard categorical and
conditional knowledge?** Specifically, how does this fraction scale with
model size, and how does it partition by knowledge type?

## Hypothesis Decomposition
- **H1 (volume):** Standard categorical + conditional knowledge consumes
  substantially less than 1% of an LLM's bit budget.
- **H2 (scaling):** As parameters increase from 160M → 2.8B, the *absolute*
  bits of standard knowledge stored grow, but the *fraction* of total
  capacity shrinks (dilution by reasoning / long-tail facts).
- **H3 (typology):** Categorical "is-a" knowledge is denser per fact than
  relational "located-in" knowledge, because category membership entails
  many properties in fewer bits.

## Proposed Methodology

### Approach
We do **probe-based accounting**, not model surgery. The pipeline:

1. For each model M ∈ {Pythia-160M, -410M, -1B, -2.8B}:
   - For each (subject, relation, answer) triple in BEAR, compute the
     pseudo-log-likelihood of every candidate answer given the relation
     template, take argmax, score accuracy.
   - For each TAXI baseline-evaluation row, score the forward and reverse
     queries via log-likelihood ranking over the choice set.
2. Convert per-relation accuracy A_r and answer-set size |A_r| to acquired
   bits: `bits_r = N_r · max(0, log2(|A_r|) · (A_r − 1/|A_r|))` —
   subtract a uniform-random baseline so we count only *acquired* bits.
3. Sum across relations / properties → total "standard-knowledge bits".
4. Divide by total parameter capacity (2 bpp · |params|) → parameter share.
5. Also report the share under Morris's 3.6 bpp upper bound.

### Experimental Steps
1. Set up environment (uv venv, install torch / transformers / lm-pub-quiz).
2. Verify pre-downloaded BEAR (small + big), TAXI evaluation files load.
3. Implement `src/probe_bear.py` — log-likelihood ranking probe for any HF
   causal LM, given a relation's templates and answer-space labels.
4. Implement `src/probe_taxi.py` — analogous probe over TAXI fwd_choices.
5. Implement `src/bit_budget.py` — convert per-relation/per-property accuracy
   into acquired bits and parameter share.
6. Run probes for Pythia-160M, -410M, -1B, -2.8B on the BEAR-small subset
   (7,731 facts × 60 relations) and full TAXI baseline (1,435 records).
7. Generate plots: bits-acquired vs. params (log-log), parameter-share vs.
   params, per-category breakdowns.

### Baselines
- **Random / majority** baseline per relation (acts as the floor we subtract).
- **Allen-Zhu 2 bpp ceiling** as the denominator → "fraction of bit budget".
- **Morris 3.6 bpp ceiling** as an alternative denominator (lower bound on
  parameter share).
- **Knowledge-Neurons per-fact footprint** (≈4 FFN intermediate × 2·d_model
  ≈ 8·d_model parameters / fact) as an independent estimate.

### Evaluation Metrics
- **Accuracy** per relation / property (probe correctness).
- **Acquired bits** = N · max(0, log2(|A|) · (acc − 1/|A|)).
- **Parameter share** = acquired_bits / (bpp · |params|), reported for
  bpp ∈ {2.0, 3.6}.
- **Bits per fact** = log2(|A|) · acc — informs "per-fact density".

### Statistical Analysis Plan
- **Bootstrap confidence intervals** (1,000 resamples) on per-model
  accuracy and acquired bits.
- **Scaling fit:** regress log(acquired_bits) on log(params), report slope
  and 95% CI; H2 predicts slope < 1 for fraction (negative log-log slope
  for share).
- **Per-category one-way ANOVA** on TAXI superordinate categories to test
  whether categorical density differs by domain (animals vs. drinks etc.).

## Expected Outcomes
- Standard categorical + relational knowledge is **0.05–0.5%** of parameter
  bit budget at Pythia-410M and roughly **flat or shrinking** at larger
  scales. (Back-of-envelope from literature review § 8.5: ~10⁵–10⁶ bits of
  standard knowledge vs. 10⁹ bits of capacity at 410M.)
- Categorical TAXI bits-per-fact ≈ 1–3 bits; BEAR bits-per-fact ≈ 3–5 bits
  (more answer-space entropy).

## Timeline and Milestones
| Phase | Time | Deliverable |
|-------|------|-------------|
| Plan + setup | 30 min | planning.md, env ready |
| Probe implementation | 60 min | src/probe_bear.py, src/probe_taxi.py |
| Run experiments | 90 min | results/probe_results.json per model |
| Analysis + plots | 45 min | figures/, results/summary.json |
| REPORT.md + README | 30 min | final docs |

## Potential Challenges
- **Tokenizer multi-token answer issue.** Mitigation: use length-normalised
  log-likelihood (mean log-prob over answer tokens), as recommended by
  Wiland et al. 2024.
- **Memory at Pythia-2.8B.** Mitigation: run in fp16; if still tight, drop
  to bs=1 with attention checkpointing or skip 2.8B.
- **Template variance.** Mitigation: average over BEAR's multiple templates
  per relation when available.
- **Allen-Zhu's 2 bpp may not transfer to undertrained models like Pythia.**
  We mitigate by *also* reporting raw bits acquired and the 3.6-bpp upper
  bound, so the result is robust to the choice of ceiling.

## Success Criteria
- All four Pythia sizes complete BEAR + TAXI probes without crashes.
- Acquired-bits and parameter-share numbers are computed and plotted.
- Bootstrap CIs are reported.
- A clear answer to "what fraction" emerges, even if order-of-magnitude.
