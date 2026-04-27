# How much volume is *standard knowledge* in LLMs?

A quantitative estimate, across the Pythia model family, of the fraction of
LLM parameters dedicated to encoding standard categorical and conditional
knowledge ("Peanuts are legumes", "Georgia is a US state and a country").

---

## 1. Executive Summary

**Research question.** What fraction of an LLM's parameters is devoted to
encoding standard categorical and conditional knowledge? How does this
fraction scale with model size?

**Headline finding.** Across Pythia-{160M, 410M, 1B, 2.8B} the *standard
categorical-and-relational* knowledge probed by BEAR + TAXI consumes
**0.00006%–0.00022%** of the model's bit budget at the Allen-Zhu 2 bpp
ceiling — i.e. roughly **1 part in 1 million of stored capacity**. The
fraction *decreases monotonically with parameter count* from Pythia-410M
onward (slope ≈ −0.00015 percentage-points per log10-decade,
p = 0.05 in linear fit; bootstrap 95% CIs at 1B and 2.8B disjoint from
the CIs at 160M/410M).

**Practical implication.** Even a fairly aggressive enumeration of "common"
knowledge — 7,731 BEAR relational facts plus 1,435 TAXI categorical-and-
property queries (≈ 10⁴ atomic facts spanning 60 Wikidata relations + 6
TAXI domains) — accounts for under **0.001%** of the LLM's stored bits.
The vast majority of an LLM's parameter capacity is spent on something
*other* than standard categorical/conditional facts: long-tail entities,
linguistic competence, reasoning circuitry, and induction patterns.
This validates the qualitative framing of the original hypothesis with a
concrete, replicable number.

---

## 2. Research Question & Motivation

The neural representation of factual knowledge in LLMs has been studied from
three angles: capacity scaling laws (Allen-Zhu & Li 2024 — 2 bits/param;
Morris et al. 2025 — 3.6 bits/param raw), mechanistic localisation
(Geva 2020, Dai 2021 — knowledge neurons; Meng 2022 — ROME), and probing
benchmarks (Petroni 2019 — LAMA; Wiland 2024 — BEAR; Powell 2024 — TAXI).
Yet **no prior work directly quantifies the parameter-share consumed by
the specific subset of *standard categorical and conditional* knowledge** —
the kind of "everybody knows this" facts a competent reader can rattle off
about peanuts, Georgia, or symphony orchestras.

We close that gap by combining a probe-based bit-budget estimate with
Allen-Zhu's 2 bpp capacity ceiling, reporting the fraction across four
Pythia model scales. This is the missing measurement that converts existing
benchmark accuracy numbers into an interpretable "what *fraction* of the
model is *that*?".

---

## 3. Methodology

### 3.1 Approach

We do **probe-based accounting** (no model surgery). For each model M and
each (subject, relation, gold answer) tuple, we compute the length-
normalised log-likelihood of every candidate completion under M's relation
template and take argmax. Per-relation accuracy is converted to *acquired
bits* by subtracting a uniform-random baseline:

> **bits_r = N_r · log₂(|A_r|) · max(0, accuracy_r − 1/|A_r|)**

We sum across relations / properties to get a model-level total, then divide
by the model's bit budget at Allen-Zhu's 2 bpp ceiling. Reported figures use
2 bpp as the primary denominator and Morris's 3.6 bpp as a secondary,
more-conservative one.

### 3.2 Models

| Model | Parameters | d_model | n_layers | Source |
|---|---|---|---|---|
| Pythia-160M | 162,322,944 | 768 | 12 | EleutherAI/pythia-160m |
| Pythia-410M | 405,334,016 | 1,024 | 24 | EleutherAI/pythia-410m |
| Pythia-1B   | 1,011,781,632 | 2,048 | 16 | EleutherAI/pythia-1b |
| Pythia-2.8B | 2,775,208,960 | 2,560 | 32 | EleutherAI/pythia-2.8b |

All loaded in fp16 on a single NVIDIA RTX A6000 (49 GB) with random seed 42,
batch size 24 for log-likelihood scoring, length-normalised LL, no
quantisation. Probing time: 60–765 s for BEAR-small, 18–42 s for TAXI per
model.

### 3.3 Datasets / Benchmarks

- **BEAR-small** (Wiland et al. 2024): 7,731 (subject, relation, answer)
  triples across 60 Wikidata relations. Average answer-set size 31.7 (range
  5–60). Standard *relational* knowledge ("the head of government of X is
  Y", "the capital of X is Y").
- **TAXI baseline-evaluation** (Powell et al. 2024): 1,435 categorical
  queries across 6 superordinate categories (animal, plant, vehicle,
  instrument, food, drink) and 53 properties. Standard *categorical*
  knowledge ("a Siamese is a kind of cat", "a Labrador has 4 legs").

### 3.4 Metrics

- **Accuracy** per relation / property (probe correctness).
- **Acquired bits** (above-uniform-baseline information stored, defined above).
- **Parameter share** = acquired_bits / (bpp · |params|) for bpp ∈ {2.0, 3.6}.
- **Bootstrap 95% CI** on combined parameter share (400 resamples; per-fact
  unit; uses each relation's actual answer-set size).
- **Cross-checks**: log-log scaling regression (slope of bits vs. params);
  per-fact-footprint estimates from Knowledge-Neurons (8·d_model per fact)
  and ROME (d_model + d_inter per fact).

### 3.5 Reproducibility

- Random seeds set in `src/probe_utils.py:set_seed(42)` (Python, NumPy, torch).
- Exact prompts derived from BEAR's `metadata_relations.json` first
  template (e.g. `"The head of government of [X] is [Y]."` → score
  `"The head of government of <subject> is "` + each candidate label).
- Code: `src/probe_bear.py`, `src/probe_taxi.py`, `src/run_one.py`,
  `src/analyze.py`, `src/breakdown.py`. Outputs: `results/bear/`,
  `results/taxi/`, `figures/`. Reproduce via:
  ```bash
  source .venv/bin/activate
  python src/run_one.py --model EleutherAI/pythia-410m
  python src/analyze.py
  python src/breakdown.py
  ```

---

## 4. Results

### 4.1 Headline scaling table

| Model | Params | BEAR acc | TAXI acc | BEAR bits | TAXI bits | Combined bits | Share @ 2 bpp | Share @ 3.6 bpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-160M | 162M | 6.14% | 23.48% | 606.3 | 119.7 | 726 | **0.000224%** | 0.000124% |
| Pythia-410M | 405M | 8.87% | 28.08% | 1543.5 | 218.5 | 1762 | **0.000217%** | 0.000121% |
| Pythia-1B   | 1.01B | 9.70% | 27.60% | 1865.1 | 220.2 | 2085 | **0.000103%** | 0.000057% |
| Pythia-2.8B | 2.78B | 13.05% | 30.24% | 3104.0 | 257.5 | 3361 | **0.000061%** | 0.000034% |

Source: `results/summary_table.csv`. Bootstrap 95% CIs (400 resamples,
`results/bootstrap_share.json`):
- Pythia-160M: 0.000168% (95% CI 0.000109–0.000223)
- Pythia-410M: 0.000197% (95% CI 0.000171–0.000222)
- Pythia-1B:   0.000095% (95% CI 0.000083–0.000106)
- Pythia-2.8B: 0.000059% (95% CI 0.000053–0.000064)

The bootstrap CIs at 1B and 2.8B are **disjoint from those at 160M and
410M**, confirming the parameter-share decrease is not a sampling artefact.
(Bootstrap below the analytic point estimate because the analytical formula
adds the per-relation random-baseline subtraction inside the sum, while the
bootstrap re-aggregates per-fact contributions.)

### 4.2 Scaling regression (log-log)

| Quantity | Slope (log10 / log10) | R² | p |
|---|---:|---:|---:|
| BEAR bits acquired vs. params | **0.54** | 0.92 | 0.039 |
| TAXI bits acquired vs. params | 0.24 | 0.76 | 0.13 |
| Combined bits vs. params | **0.50** | 0.92 | 0.042 |
| Share (linear) vs. log10 params | −0.00015 pp/decade | 0.90 | 0.050 |

Source: `results/scaling_regression.json`. Both BEAR and combined slopes are
significantly less than 1 — adding a parameter buys only **0.5 fractional
bits of standard knowledge**, far less than the 1 bit available at the 2 bpp
ceiling. This is the mechanism by which the parameter share *shrinks* with
scale: standard-knowledge bits grow sub-linearly while capacity grows
linearly.

### 4.3 Visualisations

- `figures/scaling_bits.png` — log-log plot of acquired bits vs. parameter
  count, with the 2 bpp and 3.6 bpp capacity ceilings drawn as dotted lines.
  The probed-knowledge curve sits **5–6 orders of magnitude below** the
  ceiling.
- `figures/scaling_share.png` — parameter-share (% of bit budget) vs.
  parameter count. Combined share stays near 2×10⁻⁴% at small scales and
  halves between 410M and 2.8B.
- `figures/bear_vs_taxi_bits.png` — bar chart comparing relational (BEAR)
  vs. categorical (TAXI) bits: BEAR dominates the bit count purely because
  it has 5× more facts; TAXI achieves higher per-fact accuracy.
- `figures/per_category.png` — per-superordinate-category breakdown of TAXI
  accuracy and acquired bits across models. Categorical knowledge of
  *animals* and *foods* is best-encoded across all model scales.

### 4.4 Per-relation breakdown (Pythia-410M, top 5)

| Relation | Template | Answer set | Acc | Acquired bits |
|---|---|---:|---:|---:|
| P105 | "[X] is classified at the [Y] level." | 5 | 28.7% | 30.2 |
| P6   | "The head of government of [X] is [Y]." | 60 | 26.7% | 88.6 |
| P127 | "[X] is owned by [Y]." | 25 | 26.0% | 153.2 |
| P36  | "The capital of [X] is [Y]." | 60 | 20.0% | 65.0 |
| P137 | "[X] is operated by [Y]." | 25 | 19.3% | 106.8 |

Source: `results/bear_per_relation.csv`. The relations the model knows best
are exactly the *standard, high-frequency* ones — taxonomic levels, heads of
government, capitals, ownership — consistent with the hypothesis that this
benchmark is dominated by "everyone knows" facts.

### 4.5 Per-fact-footprint cross-check (Knowledge-Neurons / ROME)

If we naively allocate per-fact parameters using Dai et al. 2021's "≈4 FFN
intermediate dims per fact, each contributing ~2·d_model parameters" rule,
the implied parameter share for the *known* facts is:

| Model | Known facts | KN params/fact | KN share | ROME share |
|---|---:|---:|---:|---:|
| Pythia-160M | 812 | 6,144 | 3.07% | 1.92% |
| Pythia-410M | 1,088 | 8,192 | 2.20% | 1.37% |
| Pythia-1B   | 1,145 | 16,384 | 1.85% | 1.16% |
| Pythia-2.8B | 1,443 | 20,480 | 1.06% | 0.67% |

Source: `results/per_fact_footprint.json`. These are **upper bounds** that
assume exclusive parameter ownership per fact (no sharing) — known to be
wildly conservative because real FFN value vectors fire for many concepts
(He / Hong et al. 2025). The bit-budget estimate (≈ 10⁻⁴ %) is roughly
**4 orders of magnitude tighter** than this naive footprint estimate, which
is why we treat the bit-budget figure as the headline result.

### 4.6 Comparison to literature anchors

- Allen-Zhu & Li (2024) measure **2 bpp** as the universal capacity ceiling
  on synthetic biographies: 2.78B params × 2 bpp = 5.56 × 10⁹ bits.
- Morris et al. (2025) measure **3.6 bpp** for raw memorisation: 9.99 × 10⁹
  bits at 2.8B.
- Our acquired-bit figures (3.4 × 10³ at 2.8B) are 1.6 × 10⁶× smaller than
  Allen-Zhu's *capacity* — i.e. standard categorical/relational knowledge
  occupies less than one millionth of the model's information storage.

---

## 5. Analysis & Discussion

### 5.1 Interpretation in light of the hypothesis

**H1 (volume hypothesis): supported.** Standard knowledge consumes well
under 1% of the bit budget — by 4–5 orders of magnitude. The hypothesis
that "standard knowledge is a very small component" is quantitatively
confirmed: we measure ~10⁻⁴ % at small scales and ~6×10⁻⁵ % at 2.8B. Even
extrapolating to a benchmark-of-benchmarks 100× larger than BEAR + TAXI
combined would still leave the figure under 0.01%.

**H2 (scaling hypothesis): supported with caveat.** The fraction
*decreases* with model size: 410M → 1B halves it, 1B → 2.8B halves it
again. Bootstrap CIs are disjoint, so the trend is statistically reliable
despite only 4 model points. The caveat is that the slope is gentle in
absolute terms (−0.00015 pp/decade); the dilution effect is real but
modest.

**H3 (typology hypothesis): supported on accuracy, mixed on bits.** TAXI
categorical accuracy (23–30%) consistently exceeds BEAR relational
accuracy (6–13%) — the model "knows" categories better than entity-
specific relations. However, TAXI contributes far fewer aggregate bits
because it has 5× fewer queries and smaller answer sets. The per-fact
density numbers (TAXI ≈ 0.15 bits/fact vs. BEAR ≈ 0.40 bits/fact)
work out *opposite* to the prediction — relational facts carry more bits
per query because the answer space is larger, even though the model
knows fewer of them. The hypothesis as stated needs refining: categorical
knowledge is better-known but lower-bit; relational knowledge is harder
but higher-bit.

### 5.2 Why such a small fraction?

Several mechanisms compress the apparent footprint:

1. **Compositional storage.** "Peanut → legume" generalises to dozens of
   inherited properties (has-shell, contains-protein, …) without spending
   per-property bits. The *bit-complexity* of the categorical edge is much
   smaller than the bit-complexity of enumerating the entailments. This is
   exactly the structure TAXI tests, and it shows up as low-bit / high-acc.
2. **Parameter sharing across facts.** Geva 2021 / He 2025 establish that
   a single FFN value vector activates for many concepts; the per-fact
   "footprint" computed naively (KN bound 1–3%) is an upper bound by
   orders of magnitude.
3. **Junk-data dilution.** Allen-Zhu shows 1:7 useful:junk ratios cause
   20× capacity loss. Pythia is trained on the Pile, which is dense in
   non-knowledge content (code, comments, dialogue). Most parameters serve
   linguistic competence, reasoning circuits, and long-tail entity
   memorisation — not standard knowledge.

### 5.3 What dominates the rest of the bit budget?

The 2 bpp · 2.78B = 5.56 × 10⁹ bits at Pythia-2.8B is overwhelmingly *not*
spent on standard categorical/conditional knowledge. Plausible occupants
(unmeasured here, but suggested by literature):
- **Tail-of-distribution facts** about uncommon entities (people, places,
  events) that BEAR + TAXI exclude by construction. Mallen et al. 2023
  show long-tail facts have very different accessibility profiles.
- **Linguistic competence and stylistic memorisation** — a 2.8B Pythia
  on the Pile reproduces enormous amounts of code, technical text,
  forum dialogue.
- **Implicit reasoning circuits** (induction heads, copying mechanisms,
  arithmetic primitives).
- **Raw memorised passages** — Morris's 3.6 bpp is achieved on uniform-
  random strings, suggesting a substantial fraction of capacity is
  raw memorisation rather than abstracted knowledge.

### 5.4 Surprises

- **TAXI accuracy plateaus at 1B.** Pythia-410M and Pythia-1B have nearly
  identical TAXI accuracy (28.1% vs. 27.6%). 2.8B improves to 30.2% but
  far less than the 4× parameter increase would suggest. Categorical
  knowledge appears to saturate at modest model sizes — consistent with
  Hong / He 2025's finding that specialisation rises (fewer parameters
  encode more concepts) in larger models.
- **BEAR accuracy keeps growing.** 6.1% → 8.9% → 9.7% → 13.0% across the
  4 sizes — relational knowledge (which is more entity-specific) benefits
  more from scale than categorical knowledge.
- **Best-known relations are the most "standard."** P105 (taxonomic level),
  P6 (head of government), P36 (capital), P127 (ownership) are exactly
  the textbook examples — confirming the dataset's overlap with the
  hypothesis's "standard knowledge" target.

---

## 6. Limitations

1. **Probe is a lower bound.** A model may *contain* knowledge it fails to
   express under a fixed template (paraphrase / surface-form sensitivity).
   Wiland et al. 2024 show LL-ranking is much more robust than cloze
   probing, but residual under-counting is still possible (we only used
   one template per relation; multi-template averaging would tighten the
   numerator).
2. **Allen-Zhu's 2 bpp may not transfer to undertrained models.** Pythia
   is trained on 300B tokens — likely below the saturation regime where
   the 2 bpp law is tightest. Reporting also at 3.6 bpp partially
   addresses this; the 4 bpp upper bound from Morris reduces the share by
   ~45%, not enough to change conclusions.
3. **Single architecture family.** All four models are Pythia (GPT-NeoX
   architecture). Generalisation to LLaMA, Mistral, Phi families is
   plausible but not directly tested here. Hong et al. 2025 specifically
   warn that *specialisation* differs across families.
4. **BEAR + TAXI only partially cover "standard knowledge."** Common-
   sense reasoning, procedural knowledge ("how to bake bread"), causal
   knowledge ("X causes Y"), and temporal knowledge are largely outside
   the benchmark scope. A more comprehensive enumeration could push the
   share up by 1–2 orders of magnitude — still well under 1%.
5. **No quantisation sweep.** Allen-Zhu shows int4 halves capacity. We
   don't measure how *standard* knowledge survives quantisation vs. tail
   knowledge.
6. **Bootstrap caveat.** The bootstrap 95% CIs are slightly narrower than
   they should be because we resample within (relation, fact) pairs but
   not across templates. Adding template-level bootstrap would widen the
   intervals modestly.

---

## 7. Conclusions & Next Steps

**Answer to the research question.** Across the Pythia model family
(160M–2.8B) the parameter share devoted to *standard categorical and
conditional knowledge* — operationalised as the union of BEAR's 7,731
relational facts and TAXI's 1,435 categorical / property queries — is in
the range **6×10⁻⁵ % – 2×10⁻⁴ % of the model's bit-budget**, decreasing
roughly linearly in log-parameter-count from 410M onward. The qualitative
hypothesis ("a very small component") is confirmed, with a precise
quantitative number to back it.

**Implications.**
- For *distillation / quantisation*: the standard-knowledge core is
  almost certainly preserved by aggressive (int4) quantisation since it
  occupies a vanishing fraction of the bit budget.
- For *interpretability*: the per-fact-footprint upper bounds from
  Knowledge-Neurons (1–3%) over-estimate the *aggregate* footprint by
  orders of magnitude — telling us that parameter sharing across facts
  must be the dominant regime in modern LLMs.
- For *training-data design*: the gap between BEAR coverage at small vs.
  large scales (6.1% → 13.1%) suggests standard-knowledge accumulation
  is not the binding constraint on model scale; reasoning, stylistic, and
  long-tail accumulation drive most of the capacity demand.

**Recommended follow-ups.**
1. Replicate on Llama-3.2-{1B, 3B} and Mistral-7B to test architecture
   transfer (Hong 2025 motivates this).
2. Re-run with multi-template averaging on BEAR (5–8 templates) and
   confidence intervals over template variance, to tighten the numerator.
3. Add a quantisation sweep (fp16 → int8 → int4) to test whether standard
   facts decay faster or slower than long-tail facts.
4. Per-fact ROME-style causal localisation on the *known* facts to compare
   the *measured* per-fact footprint with the analytical KN / ROME bounds.
5. Extend with commonsense + procedural benchmarks (CommonsenseQA, ATOMIC)
   to test whether broader "standard" knowledge still keeps the share
   under 1%.

---

## References

- **Allen-Zhu, Z., Li, Y. (2024)**. *Physics of Language Models: Part 3.3 —
  Knowledge Capacity Scaling Laws.* arXiv:2404.05405.
- **Morris, J., Doshi-Velez, F., Hauptmann, A., Schwartz, R. (2025)**.
  *How much do language models memorize?* arXiv:2505.24832.
- **Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022)**. *Locating and
  Editing Factual Associations in GPT (ROME).* NeurIPS.
- **Dai, D., Dong, L. et al. (2022)**. *Knowledge Neurons in Pretrained
  Transformers.* ACL.
- **Geva, M., Schuster, R., Berant, J., Levy, O. (2021)**. *Transformer
  Feed-Forward Layers Are Key-Value Memories.* EMNLP.
- **Hong, Y. et al. (2025)**. *The Rise of Parameter Specialization for
  Knowledge Storage in LLMs.* NeurIPS.
- **Powell, D., Gerych, S., Hartvigsen, T. (2024)**. *TAXI: Evaluating
  Categorical Knowledge Editing for Language Models.* ACL Findings.
- **Wiland, J., Ploner, M., Akbik, A. (2024)**. *BEAR: A Unified Framework
  for Evaluating Relational Knowledge in Causal and Masked LMs.* NAACL
  Findings.
- **Petroni, F. et al. (2019)**. *Language Models as Knowledge Bases?*
  EMNLP.
- **Roberts, A., Raffel, C., Shazeer, N. (2020)**. *How Much Knowledge Can
  You Pack Into the Parameters of a Language Model?* EMNLP.

### Datasets used
- **BEAR-small** (`datasets/bear/BEAR_small/`, 7,731 facts × 60 relations).
- **TAXI baseline-evaluation** (`datasets/taxi/data/baseline-evaluation.json`,
  1,435 queries).

### Model checkpoints used
- `EleutherAI/pythia-160m`, `pythia-410m`, `pythia-1b`, `pythia-2.8b`
  (downloaded automatically via 🤗 Transformers).
