# Literature Review

**Project**: How much volume is *standard knowledge* in LLMs?
**Hypothesis**: The amount of LLM parameters dedicated to encoding standard categorical and conditional knowledge ("Peanuts are legumes", "Georgia is both a US state and a country") is a very small component of the total model — but this has not yet been quantitatively estimated.

---

## 1. Research-area overview

The question "how much knowledge is stored in the weights of a language model, and where?" sits at the intersection of three rapidly converging literatures:

1. **Capacity scaling laws** — information-theoretic measurements of how many bits a transformer can encode, parameterised by size, training duration, quantization, and architecture (Allen-Zhu & Li 2024; Morris et al. 2025; Roberts, Raffel, Shazeer 2020).
2. **Mechanistic interpretability of factual knowledge** — locating *where* specific facts live (Geva et al. 2020 — FFNs as KV memories; Dai et al. 2021 — knowledge neurons; Meng et al. 2022 — ROME causal tracing; Hong et al. 2025 — parameter specialization).
3. **Knowledge probing** — *what fraction* of relational/commonsense knowledge a model actually contains (Petroni et al. 2019 — LAMA; Wiland et al. 2024 — BEAR; Powell et al. 2024 — TAXI for categorical; Cohen et al. 2024 — RippleEdits for entailments).

Despite the breadth of work, no paper directly answers our quantitative question for the specific subset of *standard* (commonsense / categorical) knowledge. The closest precedents (Allen-Zhu and Morris) measure raw bit-capacity without partitioning by knowledge type; TAXI evaluates *whether* categorical knowledge is consistent under editing but does not measure parameter footprint.

This review distils the load-bearing findings and crystallizes a methodology gap that our experiment can fill.

---

## 2. Key papers

### 2.1 Allen-Zhu & Li, *Physics of Language Models: Part 3.3 — Knowledge Capacity Scaling Laws* (arXiv:2404.05405, 2024)

- **Key contribution**: Quantitative scaling law — sufficiently-trained transformers store **2 bits of knowledge per parameter** (universal across GPT-2/LLaMA/Mistral/MoE; preserved at int8; halved at int4 → 0.7 bpp).
- **Methodology**: Synthetic biographies with known information content. Define `bioD(N,K,C,D,L,T)` as a controllable family of `(name, attribute, value)` tuples; derive a tight bit-complexity lower bound (Theorem 3.2); train hundreds of GPT-2 models from 1M to 0.5B parameters and read off the capacity ratio
  R(F) = (learned bits estimated from cross-entropy losses) / (parameter count).
- **Datasets**: bioS(N), bioSsimple(N), bioR(N) — author-controlled synthetic biographies; bioD(N,K,C,D,L,T) — fully parameterised tuple data.
- **Notable secondary results**:
  - Rare knowledge (100 exposures) → 1 bpp (50% capacity loss vs 1000 exposures).
  - GatedMLP (LLaMA/Mistral) hurts when training is short (1.3× capacity loss at 100 exposures).
  - **Junk data is catastrophic** — 1:7 useful:junk → 20× capacity loss; *prepending domain tokens recovers most of it*.
- **Code**: not publicly released; bioD is reimplementable from the paper.
- **Why central**: Defines the only published *parameter-density* number we can directly use to convert from bits-of-stored-knowledge to fraction-of-parameters.

### 2.2 Morris et al., *How much do language models memorize?* (arXiv:2505.24832, 2025)

- **Key contribution**: A model-capacity definition based on Kolmogorov-style compression that separates **unintended memorization** (information about a specific dataset) from **generalization** (information about the data-generating process). Empirically, GPT-style transformers in fp16 hold **3.5–4 bits per parameter** (point estimate α ≈ 3.64).
- **Methodology**: Train hundreds of GPT models (500K → 1.5B) on uniform random bitstrings to drive generalization to zero, then measure unintended memorization plateau. Predicts the dataset-size at which double descent kicks in — exactly when dataset bits exceed model capacity bits.
- **Tension with Allen-Zhu**: 3.6 vs 2 bpp. The two are not contradictory: Morris measures *raw memorizable bits* (uniform random strings); Allen-Zhu measures *retrievable knowledge tuples* (a structured subset of those bits with non-trivial extraction). The structured-knowledge ceiling is lower because the data has prior, the loss is summed over knowledge tokens only, and information must remain functionally accessible.
- **Why relevant**: Sets the *upper-bound* parameter budget within which all knowledge — standard or otherwise — must fit.

### 2.3 Meng, Bau, Andonian, Belinkov, *Locating and Editing Factual Associations in GPT* (NeurIPS 2022 — ROME)

- **Key contribution**: Two complementary findings:
  1. **Causal tracing**: middle MLP layers at the *last subject token* causally mediate factual recall (the "early site"). Attention is more important at the late site (the last prompt token).
  2. **Rank-One Model Editing**: factual associations can be inserted/replaced by a closed-form rank-one update to a single MLP's down-projection — preserving both generalization (paraphrases) and specificity (other facts).
- **Methodology**: Causal-mediation analysis (clean / corrupted-subject / patched runs); least-squares update with cached uncentered covariance C = K Kᵀ.
- **Datasets**: CounterFact (21,919 hand-curated counterfactual rewrites), zsRE, known_1000.
- **Why central**: ROME's rank-one update gives us a per-fact parameter footprint estimate (a single scalar Λ × C⁻¹k* contributes ~hidden_dim × intermediate_dim parameters effective).

### 2.4 Dai, Dong, Hao, Sui, Chang, Wei, *Knowledge Neurons in Pretrained Transformers* (ACL 2022)

- **Key contribution**: Integrated-gradient attribution method that identifies ~4 FFN intermediate neurons per fact in BERT-base on PARAREL/T-REx (27,738 facts × 8.6 templates each).
- **Findings**: Knowledge neurons concentrate in upper layers; suppressing them reduces fact accuracy; amplifying them increases it; surgical edits update facts without fine-tuning.
- **Why relevant**: Lower-bound on neurons-per-fact; a sparser estimate than ROME's MLP-layer view.

### 2.5 Geva, Schuster, Berant, Levy, *Transformer Feed-Forward Layers Are Key-Value Memories* (EMNLP 2021)

- **Key contribution**: The two FFN matrices in each transformer block act as keys (input pattern detectors) and values (output distributions). Lower layers detect shallow patterns; upper layers learn semantic patterns. Followed up by Geva et al. 2022 ("Promoting tokens") — the values can be projected back to vocabulary space to interpret what each neuron writes.
- **Why central**: The mechanistic scaffolding underlying every subsequent paper in this thread (ROME, Knowledge Neurons, He 2025).

### 2.6 Hong, Zhao, Tang, Deng, Rong, Zhang, *The Rise of Parameter Specialization for Knowledge Storage in LLMs* (NeurIPS 2025)

- **Key contribution**: Across 20 open-source LLMs, **stronger / more recent models exhibit greater parameter specialization** — fewer MLP value vectors per concept, each governing a narrower set of concepts. Defines a Parameter Specialization Score (PSS) using concept-specific masking. Causal training shows specialization → better knowledge utilization.
- **Implication for our project**: The *fraction* of parameters dedicated to a single fact has been *decreasing* with model generations. Any volume estimate we produce should be reported per-architecture, not as a universal constant.

### 2.7 Powell, Gerych, Hartvigsen, *TAXI: Evaluating Categorical Knowledge Editing for Language Models* (ACL Findings 2024)

- **Key contribution**: First benchmark to evaluate *categorical consistency* of knowledge edits. 41 categories × 164 subjects × 183 properties → 976 edits, 11,120 MCQ queries. Splits property success into *Consistency* (changed properties under new category) vs *Invariance* (unchanged).
- **Findings**: ROME 0.43 / ICE 0.47 / FT 0.23 consistency vs **human 0.86**. Editors achieve high invariance but low consistency — they edit a single fact but fail to propagate categorical entailments.
- **Why central**: Most directly aligned with our hypothesis. "Peanuts are legumes" entails dozens of properties (has-shell, contains-protein, edible, ...). TAXI's structure formalises *standard categorical knowledge* as exactly the property-graph our hypothesis discusses.

### 2.8 Petroni et al., *Language Models as Knowledge Bases?* — LAMA (EMNLP 2019)

- **Key contribution**: First demonstration that pretrained masked LMs answer cloze-style relational queries competitively with traditional KBs. T-REx subset: 41 relations × ≤1000 facts. Limitations (single-token answers, masked-only, unbalanced answer distribution) motivated BEAR.

### 2.9 Wiland, Ploner, Akbik, *BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked LMs* (NAACL Findings 2024)

- **Key contribution**: Probes LMs by ranking the log-likelihood of full statements rather than predicting a single masked token. Works for both causal and masked LMs without restricting answer-space. Releases BEAR (7,731 instances, 60 relations) and BEAR-big (40,916 instances).
- **Why central**: The cleanest tool to measure *what fraction of relational knowledge* a modern LLM contains, comparable across architectures.

### 2.10 Roberts, Raffel, Shazeer, *How Much Knowledge Can You Pack Into the Parameters of a Language Model?* (EMNLP 2020)

- **Key contribution**: T5 closed-book QA on Natural Questions / WebQuestions / TriviaQA scales monotonically with parameter count (220M → 11B). Foundational counterpart to Allen-Zhu in real (uncontrolled) data.

### 2.11 Examining Two-Hop Reasoning Through Information Content Scaling (arXiv:2502.03490, 2025)

- Extends bit-complexity / scaling-law framework to multi-hop reasoning — relevant if we want to factor compositional standard knowledge ("Georgia is in North America" entails "Georgia is in the Western Hemisphere").

---

## 3. Common methodologies

| Method                     | Used in                                          | Output                                    |
|----------------------------|--------------------------------------------------|-------------------------------------------|
| Synthetic-data scaling law | Allen-Zhu (Physics 3.3), Morris 2025             | Bits-per-parameter ceiling                 |
| Cloze probing              | LAMA (single-token), KAMEL (generation)          | "Did the model know fact f?" (binary)     |
| Multiple-choice probing    | BEAR, TAXI                                       | Accuracy across answer set                 |
| Closed-book QA             | Roberts et al.                                   | Coverage on a held-out QA benchmark        |
| Knowledge neurons (IG)     | Dai et al., follow-ups                           | Per-fact neuron set in FFN intermediate    |
| Causal tracing             | ROME, follow-ups                                 | Per-fact MLP-layer + token-position site   |
| Parameter specialization   | He / Hong 2025                                   | PSS per concept, per model                 |
| Editing (ROME / MEMIT / FT)| Most editing papers                              | Edit success / generalization / specificity|

## 4. Standard baselines

- **For relational knowledge volume**: BEAR (modern, general) and LAMA T-REx (legacy reference).
- **For categorical knowledge**: TAXI consistency / invariance.
- **For raw capacity**: Allen-Zhu's 2 bpp and Morris's 3.6 bpp anchors.
- **For per-fact footprint**: ROME rank-one update (≈ d_model × d_inter / fact in *one* layer); Knowledge Neurons ≈ 2–5 FFN-intermediate dims per fact.

## 5. Evaluation metrics

- **Accuracy on probe** (BEAR, LAMA, TAXI, CounterFact-test) — what fraction of facts the model knows.
- **Bits of knowledge stored** (Allen-Zhu) — sum-of-loss formulation; Theorem 3.2 lower bound.
- **Capacity ratio R(F)** = stored_bits / parameter_count.
- **Membership-inference scaling law** (Morris) — recovery rate as a function of (capacity, dataset size).
- **Editing metrics**: Efficacy (post-edit success), Generalization (paraphrases), Specificity (neighbors), Consistency (TAXI), Invariance (TAXI).

## 6. Datasets in the literature (relevant to our hypothesis)

| Dataset           | Size                              | Knowledge type                              | Source                            |
|-------------------|-----------------------------------|----------------------------------------------|-----------------------------------|
| LAMA T-REx        | 41 relations × ≤1000 facts        | Relational (Wikidata)                        | Petroni 2019                       |
| PARAREL           | 27.7K facts × 8.6 templates       | Relational paraphrased                       | Elazar 2021                        |
| BEAR / BEAR-big   | 7,731 / 40,916 instances · 60 rels| Relational, balanced, multi-token, MCQ        | Wiland 2024                        |
| CounterFact       | 21,919 cases                      | Relational (counterfactual)                  | Meng 2022                          |
| zsRE              | varied                            | Zero-shot relation extraction                | Levy 2017 / ROME redistribution    |
| known_1000        | 1,000                             | Well-known facts                              | Meng 2022                          |
| TAXI              | 976 edits / 11,120 queries        | **Categorical / taxonomic**                   | Powell 2024                        |
| RippleEdits       | varies                            | Entailment edits                              | Cohen 2024                         |
| MQuAKE            | varies                            | Multi-hop edits                               | Zhong 2023                         |
| bioS / bioR / bioD| arbitrary, controllable           | Synthetic (name, attribute, value)            | Allen-Zhu 2024                     |

## 7. Gaps and opportunities

1. **No prior work directly measures parameter-volume of *categorical* knowledge specifically.** Allen-Zhu measures uniform-random-attribute storage; Morris measures raw memorization; TAXI measures consistency, not parameter cost.
2. **No bit-budget for entailed properties.** The bit complexity of "peanut → legume" carries downstream entailments (has-shell, contains-protein) "for free" if the model has learned the category abstraction. Quantifying this *compression* is exactly the missing measurement our hypothesis demands.
3. **Cross-model comparison of per-fact footprint** is sparse. He 2025 shows specialization is rising over generations but does not give absolute parameter-density numbers per knowledge-type.
4. **Standard vs long-tail knowledge separation.** Mallen et al. 2023 ("When Not to Trust Language Models") showed long-tail facts behave differently; we should partition our measurements accordingly.
5. **Quantization × knowledge type** is unexplored. Allen-Zhu shows int4 halves capacity; do *standard* facts survive better than long-tail ones (because of redundant encoding)?

## 8. Recommendations for our experiment

Based on this review, the experiment runner should:

### 8.1 Recommended datasets

- **Primary**: **TAXI** (categorical-knowledge consistency) and **BEAR / BEAR-big** (relational coverage across LM types). These two cover the categorical side ("Peanuts are legumes") and the conditional/relational side ("Georgia, country, capital, Tbilisi") of our hypothesis.
- **Supporting**: CounterFact + ROME notebooks for per-fact parameter-footprint measurement; LAMA T-REx for legacy baselines; PARAREL templates to handle paraphrase variance.
- **Synthetic harness**: re-implement bioS / bioD generators (≈100 LoC) to obtain controlled bit-budget numbers we can directly translate to "fraction of parameters consumed".

### 8.2 Recommended baselines

- Allen-Zhu's 2 bpp ceiling (synthetic) and Morris's 3.6 bpp (raw memorization) as upper bounds.
- ROME / Knowledge Neurons per-fact footprint as the *per-fact* baseline.
- Closed-book QA accuracy (BEAR, TAXI, LAMA) as the *coverage* baseline.

### 8.3 Recommended metrics

- **Standard knowledge bit-budget**: `bits_categorical = N_facts × log2(answer_set_size)` summed over TAXI properties + BEAR relations (with majority-class baseline subtracted to get *acquired* bits).
- **Standard knowledge parameter share**: `bits_categorical / (2 bits/param × total_params)` — yields the fraction of the model's bit budget consumed by standard knowledge under the Allen-Zhu accounting.
- **Per-fact parameter footprint**: rank-one update support (ROME) or knowledge-neuron count × parameter count per neuron.

### 8.4 Methodological considerations

- **Tokenizer/answer-length effects**: prefer BEAR-style log-likelihood ranking over single-token cloze.
- **Probability calibration**: avoid majority-class artefacts (LAMA's "Antarctica" issue) by using BEAR or balancing per-relation answers.
- **Architecture coverage**: test at least one model from each of {Pythia, LLaMA, Mistral, GPT-Neo, Phi} sizes to validate generality, since He 2025 showed specialization differs across families.
- **Quantization sweep**: evaluate fp16 / int8 / int4 to test how robust *standard* knowledge is vs long-tail.
- **Junk-data sensitivity**: the 20× hit Allen-Zhu finds with junk data implies our parameter-share estimates are upper bounds; report the figure as "of the bit budget that is actually allocated to knowledge".

### 8.5 First-pass experimental protocol (sketch)

1. Pick a small, modern open LLM (Pythia-410M and Pythia-2.8B, or Llama-3.2-1B / 3B).
2. Run **BEAR** to measure relational-knowledge accuracy → bits acquired.
3. Run **TAXI** to measure categorical-knowledge consistency → bits acquired (with category-membership × property entailment double-counting controlled).
4. Combine with Allen-Zhu's 2 bpp ceiling (or measure model-specific capacity via a brief bit-string memorization probe à la Morris) to convert acquired bits → fraction-of-parameters.
5. Repeat across 3–4 model scales and report scaling.

Expected order-of-magnitude (back-of-envelope, to be refined):
- TAXI ≈ 10⁴ atomic property facts. At 47 bits/biography (Allen-Zhu's bioS) the upper bound is ≈ 10⁵–10⁶ bits.
- LAMA T-REx ≈ 10⁵ facts × ~10 bits = 10⁶ bits.
- A 1B-parameter model has 2 × 10⁹ bits of capacity (Allen-Zhu) → standard-knowledge share **≈ 0.05–0.1%**.
- This validates the qualitative hypothesis; the experiment provides the precise number, partitioned by category, and tracks how it shifts with scale and quantization.
