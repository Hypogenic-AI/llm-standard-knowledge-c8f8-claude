# Downloaded Papers

Papers gathered for the research project: **"How much volume is standard knowledge in LLMs?"**
Each entry includes purpose, why-relevant, and the most important quantitative claims to anchor the experiment design.

The PDFs are saved alongside chunked subdirectories under `pages/` produced by `.claude/skills/paper-finder/scripts/pdf_chunker.py`, which were used for in-context reading.

---

## Tier 1 — Centrepiece (must read in full)

### 1. Physics of Language Models: Part 3.3 — Knowledge Capacity Scaling Laws (Allen-Zhu & Li, 2024)
- **File**: `physics3_3_knowledge_capacity.pdf`
- **arXiv**: 2404.05405
- **Key claim**: After sufficient training, transformer LMs store **~2 bits of knowledge per parameter** (universal across GPT-2, LLaMA, Mistral; with int8 quantization). With 100 exposures it drops to ~1 bit/param. With int4 quantization it drops to 0.7 bit/param.
- **Methodology**: Synthetic biographies (`bioS`, `bioSsimple`, `bioR`, `bioD(N,K,C,D,L,T)`) — knowledge tuples `(name, attribute, value)`. The paper derives an information-theoretic bit-complexity lower bound (Theorem 3.2) and defines a **Capacity Ratio R(F) = learned_bits / parameters**.
- **Why central**: This is the directly comparable framework for our hypothesis. It defines a quantitative "knowledge bit" so we can ask what fraction of total parameters is consumed by *standard* (i.e., commonsense / categorical / conditional) knowledge.
- **Relevant results to reuse**: `bioS(N)` data generator, peak capacity ratio plots, the 2 bit/param ceiling for synthetic facts.

### 2. How much do language models memorize? (Morris et al., 2025)
- **File**: `morris2025_how_much_lms_memorize.pdf`
- **arXiv**: 2505.24832 (FAIR/Meta + Cornell + DeepMind + NVIDIA)
- **Key claim**: GPT-style transformers store **3.5–4 bits per parameter** (specifically α ≈ 3.64 bits/param in fp16). Decomposes memorization into *unintended memorization* (about a specific dataset) and *generalization* (about the data-generating process). Uses Kolmogorov-style compression to instantiate per-sample memorization at the algorithmic-information level.
- **Methodology**: Train hundreds of GPT-style models (500K → 1.5B params) on uniform random bitstrings (no generalization possible) to measure raw capacity, then on real text. Identifies double-descent as occurring exactly when dataset size exceeds capacity. Derives scaling laws for membership inference.
- **Why central**: Companion result to Allen-Zhu — different definition (raw bits about data) but the *same order of magnitude* for the headline number. Together these set the upper-bound budget within which "standard knowledge" must fit.

---

## Tier 2 — Mechanistic Interpretability (where knowledge lives)

### 3. Locating and Editing Factual Associations in GPT (Meng, Bau et al., 2022 — ROME)
- **File**: `meng2022_locating_editing_rome.pdf` · arXiv 2202.05262
- **Key claim**: Factual associations `(s, r, o)` in GPT correspond to **localized, directly-editable computations**. Causal-tracing (clean / corrupted / patched runs) shows two important sites: (1) an *early site* in middle MLP layers at the last subject token, (2) a *late site* in last layers at the last token. **Mid-layer MLPs store the associations**; attention copies them to the last token.
- **Editing**: Rank-One Model Editing (ROME) updates `W_proj` of one MLP layer with `Λ(C⁻¹k*)ᵀ` to enforce a new key→value mapping while minimising interference (closed-form least-squares with cached covariance C = KKᵀ).
- **Why relevant**: ROME's localization gives us per-fact parameter footprint estimates (rank-one updates → bytes per fact).

### 4. Knowledge Neurons in Pretrained Transformers (Dai et al., 2021 → ACL 2022)
- **File**: `dai2021_knowledge_neurons.pdf` · arXiv 2104.08696
- **Key claim**: Using integrated-gradient attribution, ~4 FFN-intermediate neurons per relational fact (BERT-base-cased, PARAREL/T-REx, 27,738 facts). Knowledge neurons concentrate in **upper layers** of BERT. Suppressing/amplifying neurons shifts fact probabilities; surgical edits update facts without fine-tuning.
- **Why relevant**: Lower-bound estimate of how few neurons it takes to express a single fact.

### 5. Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020 → EMNLP 2021)
- **File**: `geva2020_ffn_kv_memories.pdf` · arXiv 2012.14913
- **Key claim**: The two FFN matrices in each transformer block act as **keys** (input-pattern detectors) and **values** (output distributions). Lower layers detect shallow textual patterns; upper layers learn semantic patterns. This is the canonical mechanistic frame used by ROME, Knowledge Neurons, MEMIT, and parameter-specialization analyses.
- **Why relevant**: Theoretical scaffolding for "knowledge lives in FFN".

### 6. The Rise of Parameter Specialization for Knowledge Storage in LLMs (Hong et al., 2025 — NeurIPS 2025)
- **File**: `he2025_parameter_specialization.pdf` · arXiv 2505.17260
- **Key claim**: Across 20 open-source LLMs, **stronger / more recent models exhibit greater parameter specialization** — fewer MLP value vectors per concept, each governing a narrower set of concepts. Defines a Parameter Specialization Score (PSS) using concept-specific masking. Causal training experiments confirm specialization → better knowledge utilization.
- **Why relevant**: Suggests the *fraction* of parameters dedicated to a single fact has been *decreasing* with model generations — directly informs how to interpret our parameter-volume estimates.

---

## Tier 3 — Knowledge Probing Benchmarks

### 7. Language Models as Knowledge Bases? — LAMA (Petroni et al., 2019)
- **File**: `petroni2019_lama.pdf` · arXiv 1909.01066
- **Key contribution**: Cloze-style probe of relational knowledge. T-REx subset: 41 Wikidata relations × ≤1000 facts each. Standard but limited to single-subtoken answers and masked LMs.

### 8. BEAR — Unified Framework for Evaluating Relational Knowledge (Wiland, Ploner, Akbik — NAACL Findings 2024)
- **File**: `wiland2024_bear.pdf` · arXiv 2404.04113
- **Key contribution**: Reformulates probing as **log-likelihood ranking** over fixed multiple-choice answers, working for both causal and masked LMs. Releases two datasets:
  - BEAR (small): 7,731 instances, 60 relations, 4–53 answer options
  - BEAR-big: 40,916 instances
- **Why relevant**: This is the most modern, balanced, and tokenizer-agnostic probe — directly usable to measure knowledge volume across modern LLMs of any architecture.

### 9. How Much Knowledge Can You Pack Into the Parameters of a Language Model? (Roberts, Raffel, Shazeer — EMNLP 2020)
- **File**: `roberts2020_knowledge_pack.pdf` · arXiv 2002.08910
- **Key contribution**: Scaling-style closed-book QA experiments with T5 (220M → 11B). Shows knowledge accuracy on Natural Questions / TriviaQA / WebQuestions grows with parameter count. Foundational counterpart to Allen-Zhu in real (uncontrolled) data.

### 10. TAXI — Evaluating Categorical Knowledge Editing (Powell, Gerych, Hartvigsen — ACL Findings 2024)
- **File**: `powell2024_taxi.pdf` · arXiv 2404.15004
- **Key contribution**: Hand-curated benchmark for **categorical** knowledge edits: 41 categories × 164 subjects × 183 properties → 976 edits, 11,120 multiple-choice queries. Splits property success into *consistency* (changed properties) and *invariance* (unchanged). Editors (FT, ROME, ICE) achieve only 0.43–0.47 consistency vs human 0.86.
- **Why central**: Directly aligned with our hypothesis — "Peanuts are legumes" is exactly categorical knowledge (subject → category → inherited properties). TAXI's taxonomy and property metric give us a ready-made operationalization of *standard categorical knowledge*.

---

## Tier 4 — Supporting / Related

### 11. Examining Two-Hop Reasoning Through Information Content Scaling (2025)
- **File**: `mh2025_two_hop_info_content.pdf` · arXiv 2502.03490
- Relevance: Extends the bit-complexity / scaling-law framework to multi-hop reasoning over stored facts.

### 12. Kformer — Knowledge Injection in Transformer FFN (Yao et al., 2022)
- **File**: `feng2022_kformer.pdf` · arXiv 2201.05742
- Relevance: Explicit construction that injects external knowledge into FFN keys/values; corroborates that FFNs are the natural site for knowledge storage.

### 13. Constructing Efficient Fact-Storing MLPs for Transformers (2025)
- **File**: `constructing_fact_storing_mlps.pdf` · arXiv 2512.00207
- Relevance: Proposes purpose-built MLPs that store more facts per parameter — useful baseline for "what's the parameter cost per fact".

### 14. Large Language Models as Reliable Knowledge Bases? (Yu et al., 2024)
- **File**: `llms_reliable_kb.pdf` · arXiv 2407.13578
- Relevance: Empirical assessment of LLM-as-KB across models; measures knowledge coverage and reliability.

---

## Suggested reading order for the experiment runner

1. **Allen-Zhu Physics 3.3** (decide on bioS / bioD as our synthetic harness)
2. **Morris 2025** (cross-check 3.6 bpp result against the 2 bpp ceiling)
3. **TAXI** (categorical-knowledge benchmark — most aligned with the "standard knowledge" angle)
4. **BEAR** (probe modern open LLMs for relational knowledge volume)
5. **ROME / Knowledge Neurons** (per-fact parameter footprint estimation)
6. **He 2025** (parameter specialization evolution across generations)
