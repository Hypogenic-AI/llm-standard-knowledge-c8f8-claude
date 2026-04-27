# Resources Catalog

**Project**: How much volume is *standard knowledge* in LLMs?
**Generated**: 2026-04-27

## Summary

| Category    | Count |
|-------------|-------|
| Papers      | 14    |
| Datasets    | 6 (CounterFact, LAMA/T-REx, BEAR, TAXI, PARAREL, bioS/bioD spec) |
| Code repos  | 8     |

---

## Papers

| # | Title | Authors | Year | File | Key claim |
|---|-------|---------|------|------|-----------|
| 1 | **Physics of LMs Part 3.3 — Knowledge Capacity Scaling Laws** | Allen-Zhu, Li | 2024 | `papers/physics3_3_knowledge_capacity.pdf` | 2 bits/param (universal) |
| 2 | **How much do language models memorize?** | Morris et al. | 2025 | `papers/morris2025_how_much_lms_memorize.pdf` | 3.6 bits/param (raw) |
| 3 | **Locating and Editing Factual Associations in GPT (ROME)** | Meng, Bau et al. | 2022 | `papers/meng2022_locating_editing_rome.pdf` | Mid-MLP layers store facts; rank-1 edit |
| 4 | **Knowledge Neurons in Pretrained Transformers** | Dai et al. | 2021 | `papers/dai2021_knowledge_neurons.pdf` | ~4 FFN neurons per fact |
| 5 | **Transformer FFN Layers Are Key-Value Memories** | Geva et al. | 2020 | `papers/geva2020_ffn_kv_memories.pdf` | FFN = KV memory |
| 6 | **The Rise of Parameter Specialization for Knowledge Storage** | Hong et al. | 2025 | `papers/he2025_parameter_specialization.pdf` | Specialization rises with model generation |
| 7 | **Language Models as Knowledge Bases? (LAMA)** | Petroni et al. | 2019 | `papers/petroni2019_lama.pdf` | Cloze probing baseline |
| 8 | **BEAR: Unified Relational-Knowledge Probe** | Wiland, Ploner, Akbik | 2024 | `papers/wiland2024_bear.pdf` | Log-likelihood probe across LM types |
| 9 | **How Much Knowledge Can You Pack Into LM Parameters?** | Roberts, Raffel, Shazeer | 2020 | `papers/roberts2020_knowledge_pack.pdf` | Scale → more closed-book QA |
| 10 | **TAXI: Evaluating Categorical Knowledge Editing** | Powell, Gerych, Hartvigsen | 2024 | `papers/powell2024_taxi.pdf` | Categorical consistency benchmark |
| 11 | **Examining Two-Hop Reasoning Through Information Content Scaling** | — | 2025 | `papers/mh2025_two_hop_info_content.pdf` | Multi-hop info-content scaling |
| 12 | **Kformer: Knowledge Injection in FFN** | Yao et al. | 2022 | `papers/feng2022_kformer.pdf` | External knowledge into FFN |
| 13 | **Constructing Efficient Fact-Storing MLPs** | — | 2025 | `papers/constructing_fact_storing_mlps.pdf` | Hand-built fact-dense MLPs |
| 14 | **LLMs as Reliable Knowledge Bases?** | Yu et al. | 2024 | `papers/llms_reliable_kb.pdf` | LM-as-KB reliability eval |

See `papers/README.md` for detailed per-paper notes (methodology, datasets, results, relevance).

---

## Datasets

All data files are excluded from git via `datasets/.gitignore`. Tiny samples are committed under `datasets/<name>/samples/`.

| Name              | Source                                    | Scale                   | Task                                | Location                                | Notes |
|-------------------|-------------------------------------------|-------------------------|--------------------------------------|------------------------------------------|-------|
| BEAR              | github.com/lm-pub-quiz/BEAR              | 7,731 / 40,916 facts    | Relational probe (LL ranking)        | `datasets/bear/` (symlink → `code/BEAR`) | 60 Wikidata relations, balanced |
| CounterFact       | rome.baulab.info                         | 21,919 cases            | Counterfactual editing               | `datasets/counterfact/counterfact.json` (45 MB) | + paraphrase + neighborhood prompts |
| zsRE (mend eval)  | rome.baulab.info                         | ~20K                    | Zero-shot relation extraction        | `datasets/counterfact/zsre_mend_eval.json` | |
| known_1000        | rome.baulab.info                         | 1,000                   | Curated well-known facts              | `datasets/counterfact/known_1000.json` | |
| attribute_snippets| rome.baulab.info                         | ~900 MB                 | Wikipedia evidence snippets           | `datasets/counterfact/attribute_snippets.json` | needed for ROME generalization metrics |
| LAMA T-REx        | dl.fbaipublicfiles.com/LAMA/data.zip     | 41 rels × ≤1000 facts   | Cloze probing                         | `datasets/lama/data/TREx/` | + Google_RE, Squad, ConceptNet subsets |
| TAXI              | github.com/derekpowell/taxi              | 976 edits / 11,120 MCQs | Categorical knowledge editing         | `datasets/taxi/` (symlink → `code/taxi`) | 41 categories × 164 subjects × 183 properties |
| PARAREL           | github.com/yanaiela/pararel              | 27.7K facts × 8.6 templates | Paraphrased relational templates  | `datasets/pararel/pararel/` | Used by Knowledge Neurons |
| bioS / bioD       | Allen-Zhu Physics 3.3 (spec only)         | controllable            | Synthetic biographies for capacity scaling | TBD — implement during experiment phase | ~100 LoC re-implementation |

See `datasets/README.md` for download commands and per-dataset schemas.

---

## Code repositories

All cloned shallow (`--depth=1`) under `code/`.

| Name              | URL                                          | Purpose                                              | Location                       |
|-------------------|----------------------------------------------|-------------------------------------------------------|--------------------------------|
| BEAR              | github.com/lm-pub-quiz/BEAR                  | BEAR / BEAR-big datasets + relation metadata          | `code/BEAR/`                   |
| lm-pub-quiz       | github.com/lm-pub-quiz/lm-pub-quiz           | Probing library (works for causal & masked LMs)       | `code/lm-pub-quiz/`            |
| LAMA              | github.com/facebookresearch/LAMA             | Original 2019 probing code & download script          | `code/LAMA/`                   |
| ROME              | github.com/kmeng01/rome                      | Causal tracing + ROME edit                            | `code/rome/`                   |
| MEMIT             | github.com/kmeng01/memit                     | Mass editing memory in a transformer                  | `code/memit/`                  |
| knowledge-neurons | github.com/EleutherAI/knowledge-neurons      | Generalised knowledge-neuron attribution              | `code/knowledge-neurons/`      |
| ff-layers         | github.com/mega002/ff-layers                 | Geva et al. FFN-as-KV analysis                        | `code/ff-layers/`              |
| TAXI              | github.com/derekpowell/taxi                  | TAXI dataset + benchmark + EasyEdit fork               | `code/taxi/`                   |

See `code/README.md` for installation, smoke tests, and per-repo entry points.

---

## Resource gathering notes

### Search strategy

Two diligent paper-finder searches ("knowledge storage capacity transformer language models bits per parameter", "knowledge probing language models LAMA factual recall") and three fast searches were run via `.claude/skills/paper-finder/scripts/find_papers.py`. The diligent runs returned 235 + 100+ papers; we kept those with relevance ≥ 2 and prioritised:
- Centrepieces (Allen-Zhu, Morris) — *quantitative bit-budget*.
- Mechanistic (ROME, Knowledge Neurons, FFN-KV, He 2025) — *where knowledge lives*.
- Probing (LAMA, BEAR, TAXI, Roberts) — *what fraction of knowledge LMs contain*.

### Selection criteria

Each paper was kept if it provided one of:
1. A quantitative *bit-budget* claim (Allen-Zhu, Morris).
2. A localization mechanism (Geva, Dai, Meng, Hong).
3. A reusable evaluation benchmark with downloadable data (LAMA, BEAR, TAXI, CounterFact, PARAREL).
4. A specifically *categorical* / taxonomic angle (TAXI, ConcEPT, Knowledge-Neurons via PARAREL).

### Challenges encountered

- **Working directory drift in shell**: Bash `cd && cmd &` placed background outputs in the parent dir; mitigated by absolute paths thereafter and a one-shot `mv` to consolidate.
- **No `wget` / `unzip`** in the sandbox — substituted with `curl -sL` and `python -m zipfile`.
- **Semantic Scholar 429 rate-limits** when bulk-resolving CorpusIds → retried with backoff and used WebSearch as fallback.
- **Diligent paper-finder timeout** on one of the queries (mechanistic-interpretability run); fell back to fast mode.
- **No public release of the Allen-Zhu Physics 3.3 training code** — bioS / bioD will be re-implemented from the paper's Definition 2.2.
- **`attribute_snippets.json` is ~900 MB** — kept locally but `.gitignore`'d.

### Gaps and workarounds

- We did not download model checkpoints (heavy). The experiment runner can pull HuggingFace checkpoints lazily via `transformers`.
- We did not pre-build the bioS/bioD generators; this is deferred to the experiment phase but spec'd in `datasets/README.md` § 6.
- `code/lm-pub-quiz/` may be installable directly via `pip install lm-pub-quiz`; we kept the source clone for transparency.

---

## Recommendations for experiment design

1. **Primary datasets**:
   - **TAXI** for categorical-knowledge consistency (the headline test of our hypothesis).
   - **BEAR / BEAR-big** for relational-knowledge volume across modern LLMs.

2. **Baselines**:
   - Allen-Zhu's 2 bpp ceiling (synthetic) and Morris's 3.6 bpp (raw memorization) as upper bounds on bit-budget.
   - ROME / Knowledge-Neurons per-fact footprint as the per-fact baseline.

3. **Metrics**:
   - **Standard-knowledge bit-budget**: bits acquired on TAXI + BEAR (with majority-class baseline subtracted).
   - **Standard-knowledge parameter share**: bits / (2 × total_params).
   - **Per-fact parameter footprint**: rank-one update support (ROME).

4. **Code to adapt**:
   - `lm-pub-quiz` (drop-in BEAR scoring for any HF LM).
   - `rome/notebooks/causal_trace.ipynb` (per-fact localization).
   - `taxi/benchmark.py` (categorical consistency evaluation).

5. **Models to test**:
   - Pythia-{160M, 410M, 1B, 2.8B} (clean scaling family).
   - Llama-3.2-{1B, 3B} (modern, well-studied).
   - At least one Mistral-7B variant (different architecture).

This catalog and the accompanying `literature_review.md` should give the experiment runner everything needed to begin without further literature search.
