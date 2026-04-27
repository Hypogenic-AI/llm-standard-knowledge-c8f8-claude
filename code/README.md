# Cloned Repositories

External codebases providing implementations, datasets, and evaluation harnesses we will use for the experiments. All clones are shallow (`--depth=1`) to keep the workspace small.

| Repo                  | Purpose                                                                 | Used for                                              |
|-----------------------|--------------------------------------------------------------------------|-------------------------------------------------------|
| `BEAR/`               | BEAR & BEAR-big datasets + relation metadata                            | Probing relational knowledge in causal & masked LMs   |
| `lm-pub-quiz/`        | Library that runs BEAR-style log-likelihood probes                       | Evaluation harness                                     |
| `LAMA/`               | Original LAMA probe code (T-REx, Google_RE, Squad, ConceptNet)           | Single-token cloze probing baseline                    |
| `rome/`               | ROME (rank-one model editing) reference implementation                   | Locating + editing single facts in MLPs               |
| `memit/`              | MEMIT (mass editing memory in a transformer)                             | Bulk editing thousands of facts simultaneously        |
| `knowledge-neurons/`  | Integrated-gradients knowledge-neuron identification (BERT)              | Per-fact neuron-count estimates                       |
| `ff-layers/`          | Geva et al. "FFN-as-key-value-memory" analysis code                      | Mechanistic baseline                                   |
| `taxi/`               | TAXI categorical-knowledge editing benchmark + analysis                  | Categorical-knowledge consistency evaluation           |

## 1 · BEAR — `lm-pub-quiz/BEAR`

- **URL**: <https://github.com/lm-pub-quiz/BEAR>
- **Provides**: 60 Wikidata-relation `.jsonl` files (BEAR small + big), a global `relation_info.json` and `all_entities.json`. License: CC BY-SA.
- **Data only**, no Python code. Pair with `lm-pub-quiz/` to run probes.
- **Already symlinked** from `datasets/bear/`.

## 2 · lm-pub-quiz — `lm-pub-quiz/lm-pub-quiz`

- **URL**: <https://github.com/lm-pub-quiz/lm-pub-quiz>
- **Provides**: Modern probing library that scores any HF causal/masked LM via per-statement log-likelihood. Compatible with the BEAR dataset.
- **Install**:
  ```bash
  cd code/lm-pub-quiz && pip install -e .
  ```
- **Minimum example** (`lm_pub_quiz` API, also installable as `pip install lm-pub-quiz`):
  ```python
  from lm_pub_quiz import Dataset, Evaluator, MaskedLMScorer, CausalLMScorer
  ds = Dataset.from_path("datasets/bear/BEAR_small")
  scorer = CausalLMScorer.from_pretrained("EleutherAI/pythia-410m")
  results = Evaluator(scorer).evaluate(ds)
  ```
- **Why central**: Drop-in scoring works for every modern LLM family — exactly what we need to compare knowledge volume across architectures and sizes.

## 3 · LAMA — `facebookresearch/LAMA`

- **URL**: <https://github.com/facebookresearch/LAMA>
- **Provides**: Original 2019 probe code (mainly for masked LMs). Useful for replicating early baselines and templates. Includes `download_models.sh` for the original BERT/ELMo checkpoints (heavy, optional).
- **Data download**: `datasets/lama/data/` (already extracted).

## 4 · ROME — `kmeng01/rome`

- **URL**: <https://github.com/kmeng01/rome>
- **Provides**: Causal tracing notebooks, ROME edit implementation, CounterFact loader (`dsets/counterfact.py`), zsRE loader, Wikipedia-snippet covariance pre-computation. Hosts the canonical CounterFact / known_1000 / zsre_mend_eval / attribute_snippets URLs (rome.baulab.info).
- **Notable scripts**:
  - `experiments/causal_trace.py` — reproduce the Figure 2 causal traces
  - `notebooks/causal_trace.ipynb` — interactive demo
  - `experiments/evaluate.py` — run an editor on CounterFact and report Efficacy / Generalization / Specificity

## 5 · MEMIT — `kmeng01/memit`

- **URL**: <https://github.com/kmeng01/memit>
- **Provides**: Closed-form rank-multiple updates that edit many facts simultaneously (paper: arXiv:2210.07229). Useful for measuring "how many facts can we cram into a fixed parameter budget" — the same question as our hypothesis from the supply side.
- **Note**: Has its own bundled `rome/` subfolder (older snapshot) — prefer the standalone `rome/` clone for canonical ROME.

## 6 · Knowledge Neurons — `EleutherAI/knowledge-neurons`

- **URL**: <https://github.com/EleutherAI/knowledge-neurons>
- **Provides**: General library for computing knowledge-neuron attribution (works on BERT, GPT-2, GPT-Neo, etc. — generalised from the Dai et al. paper).
- **Pre-computed neurons**: `bert_base_uncased_neurons/` ships with neuron sets per relation for BERT.
- **Example notebook**: `examples/knowledge_neurons.ipynb`.

## 7 · ff-layers — `mega002/ff-layers`

- **URL**: <https://github.com/mega002/ff-layers>
- **Provides**: Code accompanying Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories" (EMNLP 2021). Used for analysing what FFN keys/values activate on, and as a baseline for the more recent parameter-specialization work.

## 8 · TAXI — `derekpowell/taxi`

- **URL**: <https://github.com/derekpowell/taxi>
- **Provides**:
  - `datasets/edits.json`, `datasets/edits-evaluation.json`, `datasets/baseline-evaluation.json`
  - `taxonomy/<group>-data.tsv` — categorical structure
  - `benchmark.py`, `build-datasets.py` — re-build pipeline
  - `easyeditor/` — vendored EasyEdit fork tweaked for TAXI metrics
  - `analysis.ipynb`, `human-tests/`
- **Why central**: TAXI implements *exactly* the categorical-consistency metric our research hypothesis demands.

---

## Installation requirements (combined)

The eight repos depend on a common stack:

- `torch`, `transformers`, `datasets`
- `numpy`, `scipy`, `tqdm`, `nltk`, `pandas`
- `huggingface_hub`
- For ROME / MEMIT: `hydra-core`, `einops`
- For lm-pub-quiz: `minicons`

The experiment runner phase will create a single environment that satisfies all of these. The current `pyproject.toml` already contains the bare resource-finder dependencies (`pypdf`, `requests`, `httpx`).

## Resources we did NOT clone (and why)

- **Physics of LMs official code** — There is no public release of the Knowledge-Capacity-Scaling-Laws training code from Allen-Zhu & Li. The bioS / bioD generators are described in enough detail to reimplement (≈100 LoC). We will write a clean re-implementation during the experiment phase rather than depend on a non-existent repo.
- **WikiUNI** (Cao et al. 2021) — superseded by BEAR for our purposes.
- **KAMEL** (Kalo & Fichtel 2022) — generation-only, less directly comparable to log-likelihood ranking.

---

## Quick smoke test

After installing dependencies, sanity-check that ROME's CounterFact loader works:

```bash
cd code/rome
python - <<'PY'
import sys; sys.path.insert(0, '.')
from dsets.counterfact import CounterFactDataset
ds = CounterFactDataset(data_dir="../../datasets/counterfact", size=5)
print(len(ds), ds[0]['requested_rewrite']['prompt'])
PY
```
Expected: `5 The mother tongue of {} is`.
