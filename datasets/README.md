# Datasets

Datasets for the research project **"How much volume is standard knowledge in LLMs?"**.
Data files are excluded from git via `.gitignore` because of their size (CounterFact `attribute_snippets.json` alone is ~900MB).
Follow the download instructions below to reproduce.

Directory layout:

```
datasets/
├── bear/         # BEAR / BEAR-big relational-knowledge probe (Wiland 2024)
├── counterfact/  # ROME / MEMIT editing benchmark (Meng 2022)
├── lama/         # LAMA T-REx probe (Petroni 2019)
├── taxi/         # TAXI categorical-knowledge editing benchmark (Powell 2024)
├── pararel/      # PARAREL paraphrased relational templates (Elazar 2021)
└── samples/      # Tiny samples committed to git for inspection
```

`bear/` and `taxi/` are symlinks into the corresponding cloned repos under `code/`. The actual data is small and ships with the repo.

---

## 1 · BEAR — relational knowledge probe

- **Source**: <https://github.com/lm-pub-quiz/BEAR> (CC BY-SA)
- **Citation**: Wiland, Ploner, Akbik. *BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models.* NAACL Findings 2024. arXiv:2404.04113
- **Size**: BEAR ≈ 7,731 instances · BEAR-big ≈ 40,916 instances · 60 relations
- **Format**: One `.jsonl` per Wikidata relation. Each line:
  ```json
  {"sub_id":"Q1356","sub_label":"West Bengal","sub_aliases":["Paschimbanga"],
   "obj_id":"Q1348","obj_label":"Kolkata","answer_idx":0}
  ```
- **Loading**: use the `lm-pub-quiz` library (already cloned under `code/lm-pub-quiz/`):
  ```bash
  pip install lm-pub-quiz
  ```
  ```python
  from lm_pub_quiz import Dataset
  ds = Dataset.from_path("datasets/bear/BEAR_small")  # or BEAR_big
  ```
- **Why**: This is the cleanest way to measure *what fraction of factual relational knowledge an LLM has internalized*, regardless of architecture. We can use it to compute: (number of correctly-known facts) / (total facts) per model size, then translate to "bits of standard relational knowledge in this model".
- **Already in workspace**: yes (symlink to `code/BEAR/BEAR/` and `code/BEAR/BEAR-big/`)

## 2 · CounterFact + zsRE — editing benchmark

- **Source**: <https://rome.baulab.info/data/dsets/>
- **Citation**: Meng et al. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. arXiv:2202.05262
- **Files** (placed under `datasets/counterfact/`):
  - `counterfact.json` (≈45 MB, 21,919 instances)
  - `known_1000.json` (≈340 KB, well-known facts the model already knows)
  - `zsre_mend_eval.json` (≈8 MB, zero-shot relation extraction edit set)
  - `attribute_snippets.json` (≈900 MB — Wikipedia snippets for compute generalization metrics)
- **Each instance**:
  ```json
  {
    "case_id": 0,
    "requested_rewrite": {
      "prompt": "The mother tongue of {} is",
      "relation_id": "P103",
      "target_new": {"str":"English","id":"Q1860"},
      "target_true": {"str":"French","id":"Q150"},
      "subject": "Danielle Darrieux"
    },
    "paraphrase_prompts": [...],
    "neighborhood_prompts": [...],
    "generation_prompts": [...]
  }
  ```
- **Download (manual)**:
  ```bash
  cd datasets/counterfact
  curl -sLO https://rome.baulab.info/data/dsets/counterfact.json
  curl -sLO https://rome.baulab.info/data/dsets/known_1000.json
  curl -sLO https://rome.baulab.info/data/dsets/zsre_mend_eval.json
  curl -sLO https://rome.baulab.info/data/dsets/attribute_snippets.json
  ```
- **Why**: Lets us measure per-fact editing cost and locate which parameters store specific facts.

## 3 · LAMA / T-REx — relational probe

- **Source**: <https://dl.fbaipublicfiles.com/LAMA/data.zip>
- **Citation**: Petroni et al. *Language Models as Knowledge Bases?* EMNLP 2019. arXiv:1909.01066
- **Contents**:
  - `data/TREx/PXX.jsonl` — 41 Wikidata relations × ≤1000 facts each (~1.5M instances total)
  - `data/Google_RE/`, `data/Squad/`, `data/ConceptNet/`
  - `data/relations.jsonl` — relation templates
- **Each instance** (T-REx):
  ```json
  {"uuid":"...","obj_uri":"Q...","obj_label":"...","sub_uri":"Q...","sub_label":"...","predicate_id":"P36","template":"[X] is the capital of [Y].",...}
  ```
- **Download**:
  ```bash
  cd datasets/lama
  curl -sLO https://dl.fbaipublicfiles.com/LAMA/data.zip
  python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"
  rm data.zip
  ```
- **Why**: Standard reference; lets us replicate prior estimates of factual coverage. Combined with PARAREL (below) for paraphrase robustness.

## 4 · TAXI — categorical knowledge edits

- **Source**: <https://github.com/derekpowell/taxi> (cloned under `code/taxi/`)
- **Citation**: Powell, Gerych, Hartvigsen. *TAXI: Evaluating Categorical Knowledge Editing for Language Models.* ACL Findings 2024. arXiv:2404.15004
- **Size**: 41 categories × 164 subjects × 183 properties → 976 edits, 11,120 MCQ queries
- **Files** (`datasets/taxi/data` is a symlink to `code/taxi/datasets/`):
  - `edits.json` — the 976 categorical edits
  - `edits-evaluation.json` — pre-built MCQ queries with property labels
  - `baseline-evaluation.json` — unedited-model evaluation set
  - `taxonomy/animal-data.tsv`, `food-data.tsv`, ... — raw category/property tables
- **Why central**: The most direct operationalization of *standard categorical knowledge* in our hypothesis. Each (subject, category, property) is exactly the kind of commonsense fact our research targets ("peanuts → legumes → has-shell, contains-protein", etc.). Lets us isolate the parameter sub-budget for *taxonomic* knowledge specifically.

## 5 · PARAREL — paraphrased relational templates

- **Source**: <https://github.com/yanaiela/pararel> (cloned under `datasets/pararel/pararel/`)
- **Citation**: Elazar et al. *Measuring and Improving Consistency in Pretrained Language Models.* TACL 2021.
- **Contents**: Multiple paraphrased templates per LAMA relation — used by Knowledge Neurons paper. Lives in `datasets/pararel/pararel/data/pattern_data/`.
- **Why**: Removes single-template artifacts when measuring whether a model "knows" a fact.

## 6 · bioS / bioD — synthetic biographies (for capacity-scaling experiments)

- **Source**: To be regenerated locally (no canonical release; the Allen-Zhu Physics 3.3 paper publishes generation specs but not the data).
- **Reference**: Allen-Zhu & Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* arXiv:2404.05405. See Definition 2.2 (`bioD(N,K,C,D,L,T)`) and Section 2.2 for `bioS(N)`.
- **Generation sketch** (we will implement during the experiment phase):
  ```
  N0 = 400 × 400 × 1000 candidate names
  6 attributes: birth_date (12·28·200), birth_city (200), university (300),
                major (100), employer (263), gender (2)
  → ≈47.6 bits of knowledge per person, ignoring names.
  Each biography rendered with 50 templates per attribute, randomly ordered.
  ```
- **Why**: Allows controlled estimation of bits-per-parameter consumed by *standard categorical knowledge* with no benchmark contamination.

---

## Sample files (committed to git)

Tiny fragments are kept under `<dataset>/samples/` so reviewers can see the schema without downloading anything:

- `bear/samples/P36_HasCapital_first5.jsonl` — first 5 BEAR HAS-CAPITAL queries
- `bear/samples/P30_OnContinent_first5.jsonl` — first 5 BEAR ON-CONTINENT queries
- `counterfact/samples/counterfact_first5.json` — first 5 CounterFact edit cases
- `lama/samples/P36_HasCapital_first5.jsonl` — first 5 LAMA T-REx HAS-CAPITAL facts
- `lama/samples/P101_FieldOfWork_first5.jsonl` — first 5 LAMA T-REx FIELD-OF-WORK facts
- `taxi/samples/edits_eval_head.json` — first 2 KB of TAXI evaluation queries

---

## Recommended primary datasets for the experiment

| Goal                                                          | Primary dataset            | Secondary           |
|---------------------------------------------------------------|----------------------------|---------------------|
| Estimate *categorical* knowledge volume (the hypothesis)      | **TAXI** + LAMA T-REx      | PARAREL             |
| Estimate *relational* knowledge volume across modern LLMs     | **BEAR / BEAR-big**        | LAMA T-REx          |
| Estimate per-fact parameter footprint                          | **CounterFact** + ROME     | Knowledge Neurons   |
| Run controlled bits-per-parameter scaling experiments          | **bioS / bioD** (synthesize) | bioR              |
