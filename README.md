# XOver

Code and dataset for **XOver**, a cross-lingual Text-to-OverpassQL framework, together
with the **MOsmNL** multilingual benchmark (8 languages).

## Repository layout

```
pipeline/
  c1_eval/        Benchmark evaluation harness (EM / OQS / EX / EX_soft)
  c2_retrieval/   Cross-lingual demonstration mining: utility scoring
                  (similarity + grounding), submodular/DPP selection, prompt assembly
  c3_selection/   Execution-aware multi-candidate selection
                  (denotation signatures, voting selectors, oracle Pass@K)
  c4_revision/    Conservative post-hoc revision (syntax / geo-binding / hygiene repair)
scripts/          Entry-point scripts (index building, few-shot prompt building,
                  inference, candidate generation, execution, analysis, plotting)
conf/             Per-language YAML configurations for demonstration mining
tests/            Unit tests (pytest)
dataset/          MOsmNL benchmark data (see below)
```

## Dataset

`dataset/OsmNL_<lang>/<split>_final_<lang>.json` for 8 languages: `english`,
`mandarin_simplified`, `cantonese`, `french`, `german`, `japanese`, `korean`,
`russian`. Splits: `train` (6047), `dev` (1000), `test` (1000), `syn` (133).
The OverpassQL gold target is identical across languages.

Each item contains: `sample_index`, `split`, `nl_question`, `OverpassQL`,
`location` (`bbox`, `mentioned_region`), and `tags`.

`dataset/relation_balanced/` holds the relation-balanced evaluation subset
(English plus a 400-item multilingual slice).

## Pipeline

```
Multilingual NL question
   |  C2  cross-lingual demonstration mining        pipeline/c2_retrieval/
   |      score -> select (indices) -> assemble (English demo pool)
   |  few-shot LLM candidate generation             scripts/{build_fewshot,infer_fewshot}.py
   |  C3  execution-aware candidate selection       pipeline/c3_selection/
   |  C4  conservative revision                     pipeline/c4_revision/
   v  C1  evaluation: EM / OQS / EX / EX_soft       pipeline/c1_eval/runner.py
Final OverpassQL
```

C2 is split into three pluggable stages: **score** (`scores.py`, a `ScoreFunction`
protocol over the demo pool), **select** (`select.py`, a `Selector` protocol returning
ordered pool indices; `TopKSelector` for utility top-k, `ProfileDppSelector` for the
cross-lingual profile-DPP diversity variant), and **assemble** (`assemble.py`, indices
to plain `(#question, #OverpassQL)` prompts).

## Requirements

- Python 3.10+, `pytest`, `numpy`, `openai`, `sentence-transformers`, `matplotlib`.
- **Execution-based metrics (EX / EX_soft)** require a local Overpass API instance
  with a frozen OSM snapshot, configured via `--overpass_urls`
  (e.g. `http://localhost:12346/api/interpreter`). Placeholders such as
  `REMOTE_OVERPASS_HOST` and `/path/to/...` in scripts and configs must be replaced
  with your own endpoints and paths.
- LLM inference scripts require provider API keys (OpenAI-compatible endpoints);
  the key-loading hook in `scripts/run_baseline_llm.py` and
  `pipeline/c4_revision/llm.py` should be pointed at your own key store.

## Quick start

```bash
# Unit tests (no server or keys needed)
python -m pytest tests/ -q

# C2: build the per-language sBERT ICL index, then few-shot prompts
python scripts/build_icl_index.py
python scripts/build_fewshot.py --split dev --k 6 --alpha 0.2

# Generate candidates with an LLM
python scripts/infer_fewshot.py --questions data/fewshot/questions_dev_k6_a0.2_plain.json \
    --model <model> --provider <provider>

# C3: sample N candidates, execute, and run the selectors
python scripts/diag_gen_candidates.py ...
python scripts/diag_execute.py ...
python scripts/diag_analyze.py ...

# C4: conservative revision of a prediction file
python scripts/run_revision.py --input_file <preds.json> --lang english --split dev \
    --model <model> --provider <provider> --max_repair_attempts 3

# C1: evaluate (dev: EM/OQS; test: add --compute_execution for EX/EX_soft)
python -m pipeline.c1_eval.runner --input_file <preds.json> --lang english --split dev
```

## Prediction file contract

Any results JSON passed to the evaluation runner must contain entries with
`query_index`, `query_target_ovq`, and `generated_text`, either as a bare list or
as `{"data": [...], "statistics": {...}}`.
