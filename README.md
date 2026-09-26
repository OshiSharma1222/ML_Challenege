# Business Entity Resolution — reproducible pipeline

Matches Source 2 / Source 3 business records to deduplicated Source 1 entities.
CPU only (developed on 18 cores, 16 GB RAM, no GPU). No external data or APIs.
The only learned models are two LightGBM gradient-boosted tree models (MIT license) trained from scratch on the provided training data.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux / macOS
```

Point the pipeline at the dataset folder (the one containing `train/` and `test/`):

```bash
export ER_DATA_DIR=/path/to/student_resource/dataset
```

Intermediate files go to `work/` (about 8 GB), and final outputs go to `output/`. Both locations can be changed with `ER_WORK_DIR` and `ER_OUT_DIR`.

## Run end to end

```bash
PYTHON=.venv/Scripts/python bash run_all.sh
```

The script runs these stages in order. Run them one at a time, because two heavy stages in parallel exhaust 16 GB of RAM.

| Step | Script | What it does | Time |
| --- | --- | --- | --- |
| 1 | `src/prep.py train test` | Transliterate (anyascii) and normalise names and addresses; build legal-form, state and house-number fields (US states, Indian states, French regions/departments) | ~5 min |
| 2 | `src/blocking.py train` / `test` | TF-IDF sparse top-K candidate generation over phonetic *and* exact-spelling name tokens plus address tokens: a name+address index (K=20) unioned with a name-only index (K=10) | ~15-18 min each |
| 3 | `src/train.py 500000` | Stage-1 LightGBM pair model (66 features), validation split, threshold sweep | ~22 min |
| 4 | `src/extend_val.py 0.08`, then `0.10 --out=val3` | Stage-1 scores for two further disjoint held-out entity sets (more stage-2 training data) | ~20 min each |
| 5 | `src/train_s2.py --ext=3 --noent --tag=_noent3` | Stage-2 LightGBM re-scorer (43 features, no entity-side features), 5-fold CV, test-like check, decision rule | ~23 min |
| 6 | `src/predict.py --tag=_noent3` | Test inference; writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` | ~3 h |

Times are with 10 threads (`ER_JOBS=10`, `POLARS_MAX_THREADS=10`).

Validate the outputs:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source layout

| File | Role |
| --- | --- |
| `src/config.py` | Paths and thread count (environment overridable) |
| `src/normalize.py` | Transliteration, phonetic skeleton, legal suffixes, address abbreviations, state canonicalisation |
| `src/prep.py` | Stage 1: normalise raw TSVs into `work/{split}_records.parquet` |
| `src/blocking.py` | Stage 2: candidate generation (sparse_dot_topn) |
| `src/eval_block.py` | Blocking recall@K diagnostics |
| `src/features.py` | Pairwise similarity features (rapidfuzz, IDF cosines, numbers, within-query margins) |
| `src/pipeline.py` | Ground truth, chunked featurisation, assignment, macro-F0.5 scorer |
| `src/train.py` | Stage-1 model training and validation |
| `src/stage2.py`, `src/fine.py` | Stage-2 context features and fine-grained name/number/legal features |
| `src/train_s2.py` | Stage-2 training, CV, and choosing the final configuration (`work/final_ext.json`) |
| `src/extend_val.py` | Extra out-of-sample held-out entities for stage-2 training |
| `src/decide.py` | Entity-level decision rule maximising expected F0.5 |
| `src/analyze.py` | Validation error buckets and the F0.5 each one costs |
| `src/predict.py` | Test inference and output writing |
| `src/fix_fr.py` | Patch utility only: re-normalises France test records and re-scores the affected queries in an existing `work/` (a clean `run_all.sh` does not need it) |

## Decision rule

Each Source 2/3 record belongs to at most one Source 1 entity (this holds for every record in the training labels).
Each record therefore first picks its single highest-scoring Source 1 candidate.
Then, for each Source 1 entity, the records that picked it are sorted by stage-2 probability, and the entity keeps the top k that maximise *expected* F0.5 (k = 0 allowed), with outcomes sampled from those probabilities (`src/decide.py`).
Because the metric is a per-entity average, this beats one global threshold: an entity with no true match drops from F = 1 to 0 on a single false merge, so doubtful records are only added where they are likely to help.
Entities with no kept records are output as singletons (an empty list).
