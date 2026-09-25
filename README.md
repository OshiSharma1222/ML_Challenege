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
| 1 | `src/prep.py train test` | Transliterate (anyascii) and normalise names and addresses; build legal-form, state and house-number fields | ~5 min |
| 2 | `src/blocking.py train` / `test` | TF-IDF sparse top-K candidate generation: a name+address index (K=20) unioned with a name-only index (K=10) | ~17 min each |
| 3 | `src/train.py 500000` | Stage-1 LightGBM pair model (68 features), validation split, threshold sweep | ~25 min |
| 4 | `src/train_s2.py --tag=_fine` | Stage-2 LightGBM re-scorer (context + fine-grained features), 5-fold CV, threshold | ~10 min |
| 5 | `src/predict.py --tag=_fine` | Test inference; writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` | ~90 min |

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
| `src/train_s2.py` | Stage-2 training, CV, and choosing the final configuration (`work/final_fine.json`) |
| `src/predict.py` | Test inference and output writing |

## Decision rule

Each Source 2/3 record belongs to at most one Source 1 entity (this holds for every record in the training labels).
Each record is therefore assigned to its single highest-scoring Source 1 candidate, but only if the stage-2 probability is at least the tuned threshold.
A Source 1 entity's matches are all the records assigned to it. Entities with no assigned records are output as singletons (an empty list).
