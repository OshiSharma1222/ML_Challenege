#!/usr/bin/env bash
# End-to-end reproduction: raw TSVs -> blocking -> matching -> output/*.tsv
# Set ER_DATA_DIR to the folder that contains train/ and test/ (default in src/config.py).
# Stages must run one after another: the machine used had 16 GB RAM. Threads are capped
# at 10 (ER_JOBS / POLARS_MAX_THREADS); times below are for 10 threads.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export ER_JOBS=${ER_JOBS:-16} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-16} OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

$PY prep.py train test                                # ~5 min   normalise + transliterate all records
$PY blocking.py train                                 # ~18 min  candidate generation (train)
$PY blocking.py test                                  # ~15 min  candidate generation (test)
$PY train.py 500000                                   # ~22 min  stage-1 LightGBM + validation scores
$PY extend_val.py 0.08                                # ~20 min  stage-1 scores, 2nd held-out entity set
$PY extend_val.py 0.10 --out=val3                     # ~22 min  stage-1 scores, 3rd held-out entity set
$PY predict.py --s1-only                              # ~2.5 h   stage-1 scores for all test pairs
$PY neighbors.py train --score                        # ~35 min  record graph on the held-out sets
$PY xwords.py                                         # ~1 min   fingerprint tables (extra-word match rates)
$PY train_s2.py --ext=3 --noent --big --nb --tag=_nb4 # ~50 min  stage-2 re-scorer (CV) + decision rule
$PY neighbors.py test --score                         # ~50 min  record graph on test
$PY predict.py --tag=_nb4 --nb                        # ~40 min  test inference, writes output/ (--stream: low memory)
