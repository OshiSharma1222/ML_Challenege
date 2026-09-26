#!/usr/bin/env bash
# End-to-end reproduction: raw TSVs -> blocking -> matching -> output/*.tsv
# Set ER_DATA_DIR to the folder that contains train/ and test/ (default in src/config.py).
# Stages must run one after another: the machine used had 16 GB RAM. Threads are capped
# at 10 (ER_JOBS / POLARS_MAX_THREADS); times below are for 10 threads.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export ER_JOBS=${ER_JOBS:-10} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-10} OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}

$PY prep.py train test                                # ~5 min   normalise + transliterate all records
$PY blocking.py train                                 # ~18 min  candidate generation (train)
$PY blocking.py test                                  # ~15 min  candidate generation (test)
$PY train.py 500000                                   # ~22 min  stage-1 LightGBM + validation scores
$PY extend_val.py 0.08                                # ~20 min  stage-1 scores, 2nd held-out entity set
$PY extend_val.py 0.10 --out=val3                     # ~22 min  stage-1 scores, 3rd held-out entity set
$PY train_s2.py --ext=3 --noent --tag=_noent3         # ~23 min  stage-2 re-scorer (CV) + decision rule
$PY predict.py --tag=_noent3                          # ~3 h     test inference, writes output/
