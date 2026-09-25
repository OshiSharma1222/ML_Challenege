#!/usr/bin/env bash
# End-to-end reproduction: raw TSVs -> blocking -> matching -> output/*.tsv
# Set ER_DATA_DIR to the folder that contains train/ and test/ (default in src/config.py).
# Stages must run one after another: the machine used had 16 GB RAM.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

$PY prep.py train test            # ~5 min   normalise + transliterate all records
$PY blocking.py train             # ~17 min  candidate generation (train)
$PY blocking.py test              # ~17 min  candidate generation (test)
$PY train.py 500000               # ~25 min  stage-1 LightGBM + validation scores
$PY extend_val.py 0.08            # ~20 min  stage-1 scores for 8% more held-out entities
$PY train_s2.py --ext --tag=_ext  # ~10 min  stage-2 re-scorer (CV) + decision rule
$PY predict.py --tag=_ext         # ~2.5 h   test inference, writes output/
