#!/bin/bash
# Reproduction steps. Set DATA_DIR, FAIRSEQ_SRC, SRC, TGT as needed.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export SRC="${SRC:-en}" TGT="${TGT:-de}"

bash scripts/setup_fairseq.sh
# DATA_DIR=$ROOT/data bash scripts/prepare_data.sh
# Build prior then: bash scripts/train_model_level.sh  # or train_parallel_level.sh
# SAVE=... DATA=... TEST_PREF=... bash scripts/eval_bleu.sh
