#!/bin/bash
# Full reproduction pipeline for Bottleneck_LC (WMT14 En-De example).
# Prerequisites: Clone RLFW-NAT (or set FAIRSEQ_SRC), prepare WMT14 data in data/ende/, train AT teacher for KD.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export SRC=en TGT=de

# 0) Setup fairseq (copy from RLFW-NAT + our criterion)
bash scripts/setup_fairseq.sh

# 1) Data: place train.en, train.de, valid.*, test.* in data/ende/ then run:
#    DATA_DIR=$ROOT/data SRC=en TGT=de bash scripts/prepare_data.sh
#    Generate train_kd.de with AT teacher (forward KD) and train_bt.en (backward AT) if needed.

# 2) Build prior (WAD from alignment)
#    Run fast_align on train_raw to get alignments, then:
#    python scripts/build_prior.py --wad-align data/ende/train.align --dict-src data/ende/databin/raw_PT/dict.en.txt \
#      --dict-tgt data/ende/databin/raw_PT/dict.de.txt --output data/ende/prior.h5 \
#      --prior-src-vocab data/ende/lc.en-de.en.vocab --prior-tgt-vocab data/ende/lc.en-de.de.vocab
#    python scripts/extract_vocab.py data/ende/databin/forward_KD/dict.en.txt data/ende/lc.en-de.en.vocab
#    python scripts/extract_vocab.py data/ende/databin/forward_KD/dict.de.txt data/ende/lc.en-de.de.vocab

# 3) Training
# Baseline:
#   DATA=$ROOT/data/ende/databin/forward_KD SAVE=$ROOT/checkpoints/ende/baseline bash scripts/train_baseline.sh
# Model Level (+ prior):
#   PRIOR_WEIGHT=$ROOT/data/ende/prior.h5 ... bash scripts/train_model_level.sh
# Parallel Data Level (Raw -> Bidir KD -> Forward KD):
#   bash scripts/train_parallel_level.sh

# 4) Eval
#   SAVE=$ROOT/checkpoints/ende/baseline DATA=... TEST_PREF=$ROOT/data/ende/test bash scripts/eval_bleu.sh

echo "Edit this script to point DATA_DIR and run each step. See README.md for full instructions."
