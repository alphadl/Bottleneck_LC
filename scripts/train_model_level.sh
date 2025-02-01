#!/bin/bash
# Model Level: NAT + data-dependent prior (WAD). Requires prior .h5 and vocab files.
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:-./data/${SRC}${TGT}/databin/forward_KD}"
SAVE="${SAVE:-./checkpoints/${SRC}${TGT}/model_level_mask}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-./data/${SRC}${TGT}/prior.h5}"
PRIOR_SRC_VOCAB="${PRIOR_SRC_VOCAB:-./data/${SRC}${TGT}/databin/forward_KD/lc.${SRC}-${TGT}.${SRC}.vocab}"
PRIOR_TGT_VOCAB="${PRIOR_TGT_VOCAB:-./data/${SRC}${TGT}/databin/forward_KD/lc.${SRC}-${TGT}.${TGT}.vocab}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Extract vocab from dict if lc.*.vocab not present
DATABIN="${DATA}"
[ ! -f "$PRIOR_SRC_VOCAB" ] && python -c "
import sys
with open('$DATABIN/dict.$SRC.txt') as f:
    for line in f:
        idx = line.rfind(' ')
        print(line[:idx].strip() if idx>=0 else line.strip())
" > "$PRIOR_SRC_VOCAB" 2>/dev/null || true
[ ! -f "$PRIOR_TGT_VOCAB" ] && python -c "
with open('$DATABIN/dict.$TGT.txt') as f:
    for line in f:
        idx = line.rfind(' ')
        print(line[:idx].strip() if idx>=0 else line.strip())
" > "$PRIOR_TGT_VOCAB" 2>/dev/null || true
mkdir -p "$SAVE"
python "$ROOT/fairseq/fairseq_cli/train.py" "$DATA" \
  --save-dir "$SAVE" -s $SRC -t $TGT \
  --task translation_lev --criterion bottleneck_lc_loss --arch cmlm_transformer \
  --prior-weight-path "$PRIOR_WEIGHT" --prior-src-vocab-path "$PRIOR_SRC_VOCAB" --prior-tgt-vocab-path "$PRIOR_TGT_VOCAB" \
  --prior-total-steps 300000 --label-smoothing 0.1 --dropout 0.2 --noise random_mask \
  --share-decoder-input-output-embed --decoder-learned-pos --encoder-learned-pos --apply-bert-init \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-7 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-7 --warmup-updates 10000 --lr-period-updates 290000 \
  --max-update 300000 --weight-decay 0.01 --clip-norm 0.1 \
  --max-tokens 8192 --update-freq 4 --fp16 \
  --save-interval-updates 2000 --keep-last-epochs 5 --seed 1
