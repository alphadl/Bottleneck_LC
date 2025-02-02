#!/bin/bash
# Parallel Data Level: Raw pretrain -> Bidirectional KD -> Forward KD (LFR, Section 4.2).
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA_ROOT="./data/${SRC}${TGT}/databin"
SAVE_ROOT="./checkpoints/${SRC}${TGT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Step 1: Raw pretrain (early stop at ~90% of best raw BLEU; paper uses fixed 20k for En-De)
RAW_UPDATES="${RAW_UPDATES:-20000}"
python "$ROOT/fairseq/fairseq_cli/train.py" "$DATA_ROOT/raw_PT" \
  --save-dir "$SAVE_ROOT/raw_PT" -s $SRC -t $TGT \
  --task translation_lev --criterion nat_loss --arch cmlm_transformer \
  --label-smoothing 0.1 --dropout 0.2 --noise random_mask \
  --share-decoder-input-output-embed --decoder-learned-pos --encoder-learned-pos --apply-bert-init \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-7 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-7 --warmup-updates 10000 --max-update $RAW_UPDATES \
  --weight-decay 0 --clip-norm 0.1 --max-tokens 20000 --update-freq 3 --fp16 \
  --save-interval-updates 2000 --keep-last-epochs 1 --seed 1
# Step 2: Bidirectional KD (optionally restore from raw_PT)
BIDIR_UPDATES="${BIDIR_UPDATES:-40000}"
mkdir -p "$SAVE_ROOT/bidirectional_KD"
restore=""
[ -f "$SAVE_ROOT/raw_PT/checkpoint_best.pt" ] && restore="--restore-file $SAVE_ROOT/raw_PT/checkpoint_best.pt"
python "$ROOT/fairseq/fairseq_cli/train.py" "$DATA_ROOT/bidirectional_KD" \
  --save-dir "$SAVE_ROOT/bidirectional_KD" -s $SRC -t $TGT \
  --task translation_lev --criterion nat_loss --arch cmlm_transformer \
  --label-smoothing 0.1 --dropout 0.2 --noise random_mask \
  --share-decoder-input-output-embed --decoder-learned-pos --encoder-learned-pos --apply-bert-init \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-7 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-7 --warmup-updates 10000 --max-update $BIDIR_UPDATES \
  --weight-decay 0 --clip-norm 0.1 --max-tokens 20000 --update-freq 3 --fp16 \
  --save-interval-updates 2000 --keep-last-epochs 1 --seed 1 $restore
# Step 3: Forward KD (finetune from bidirectional)
mkdir -p "$SAVE_ROOT/parallel_level"
restore3=""
[ -f "$SAVE_ROOT/bidirectional_KD/checkpoint_best.pt" ] && restore3="--restore-file $SAVE_ROOT/bidirectional_KD/checkpoint_best.pt"
python "$ROOT/fairseq/fairseq_cli/train.py" "$DATA_ROOT/forward_KD" \
  --save-dir "$SAVE_ROOT/parallel_level" -s $SRC -t $TGT \
  --task translation_lev --criterion nat_loss --arch cmlm_transformer \
  --label-smoothing 0.1 --dropout 0.2 --noise random_mask \
  --share-decoder-input-output-embed --decoder-learned-pos --encoder-learned-pos --apply-bert-init \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-7 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-7 --warmup-updates 10000 --max-update 70000 \
  --weight-decay 0 --clip-norm 0.1 --max-tokens 20000 --update-freq 3 --fp16 \
  --save-interval-updates 2000 --keep-last-epochs 5 --seed 1 $restore3
echo "Parallel Data Level training done. Best checkpoint in $SAVE_ROOT/parallel_level/"
