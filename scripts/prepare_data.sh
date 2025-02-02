#!/bin/bash
# Data preparation for Bottleneck_LC (WMT14 En-De example).
# Produces: train_raw, train_kd (forward KD), train_bt (reverse KD), bidirectional_KD, binarized dirs.
# Requires: raw parallel data (train.en, train.de), valid/test, and optionally AT teacher for KD.
set -e
DATA_DIR="${DATA_DIR:-./data}"
SRC="${SRC:-en}"
TGT="${TGT:-de}"
PAIR="${SRC}${TGT}"
BPE_SIZE="${BPE_SIZE:-32000}"
WORKERS="${WORKERS:-8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-8}"

mkdir -p "$DATA_DIR/$PAIR"
cd "$DATA_DIR/$PAIR"

# 1) If you have raw text: train.$SRC, train.$TGT, valid.$SRC, valid.$TGT, test.$SRC, test.$TGT
#    Place them in $DATA_DIR/$PAIR/ or set paths below.
TRAIN_RAW_SRC="${TRAIN_RAW_SRC:-$DATA_DIR/$PAIR/train.$SRC}"
TRAIN_RAW_TGT="${TRAIN_RAW_TGT:-$DATA_DIR/$PAIR/train.$TGT}"
VALID_SRC="${VALID_SRC:-$DATA_DIR/$PAIR/valid.$SRC}"
VALID_TGT="${VALID_TGT:-$DATA_DIR/$PAIR/valid.$TGT}"
TEST_SRC="${TEST_SRC:-$DATA_DIR/$PAIR/test.$SRC}"
TEST_TGT="${TEST_TGT:-$DATA_DIR/$PAIR/test.$TGT}"

if [ ! -f "$TRAIN_RAW_SRC" ] || [ ! -f "$TRAIN_RAW_TGT" ]; then
  echo "Place raw train.$SRC / train.$TGT in $DATA_DIR/$PAIR/ (or set TRAIN_RAW_SRC/TGT)."
  echo "For WMT14 En-De, download from statmt.org and preprocess (tokenize, BPE)."
  exit 1
fi

# 2) Learn BPE (if not done)
if [ ! -f bpe.codes ]; then
  echo "Learning BPE..."
  cat "$TRAIN_RAW_SRC" "$TRAIN_RAW_TGT" | subword-nmt learn-bpe -s $BPE_SIZE -o bpe.codes
fi
for split in train valid test; do
  for lang in $SRC $TGT; do
    pref="${!split}_RAW_$lang"
    [ -z "${!pref}" ] && pref="${split}.$lang"
    f="${!pref:-$DATA_DIR/$PAIR/$split.$lang}"
    [ -f "$f" ] || continue
    if [ ! -f "${split}.bpe.$lang" ]; then
      subword-nmt apply-bpe -c bpe.codes < "$f" > "${split}.bpe.$lang"
    fi
  done
done
[ -f train.bpe.$SRC ] || { subword-nmt apply-bpe -c bpe.codes < "$TRAIN_RAW_SRC" > train.bpe.$SRC; subword-nmt apply-bpe -c bpe.codes < "$TRAIN_RAW_TGT" > train.bpe.$TGT; }
[ -f valid.bpe.$SRC ] || { subword-nmt apply-bpe -c bpe.codes < "$VALID_SRC" > valid.bpe.$SRC; subword-nmt apply-bpe -c bpe.codes < "$VALID_TGT" > valid.bpe.$TGT; }
[ -f test.bpe.$SRC ] || { subword-nmt apply-bpe -c bpe.codes < "$TEST_SRC" > test.bpe.$SRC; subword-nmt apply-bpe -c bpe.codes < "$TEST_TGT" > test.bpe.$TGT; }

# 3) Raw data (for raw pretrain and prior)
cp train.bpe.$SRC train_raw.$SRC
cp train.bpe.$TGT train_raw.$TGT
cp valid.bpe.$SRC valid.$SRC
cp valid.bpe.$TGT valid.$TGT
cp test.bpe.$SRC test.$SRC
cp test.bpe.$TGT test.$TGT

# 4) Forward KD: train_kd.$SRC = train_raw.$SRC, train_kd.$TGT = AT teacher output on train_raw.$SRC
#    You must generate train_kd.$TGT with your AT model. Place as train_kd.$TGT or set TRAIN_KD_TGT.
if [ -f train_kd.$TGT ]; then
  cp train.bpe.$SRC train_kd.$SRC
else
  echo "Create train_kd.$TGT (AT teacher translations of train_raw.$SRC) and re-run."
  cp train.bpe.$SRC train_kd.$SRC
  cp train.bpe.$TGT train_kd.$TGT
fi

# 5) Reverse KD: train_bt.$TGT = train_raw.$TGT, train_bt.$SRC = backward AT output on train_raw.$TGT
if [ -f train_bt.$SRC ]; then
  cp train.bpe.$TGT train_bt.$TGT
else
  cp train.bpe.$TGT train_bt.$TGT
  cp train.bpe.$SRC train_bt.$SRC
fi

# 6) Bidirectional KD: concat train_kd + train_bt, dedup
mkdir -p bidirectional_KD
cat train_kd.$SRC train_bt.$SRC > bidirectional_KD/train_mix.$SRC
cat train_kd.$TGT train_bt.$TGT > bidirectional_KD/train_mix.$TGT
paste bidirectional_KD/train_mix.$SRC bidirectional_KD/train_mix.$TGT | awk '!x[$0]++' > bidirectional_KD/train_mix.dedup
cut -f1 bidirectional_KD/train_mix.dedup > bidirectional_KD/train_mix.dedup.$SRC
cut -f2 bidirectional_KD/train_mix.dedup > bidirectional_KD/train_mix.dedup.$TGT

# 7) Binarize (fairseq preprocess): raw_PT first to get dict, then others re-use dict
mkdir -p databin
if [ ! -d databin/raw_PT ]; then
  python "$ROOT/fairseq/fairseq_cli/preprocess.py" \
    --source-lang $SRC --target-lang $TGT \
    --trainpref train_raw --validpref valid --testpref test \
    --destdir databin/raw_PT --workers $WORKERS --joined-dictionary
fi
for name in forward_KD reversed_KD; do
  pref="train_kd"; [ "$name" = "reversed_KD" ] && pref="train_bt"
  dest="databin/$name"
  mkdir -p "$dest"
  if [ ! -f "$dest/train.$SRC-$TGT.$SRC.bin" ]; then
    python "$ROOT/fairseq/fairseq_cli/preprocess.py" \
      --source-lang $SRC --target-lang $TGT \
      --trainpref "$pref" --validpref valid --testpref test \
      --srcdict databin/raw_PT/dict.$SRC.txt --tgtdict databin/raw_PT/dict.$TGT.txt \
      --destdir "$dest" --workers $WORKERS
  fi
done
dest="databin/bidirectional_KD"
mkdir -p "$dest"
if [ ! -f "$dest/train.$SRC-$TGT.$SRC.bin" ]; then
  python "$ROOT/fairseq/fairseq_cli/preprocess.py" \
    --source-lang $SRC --target-lang $TGT \
    --trainpref bidirectional_KD/train_mix.dedup --validpref valid --testpref test \
    --srcdict databin/raw_PT/dict.$SRC.txt --tgtdict databin/raw_PT/dict.$TGT.txt \
    --destdir "$dest" --workers $WORKERS
fi

echo "Data ready in $DATA_DIR/$PAIR/databin/"
