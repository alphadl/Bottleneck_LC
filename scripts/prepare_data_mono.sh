#!/bin/bash
# Monolingual Data Level: prepare →KDM and ←→KDM data (Section 4.3).
# →KDM: (mono_src_sentences, AT_teacher(mono_src_sentences))
# ←KDM: (AT_backward(mono_tgt_sentences), mono_tgt_sentences)
# Concatenate →KDM + ←KDM for ←→KDM. Use same dict as bilingual.
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA_DIR="${DATA_DIR:-./data/${SRC}${TGT}}"
MONO_SRC="${MONO_SRC:-$DATA_DIR/mono.$SRC}"   # Monolingual source-side data
MONO_TGT="${MONO_TGT:-$DATA_DIR/mono.$TGT}"   # Monolingual target-side data
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$DATA_DIR"
# User must create:
# - mono.$SRC, mono.$TGT (same BPE as train)
# - mono_kd_forward.$TGT = AT(mono.$SRC)  (forward KD on mono src)
# - mono_kd_backward.$SRC = AT_rev(mono.$TGT)  (backward KD on mono tgt)
if [ ! -f "$MONO_SRC" ] || [ ! -f "$MONO_TGT" ]; then
  echo "Place mono.$SRC and mono.$TGT in $DATA_DIR (BPE applied)."
  exit 1
fi
# →KDM: source = mono_src, target = AT(mono_src)
mkdir -p "$DATA_DIR/mono_forward_KD"
cp "$MONO_SRC" "$DATA_DIR/mono_forward_KD/train.$SRC"
if [ -f "$DATA_DIR/mono_kd_forward.$TGT" ]; then
  cp "$DATA_DIR/mono_kd_forward.$TGT" "$DATA_DIR/mono_forward_KD/train.$TGT"
else
  echo "Generate mono_kd_forward.$TGT = AT(mono.$SRC) and re-run."
  cp "$MONO_TGT" "$DATA_DIR/mono_forward_KD/train.$TGT"
fi
# ←KDM: source = AT_rev(mono_tgt), target = mono_tgt
mkdir -p "$DATA_DIR/mono_backward_KD"
cp "$MONO_TGT" "$DATA_DIR/mono_backward_KD/train.$TGT"
if [ -f "$DATA_DIR/mono_kd_backward.$SRC" ]; then
  cp "$DATA_DIR/mono_kd_backward.$SRC" "$DATA_DIR/mono_backward_KD/train.$SRC"
else
  echo "Generate mono_kd_backward.$SRC = AT_rev(mono.$TGT) and re-run."
  cp "$MONO_SRC" "$DATA_DIR/mono_backward_KD/train.$SRC"
fi
# ←→KDM: concat and dedup
mkdir -p "$DATA_DIR/mono_bidirectional_KD"
cat "$DATA_DIR/mono_forward_KD/train.$SRC" "$DATA_DIR/mono_backward_KD/train.$SRC" > "$DATA_DIR/mono_bidirectional_KD/train_mix.$SRC"
cat "$DATA_DIR/mono_forward_KD/train.$TGT" "$DATA_DIR/mono_backward_KD/train.$TGT" > "$DATA_DIR/mono_bidirectional_KD/train_mix.$TGT"
paste "$DATA_DIR/mono_bidirectional_KD/train_mix.$SRC" "$DATA_DIR/mono_bidirectional_KD/train_mix.$TGT" | awk '!x[$0]++' > "$DATA_DIR/mono_bidirectional_KD/train_mix.dedup"
cut -f1 "$DATA_DIR/mono_bidirectional_KD/train_mix.dedup" > "$DATA_DIR/mono_bidirectional_KD/train_mix.dedup.$SRC"
cut -f2 "$DATA_DIR/mono_bidirectional_KD/train_mix.dedup" > "$DATA_DIR/mono_bidirectional_KD/train_mix.dedup.$TGT"
# Binarize with dict from bilingual
for name in mono_forward_KD mono_backward_KD mono_bidirectional_KD; do
  dest="$DATA_DIR/databin/$name"
  mkdir -p "$dest"
  pref="$DATA_DIR/$name/train"
  [ "$name" = "mono_bidirectional_KD" ] && pref="$DATA_DIR/$name/train_mix.dedup"
  if [ ! -f "$dest/train.$SRC-$TGT.$SRC.bin" ]; then
    python "$ROOT/fairseq/fairseq_cli/preprocess.py" \
      --source-lang $SRC --target-lang $TGT \
      --trainpref "$pref" --validpref "$DATA_DIR/valid" --testpref "$DATA_DIR/test" \
      --srcdict "$DATA_DIR/databin/raw_PT/dict.$SRC.txt" --tgtdict "$DATA_DIR/databin/raw_PT/dict.$TGT.txt" \
      --destdir "$dest" --workers 8
  fi
done
echo "Monolingual KD data ready in $DATA_DIR/databin/mono_*"
