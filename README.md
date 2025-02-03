# Bottleneck_LC

Official implementation for **"Widening The Bottleneck of Lexical Choice for Non-Autoregressive Translation"** (Computer Speech & Language).

We address lexical choice errors on low-frequency words in NAT by three approaches: (1) **Model Level** — data-dependent prior (WAD/SDD) with KL and λ decay; (2) **Parallel Data Level** — raw pretraining + bidirectional KD + forward KD (LFR); (3) **Monolingual Data Level** — bidirectional monolingual KD.

This work extends our prior conference papers: [LCNAT](https://github.com/alphadl/LCNAT) (ICLR 2021), [RLFW-NAT](https://github.com/alphadl/RLFW-NAT) (ACL 2021), [RLFW-NAT.mono](https://github.com/alphadl/RLFW-NAT.mono) (ACL 2022).

## Setup

```bash
git clone https://github.com/alphadl/Bottleneck_LC.git
cd Bottleneck_LC
pip install -r requirements.txt
```

Clone [RLFW-NAT](https://github.com/alphadl/RLFW-NAT) (or RLFW-NAT.mono) alongside this repo, then:

```bash
export FAIRSEQ_SRC=/path/to/RLFW-NAT/fairseq_mask
bash scripts/setup_fairseq.sh
```

## Data

Put `train.{src,tgt}`, `valid.{src,tgt}`, `test.{src,tgt}` in `data/<pair>/`. Generate forward KD target (AT translation of train.src) as `train_kd.tgt` and reverse KD source (backward AT of train.tgt) as `train_bt.src`. Then:

```bash
export DATA_DIR=$PWD/data SRC=en TGT=de
bash scripts/prepare_data.sh
```

## Training

- **Baseline:** `DATA=./data/ende/databin/forward_KD SAVE=./checkpoints/ende/baseline bash scripts/train_baseline.sh`
- **Model Level:** Build prior (see below), then run `scripts/train_model_level.sh` with `PRIOR_WEIGHT`, `PRIOR_SRC_VOCAB`, `PRIOR_TGT_VOCAB` set.
- **Parallel Data Level:** `bash scripts/train_parallel_level.sh`
- **Monolingual:** `scripts/prepare_data_mono.sh` then train on `databin/mono_bidirectional_KD`.

## Prior (Model Level)

1. Run word alignment (e.g. fast_align) on raw data → `train.align`.
2. `python scripts/extract_vocab.py data/ende/databin/raw_PT/dict.en.txt data/ende/lc.en-de.en.vocab` (and target).
3. `python scripts/build_prior.py --wad-align data/ende/train.align --dict-src data/ende/databin/raw_PT/dict.en.txt --dict-tgt data/ende/databin/raw_PT/dict.de.txt --output data/ende/prior.h5 --prior-src-vocab data/ende/lc.en-de.en.vocab --prior-tgt-vocab data/ende/lc.en-de.de.vocab`

## Evaluation

```bash
SAVE=./checkpoints/ende/baseline DATA=./data/ende/databin/forward_KD TEST_PREF=./data/ende/test bash scripts/eval_bleu.sh
```

## Citation

```bibtex
@article{ding2026widening,
  title={Widening The Bottleneck of Lexical Choice for Non-Autoregressive Translation},
  author={Ding, Liang and Wang, Longyue and Liu, Siyou and Luo, Weihua and Zhang, Kaifu},
  journal={Computer Speech \& Language},
  year={2026}
}
```

## License

CC-BY-NC 4.0.
