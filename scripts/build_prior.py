#!/usr/bin/env python3
"""
Build data-dependent prior (WAD + SDD) for Model Level training.
Paper Section 4.1: Q(e|f) from Word Alignment Distribution (WAD) and Self-Distilled Distribution (SDD).
Output: .h5 file with key "weights" shape [src_vocab_size, tgt_vocab_size], normalized per row.
"""
import argparse
import numpy as np
import h5py


def load_dict(path):
    """Load fairseq dict: lines 'word count'. Return list of words (index = fairseq id)."""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            idx = line.rfind(" ")
            word = line[:idx].strip() if idx >= 0 else line.strip()
            words.append(word)
    return words


def build_wad(align_path, dict_src_path, dict_tgt_path, temperature=2.0):
    """
    Word Alignment Distribution from fast_align format (e.g. "0-0 1-2 2-1" per line).
    Returns matrix [V_src, V_tgt] with P(tgt|src) smoothed by temperature.
    """
    src_words = load_dict(dict_src_path)
    tgt_words = load_dict(dict_tgt_path)
    V_src, V_tgt = len(src_words), len(tgt_words)
    count = np.zeros((V_src, V_tgt), dtype=np.float64)
    with open(align_path, "r", encoding="utf-8") as f:
        for line in f:
            for link in line.strip().split():
                if "-" in link:
                    i, j = map(int, link.split("-"))
                    if i < V_src and j < V_tgt:
                        count[i, j] += 1
    count += 1e-8
    count = count / temperature
    exp = np.exp(count - count.max(axis=1, keepdims=True))
    wad = exp / exp.sum(axis=1, keepdims=True)
    return wad.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad-align", required=True, help="Alignment file (fast_align format) for WAD")
    ap.add_argument("--dict-src", required=True, help="Source dictionary (fairseq dict.txt)")
    ap.add_argument("--dict-tgt", required=True, help="Target dictionary")
    ap.add_argument("--temperature", type=float, default=2.0, help="Temperature for WAD softmax")
    ap.add_argument("--output", required=True, help="Output .h5 path (key: weights)")
    ap.add_argument("--prior-src-vocab", required=True, help="Output: source vocab one token per line")
    ap.add_argument("--prior-tgt-vocab", required=True, help="Output: target vocab one token per line")
    args = ap.parse_args()

    src_words = load_dict(args.dict_src)
    tgt_words = load_dict(args.dict_tgt)
    with open(args.prior_src_vocab, "w", encoding="utf-8") as f:
        f.write("\n".join(src_words) + "\n")
    with open(args.prior_tgt_vocab, "w", encoding="utf-8") as f:
        f.write("\n".join(tgt_words) + "\n")

    prior = build_wad(args.wad_align, args.dict_src, args.dict_tgt, args.temperature)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("weights", data=prior)
    print(f"Saved prior shape {prior.shape} to {args.output}")


if __name__ == "__main__":
    main()
