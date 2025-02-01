#!/usr/bin/env python3
"""
AoLC: Accuracy of Lexical Choice (Paper Section 3.1).
AoLC = average over source test tokens of P_model(gold_tgt | src_word).
Gold target word is chosen via word alignment (fast_align/GIZA++) from references.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Compute AoLC given model predictions and alignments")
    ap.add_argument("--hyp", required=True, help="Model hypotheses (one per line)")
    ap.add_argument("--ref", required=True, help="References (one per line)")
    ap.add_argument("--align", required=True, help="Word alignment (fast_align format) hyp-ref")
    ap.add_argument("--src", required=True, help="Source sentences (for tokenization)")
    args = ap.parse_args()
    # Minimal stub: full implementation needs NAT model forward to get P(e|f) per word
    # and alignment to get gold e for each f. See paper Section 3.1.
    print("AoLC requires running model forward to get P(gold|src) per aligned pair. Use fairseq + alignment.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
