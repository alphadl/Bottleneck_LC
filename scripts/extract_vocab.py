#!/usr/bin/env python3
"""Extract one token per line from fairseq dict (for prior vocab)."""
import sys
with open(sys.argv[1], "r", encoding="utf-8") as fin:
    with open(sys.argv[2], "w", encoding="utf-8") as fout:
        for line in fin:
            idx = line.rfind(" ")
            word = line[:idx].strip() if idx >= 0 else line.strip()
            fout.write(word + "\n")
