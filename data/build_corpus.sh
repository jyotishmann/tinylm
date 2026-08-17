#!/usr/bin/env bash
# data/build_corpus.sh
# Full data pipeline: download → preprocess → verify
# Usage: bash data/build_corpus.sh

set -e  # Exit immediately on any error

echo "════════════════════════════════════"
echo "  TinyLM — Corpus Build Pipeline"
echo "  Dataset: H.P. Lovecraft (PD)"
echo "════════════════════════════════════"

echo ""
echo "Step 1/3: Download from Project Gutenberg"
PYTHONPATH=. python data/download.py

echo ""
echo "Step 2/3: Preprocess & clean"
python data/preprocess.py

echo ""
echo "Step 3/3: Verify corpus integrity"
python data/verify.py

echo ""
echo "════════════════════════════════════"
echo "  Corpus ready at data/processed/corpus.txt"
echo "  Next: python -m tinylm.tokenizer.train"
echo "════════════════════════════════════"