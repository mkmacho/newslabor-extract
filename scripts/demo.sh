#!/usr/bin/env bash
# End-to-end demonstration on the bundled sample. No API key, no network.
#
# Runs the whole pipeline over 2,000 synthetic advertisements:
#   extract   raw OCR text        -> candidate addresses + wages
#   recompute candidate addresses -> counties and FIPS codes, offline
#   validate  extract output      -> a stratified accuracy-coding sample
#
# The geocoding stage (resolve.py) is deliberately not run: it needs a paid
# API key, and the county derivation below does not.
#
# Usage:  bash scripts/demo.sh [OUTPUT_DIR]

set -euo pipefail

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve the output path against the CALLER's directory before changing to the
# repo root, so `bash /path/to/scripts/demo.sh out` puts `out` where the caller
# expects rather than inside the checkout.
OUT="${1:-$HERE/demo-output}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
cd "$HERE"

if ! "$PY" -c 'import pandas, spacy, symspellpy, thefuzz, pyzipcode' 2>/dev/null; then
  echo "error: dependencies missing. Create the environment first:" >&2
  echo "  python3.11 -m venv .venv && source .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  echo "(Python 3.10-3.11; pandas 1.5.3 has no wheels above 3.11.)" >&2
  exit 1
fi

mkdir -p "$OUT"
echo
echo "=============================================================="
echo " newslabor-extract demo - 2,000 ads, no API key needed"
echo "=============================================================="

echo
echo "[1/4] Extracting addresses and wages from raw OCR text..."
"$PY" scripts/extract.py \
  --filepath=./test_data/NJG-sample.csv \
  --aux_dir=./auxiliary_files \
  --output_dir="$OUT" \
  --extract_wage=1 \
  --write_csv=1 \
  | sed 's/^/      /'

echo
echo "[2/4] Deriving counties offline - no geocoding requests..."
"$PY" scripts/recompute.py --from_addresses \
  --filepath="$OUT/NJG-extract-all.gzip" \
  --aux_dir=./auxiliary_files \
  --output_dir="$OUT" \
  | sed 's/^/      /'

echo
echo "[3/4] Drawing an accuracy-validation sample..."
"$PY" scripts/validate.py sample \
  --filepath="$OUT/NJG-extract-all.gzip" \
  --n=200 --out="$OUT/validation-sample.csv" \
  | sed 's/^/      /'

echo
echo "[4/4] Summary"
"$PY" - "$OUT" <<'PYEOF' | sed 's/^/      /'
import sys, pandas as pd
out = sys.argv[1]
df = pd.read_parquet(f"{out}/NJG-extract-all.gzip")
co = pd.read_parquet(f"{out}/NJG-offline-county-all.gzip")
n = len(df)
has_addr = (df.addresses.str.len() > 0).sum()
cands = int(df.addresses.str.len().sum())
has_wage = int(df.wage.notna().sum())
county = int(co.offline_zip_county.notna().sum())
fips = int(co.offline_zip_county_fips.notna().sum())
print(f"ads processed                    {n:>8,}")
print(f"ads with >=1 candidate address   {has_addr:>8,}  ({has_addr/n:.1%})")
print(f"candidate addresses              {cands:>8,}")
print(f"ads with a wage                  {has_wage:>8,}  ({has_wage/n:.1%})")
print(f"ads with a county, zero API calls{county:>8,}  ({county/n:.1%})")
print(f"  ...carrying a joinable FIPS    {fips:>8,}")
if 'wage_period' in df.columns:
    per = df.wage_period.dropna().value_counts().to_dict()
    print(f"wage periods parsed              {per}")

# These labels exist only because this fixture is generated. They are useful as
# a deterministic regression diagnostic, not as evidence about historical ads.
for label, predicted, truth_col in (
        ('address', df.addresses.apply(len).gt(0), '_truth_has_address'),
        ('wage', df.wage.notna(), '_truth_has_wage')):
    truth = df[truth_col].astype(bool)
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else float('nan')
    recall = tp / (tp + fn) if tp + fn else float('nan')
    print(f"synthetic {label:7s} P/R           {precision:>7.2%} / {recall:.2%}"
          f"  (TP={tp}, FP={fp}, FN={fn})")
PYEOF

echo
echo "Done. Outputs in $OUT/"
echo "Next: open $OUT/validation-sample.csv and its CODEBOOK to score accuracy,"
echo "      then: $PY scripts/validate.py score --filepath=$OUT/validation-sample.csv"
echo
