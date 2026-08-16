"""Measure how accurate the extractors actually are.

Nothing in this pipeline has ever been scored. Coverage statistics answer "how
often did we output something", which is not the same question as "how often was
it right", and they cannot see a false negative at all. This script closes that
gap in two steps.

  sample  draw a stratified sample and write a coding template for a human
  score   read the completed template and report precision, recall and coverage

The sampling design matters more than the code. Two properties are deliberate:

1. Ads the pipeline left EMPTY are included. A sample drawn only from ads that
   produced an address can measure precision but never recall — it cannot
   contain a miss. Roughly half of each stratum is drawn from the empty side.
2. Strata are decades. Coverage on the bundled NJG sample runs from 35% in the
   1960s to 71% in the 2000s, so OCR quality is correlated with the time
   dimension the research compares along; a sample that ignores that would give
   one blended number hiding a trend.

Scoring reports Wilson intervals rather than bare point estimates, because at
n=200 the interval is wide enough that quoting a single number would overstate
what the exercise establishes.

Usage:
    python scripts/validate.py sample --filepath=<EXTRACT_OUTPUT.gzip> \
        --n=200 --out=validation/njg-sample.csv

    # ... a human fills in the truth_* columns ...

    python scripts/validate.py score --filepath=validation/njg-sample.csv
"""
import argparse
import math
import os
import textwrap

import pandas as pd

from common import newspaper_from_path

# Columns the coder fills in. Kept few and concrete: every extra field is a
# judgement call the coder has to make consistently 200 times.
CODING_COLUMNS = {
    'truth_has_address': 'y / n  — does the ad state a street address or a city?',
    'truth_address': '     free text — the address as printed, if any',
    'truth_is_job_ad': 'y / n  — is this a job advertisement at all?',
    'truth_address_is_worksite': 'y / n / unclear — worksite, or an application address?',
    'truth_has_wage': 'y / n  — does the ad state pay?',
    'truth_wage': '     free text — the wage as printed, if any',
    'pred_address_correct': 'y / n / partial — judge AFTER filling the truth columns',
    'pred_wage_correct': 'y / n / partial — judge AFTER filling the truth columns',
    'coder_notes': '     anything ambiguous',
}


def decade(year):
    try:
        return int(year) // 10 * 10
    except (TypeError, ValueError):
        return None


def draw_sample(df, n, seed):
    ''' Stratify by decade, and within each decade split between ads the
    pipeline found an address for and ads it did not. '''
    df = df.copy()
    df['_decade'] = df['year'].apply(decade) if 'year' in df.columns else None
    df['_found'] = df.addresses.apply(lambda a: a is not None and len(a) > 0)

    strata = [d for d in sorted(df['_decade'].dropna().unique())] or [None]
    per_stratum = max(2, n // (len(strata) * 2))
    chunks = []
    for d in strata:
        pool = df[df['_decade'] == d] if d is not None else df
        for found in (True, False):
            side = pool[pool['_found'] == found]
            if side.empty:
                continue
            take = min(per_stratum, len(side))
            chunks.append(side.sample(take, random_state=seed))
    sample = pd.concat(chunks).sample(frac=1, random_state=seed)  # shuffle
    return sample.head(n)


def write_template(sample, out_path, newspaper):
    cols = {}
    cols['sample_id'] = range(len(sample))
    if 'id' in sample.columns:
        cols['ad_id'] = sample['id'].values
    if 'year' in sample.columns:
        cols['year'] = sample['year'].values
    # The coder must read the ad, so give them the text, trimmed to the first ad.
    cols['ad_text'] = [textwrap.shorten(str(t), width=1200, placeholder=' …')
                       for t in sample.raw_content]
    cols['pred_addresses'] = [
        ' | '.join(sorted({_fmt_addr(a) for a in (arr if arr is not None else [])}))
        for arr in sample.addresses]
    cols['pred_wage'] = (sample['wage'].values if 'wage' in sample.columns
                         else [None] * len(sample))
    out = pd.DataFrame(cols)
    for c in CODING_COLUMNS:
        out[c] = ''
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)

    key_path = os.path.splitext(out_path)[0] + '-CODEBOOK.md'
    with open(key_path, 'w') as fh:
        fh.write(_codebook(newspaper, len(out)))
    return key_path


def _fmt_addr(a):
    if not isinstance(a, dict):
        return str(a)
    parts = [a.get('housenumber'), a.get('street'), a.get('city'),
             a.get('state'), a.get('zipcode')]
    return ' '.join(str(p) for p in parts if p) or '(empty)'


def _codebook(newspaper, n):
    lines = [
        '# Coding instructions — {} validation sample (n={})'.format(newspaper, n),
        '',
        'Fill in the `truth_*` columns from the ad text ALONE. Only after that,',
        'compare against `pred_*` and fill the two judgement columns. Coding the',
        'truth first is what keeps the prediction from anchoring the judgement.',
        '',
        '## Columns you fill',
        '',
    ]
    for col, desc in CODING_COLUMNS.items():
        lines.append('- `{}` — {}'.format(col, desc.strip()))
    lines += [
        '',
        '## Conventions',
        '',
        '- **Address** means a street address OR a bare city/neighbourhood that',
        '  locates the job. "Apply 509 Granby St" and "Norfolk" both count;',
        '  "apply in person" does not.',
        '- **Worksite vs application address.** Many ads give a personnel office,',
        '  a newspaper box number, or an agency. Code those `n`. This distinction',
        '  is the one most likely to matter for the research and the one the',
        '  pipeline cannot make at all, so it is worth coding carefully.',
        '- **partial** for `pred_*_correct` means the right place or amount but',
        '  mangled — wrong street spelling, missing the rate, truncated number.',
        '- OCR is often unreadable. If you cannot tell, use `unclear` in',
        '  `truth_address_is_worksite` or leave a note; do not guess.',
        '- Non-job ads (real estate, personals, notices) are in the sample on',
        '  purpose: the pipeline is supposed to skip them, and coding',
        '  `truth_is_job_ad = n` is how its filter gets scored.',
        '',
        '## What the numbers will mean',
        '',
        '- **Precision** = of the ads where the pipeline emitted an address, how',
        '  many were right. Measurable only on the non-empty side.',
        '- **Recall** = of the ads that genuinely contain an address, how many the',
        '  pipeline found. Measurable ONLY because empty ads are in the sample.',
        '- Both are reported per decade, since coverage varies strongly over time.',
        '',
        'Run `python scripts/validate.py score --filepath=<this csv>` when done.',
    ]
    return '\n'.join(lines) + '\n'


def wilson(k, n, z=1.96):
    ''' Wilson score interval — behaves sensibly at small n and at 0/1, where
    the normal approximation does not. '''
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _yes(series):
    return series.astype(str).str.strip().str.lower().isin(['y', 'yes', 'true', '1'])


def score(df):
    ''' Report precision/recall/coverage with intervals, overall and by decade. '''
    coded = df[df.truth_has_address.astype(str).str.strip() != '']
    if coded.empty:
        raise SystemExit("No coded rows found — fill the truth_* columns first.")
    print("Scoring {} coded rows of {} in the sample.\n".format(len(coded), len(df)))

    pred_any = coded.pred_addresses.astype(str).str.strip().ne('')
    truth_any = _yes(coded.truth_has_address)
    correct = coded.pred_address_correct.astype(str).str.strip().str.lower()

    def line(label, k, n):
        p, lo, hi = wilson(k, n)
        if n == 0:
            print("  {:<34s}      n/a  (no rows)".format(label))
        else:
            print("  {:<34s} {:>6.1%}   [{:.1%}, {:.1%}]   n={}".format(
                label, p, lo, hi, n))

    print("ADDRESSES")
    n_pred = int(pred_any.sum())
    line("precision (strict)", int((correct[pred_any] == 'y').sum()), n_pred)
    line("precision (incl. partial)",
         int(correct[pred_any].isin(['y', 'partial']).sum()), n_pred)
    n_truth = int(truth_any.sum())
    line("recall", int((pred_any & truth_any).sum()), n_truth)
    line("coverage (emitted anything)", n_pred, len(coded))

    if 'truth_address_is_worksite' in coded.columns:
        ws = coded.truth_address_is_worksite.astype(str).str.strip().str.lower()
        judged = ws.isin(['y', 'n'])
        if judged.any():
            print("\nCONSTRUCT VALIDITY")
            line("addresses that are the worksite",
                 int((ws == 'y').sum()), int(judged.sum()))
            print("       (the rest are application addresses — the pipeline")
            print("        cannot distinguish these, so this bounds what the")
            print("        geography variable can mean)")

    if 'truth_has_wage' in coded.columns and coded.pred_wage.notna().any():
        print("\nWAGES")
        pw = coded.pred_wage.astype(str).str.strip().ne('') & coded.pred_wage.notna()
        tw = _yes(coded.truth_has_wage)
        wc = coded.pred_wage_correct.astype(str).str.strip().str.lower()
        line("precision (strict)", int((wc[pw] == 'y').sum()), int(pw.sum()))
        line("recall", int((pw & tw).sum()), int(tw.sum()))

    if 'truth_is_job_ad' in coded.columns:
        job = _yes(coded.truth_is_job_ad)
        if (~job).any():
            print("\nAD FILTER")
            line("non-job ads that produced an address",
                 int((pred_any & ~job).sum()), int((~job).sum()))

    if 'year' in coded.columns:
        print("\nBY DECADE (recall — the number coverage statistics cannot show)")
        for d in sorted({decade(y) for y in coded.year} - {None}):
            rows = coded[coded.year.apply(decade) == d]
            t = _yes(rows.truth_has_address)
            p_ = rows.pred_addresses.astype(str).str.strip().ne('')
            line("  {}s".format(d), int((p_ & t).sum()), int(t.sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['sample', 'score'])
    parser.add_argument('--filepath', required=True,
        help="Extract output (.gzip) for `sample`; the coded CSV for `score`.")
    parser.add_argument('-n', '--n', type=int, default=200,
        help="Target sample size (default 200).")
    parser.add_argument('--seed', type=int, default=20260816,
        help="Sampling seed; record it in the appendix for reproducibility.")
    parser.add_argument('--out', type=str, default='validation/sample.csv')
    args = parser.parse_args()

    assert os.path.isfile(args.filepath), 'Invalid filepath.'

    if args.mode == 'sample':
        df = pd.read_parquet(args.filepath)
        assert 'addresses' in df.columns, "Needs an extract output with `addresses`."
        sample = draw_sample(df, args.n, args.seed)
        key = write_template(sample, args.out, newspaper_from_path(args.filepath))
        found = sample.addresses.apply(lambda a: a is not None and len(a) > 0)
        print("Wrote {} rows to {} (seed {}).".format(len(sample), args.out, args.seed))
        print("  {} ads where the pipeline found an address, {} where it did not"
              " — the empty half is what makes recall estimable.".format(
                  int(found.sum()), int((~found).sum())))
        print("Coding instructions: {}".format(key))
    else:
        score(pd.read_csv(args.filepath, keep_default_na=False))
