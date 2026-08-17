"""Measure how accurate the extractors actually are.

Nothing in this pipeline has ever been scored. Coverage statistics answer "how
often did we output something", which is not the same question as "how often was
it right", and they cannot observe a false negative at all.

  sample  draw a stratified sample and write a coding template for a human
  score   read the completed template and report precision, recall and coverage

SAMPLING DESIGN

Strata are (decade x whether the pipeline emitted an address). Both halves are
deliberately over-represented relative to their share of the corpus:

- Ads the pipeline left EMPTY are drawn at ~50% of the sample even though they
  are ~44% of the corpus, because a sample drawn only from ads that produced an
  address cannot contain a false negative, and recall would be unmeasurable.
- Small decades are over-drawn relative to their size, because the 1930s hold
  203 ads and the 1990s hold 3,781, and a proportional sample would say almost
  nothing about the early period.

Over-sampling a stratum makes an estimate *identifiable*; it does not make it
*unbiased*. Every row therefore carries a design weight — the number of corpus
ads that row stands for, N_stratum / n_stratum — and every statistic below is
weighted by it. Reporting the raw sample mean instead would estimate accuracy
under an artificial 50/50 mixture: measured against a held-out label on the
bundled sample, that error runs to 11 points, and its sign flips with the
decade's coverage.

Because the rows are weighted, the usual unweighted binomial interval is not
appropriate. The intervals below use a Kish effective-sample-size Wilson
interval, n_eff = (sum w)^2 / sum(w^2). This is a working approximation, not an
exact stratified finite-population interval: it does not exploit the finite-
population correction, and it assumes any uncoded rows are ignorable. The
reproducible Monte Carlo in `verify_validation_design.py` checks its repeated-
sampling behavior under several fixed finite populations, but human-coding
nonresponse and error still require separate sensitivity analysis.

Usage:
    python scripts/validate.py sample --filepath=<EXTRACT_OUTPUT.gzip> \
        --n=200 --out=validation/njg-sample.csv

    # ... a human fills in the truth_* columns ...

    python scripts/validate.py score --filepath=validation/njg-sample.csv
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from common import newspaper_from_path

# Columns the coder fills in. Kept few and concrete: every extra field is a
# judgement call the coder has to make consistently 200 times.
CODING_COLUMNS = {
    'truth_has_address': 'y / n / unclear — does the ad state a street address or a city?',
    'truth_address': '     free text — the address as printed, if any',
    'truth_is_job_ad': 'y / n / unclear — is this a job advertisement at all?',
    'truth_address_is_worksite': 'y / n / unclear — worksite, or an application address?',
    'truth_has_wage': 'y / n / unclear — does the ad state pay?',
    'truth_wage': '     free text — the wage as printed, if any',
    'pred_address_correct': 'y / n / partial — REQUIRED when pred_addresses is non-empty',
    'pred_wage_correct': 'y / n / partial — REQUIRED when pred_wage is non-empty',
    'coder_notes': '     anything ambiguous',
}

# Written by `sample`, consumed by `score`. Without these the design cannot be
# undone and every reported number is an artefact of the sampling.
DESIGN_COLUMNS = ['stratum', 'stratum_size', 'stratum_drawn', 'design_weight']

YES = {'y', 'yes', 'true', '1'}
NO = {'n', 'no', 'false', '0'}


def decade(year):
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    return y // 10 * 10 if y > 0 else None


def address_values(value):
    '''Normalize the parquet address cell or reject a malformed value.'''
    if isinstance(value, (list, tuple, np.ndarray)):
        return value
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return []
    raise ValueError(
        "Each `addresses` value must be a list/array of candidates or missing; "
        "found {!r}.".format(type(value).__name__))


def emitted_address(value):
    return len(address_values(value)) > 0


# ---------------------------------------------------------------- sampling ---

def _cell_sort_key(cell):
    d, found = cell
    return (d is None, d if d is not None else 0, not found)


def _sqrt_allocation(side, target):
    '''Allocate an exact side total across cells, with one row per cell.'''
    if not side:
        if target:
            raise ValueError("Cannot allocate rows to an empty sampling side.")
        return {}
    if target < len(side) or target > sum(len(group) for group in side.values()):
        raise ValueError("Side target is incompatible with its stratum capacities.")

    keys = sorted(side, key=_cell_sort_key)
    roots = {key: math.sqrt(len(side[key])) for key in keys}

    # Find the continuous, capacity-constrained square-root allocation
    # x_h = clip(lambda*sqrt(N_h), 1, N_h). Binary search is O(H), independent
    # of requested n.
    low, high = 0.0, target / min(roots.values())
    for _ in range(80):
        scale = (low + high) / 2
        total = sum(min(len(side[key]), max(1.0, scale * roots[key]))
                    for key in keys)
        if total < target:
            low = scale
        else:
            high = scale
    continuous = {
        key: min(len(side[key]), max(1.0, high * roots[key])) for key in keys}
    alloc = {key: int(math.floor(continuous[key])) for key in keys}

    # Largest remainders make the integer allocation exact and deterministic.
    remaining = target - sum(alloc.values())
    eligible = [key for key in keys if alloc[key] < len(side[key])]
    ranked = sorted(eligible,
        key=lambda key: (-(continuous[key] - alloc[key]), _cell_sort_key(key)))
    if remaining < 0 or remaining > len(ranked):
        raise RuntimeError("Could not round the constrained sampling allocation.")
    for key in ranked[:remaining]:
        alloc[key] += 1
    return alloc


def sample_allocation(cells, n):
    '''Return exact cell allocations, balancing found/empty when feasible.'''
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("Sample size n must be an integer.")
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    population = sum(len(group) for group in cells.values())
    if population <= 0:
        raise ValueError("Cannot sample an empty population.")
    target = min(n, population)
    if target < len(cells):
        raise ValueError(
            "Sample size {} is smaller than the {} nonempty strata; increase n "
            "so every stratum has a positive inclusion probability.".format(
                target, len(cells)))

    sides = {
        found: {key: group for key, group in cells.items() if key[1] == found}
        for found in (True, False)
    }
    lower = {found: len(sides[found]) for found in sides}
    upper = {found: sum(len(group) for group in sides[found].values())
             for found in sides}

    # Choose the closest feasible split to 50/50. The bounds guarantee at least
    # one draw per cell and automatically transfer shortfalls from a sparse or
    # absent side to the other side.
    found_low = max(lower[True], target - upper[False])
    found_high = min(upper[True], target - lower[False])
    found_target = min(max(target // 2, found_low), found_high)
    side_targets = {True: found_target, False: target - found_target}

    allocation = {}
    for found in (True, False):
        allocation.update(_sqrt_allocation(sides[found], side_targets[found]))
    if sum(allocation.values()) != target:
        raise RuntimeError("Sampling allocation did not reach its exact target.")
    return allocation

def draw_sample(df, n, seed):
    ''' Stratify by (decade, whether the pipeline emitted an address).

    Half the sample is allocated to each side so recall is estimable; within a
    side, decades are allocated proportionally to the square root of their size,
    which keeps small decades usable without letting them dominate. Each row
    carries the weight needed to undo all of this at scoring time.
    '''
    if 'addresses' not in df.columns:
        raise ValueError("Sampling requires an `addresses` column.")
    df = df.copy()
    df['_decade'] = df['year'].apply(decade) if 'year' in df.columns else None
    df['_found'] = df.addresses.apply(emitted_address)

    # Ads with no usable year still belong in the frame; they form their own
    # stratum rather than being silently dropped.
    cells = {}
    for (d, found), grp in df.groupby(['_decade', '_found'], dropna=False):
        if len(grp):
            normalized_decade = None if pd.isna(d) else int(d)
            cells[(normalized_decade, bool(found))] = grp

    allocation = sample_allocation(cells, n)
    rng = np.random.RandomState(seed)
    chunks = []
    for k in sorted(cells, key=_cell_sort_key):
        take, grp = allocation[k], cells[k]
        picked = grp.sample(take, random_state=rng)
        picked = picked.assign(
            stratum='{}|{}'.format(k[0] if k[0] is not None else 'no-year',
                                   'found' if k[1] else 'empty'),
            stratum_size=len(grp),
            stratum_drawn=take,
            design_weight=len(grp) / take)
        chunks.append(picked)

    sample = pd.concat(chunks).sample(frac=1, random_state=rng)
    return sample


def write_template(sample, out_path, newspaper, population=None):
    cols = {'sample_id': range(len(sample))}
    if 'id' in sample.columns:
        cols['ad_id'] = sample['id'].values
    if 'year' in sample.columns:
        cols['year'] = sample['year'].values
    cols['ad_text'] = [_shorten(t) for t in sample.raw_content]
    cols['pred_addresses'] = [
        ' | '.join(sorted({_fmt_addr(a) for a in address_values(arr)}))
        for arr in sample.addresses]
    cols['pred_wage'] = (sample['wage'].values if 'wage' in sample.columns
                         else [None] * len(sample))
    out = pd.DataFrame(cols)
    for c in DESIGN_COLUMNS:
        out[c] = sample[c].values
    for c in CODING_COLUMNS:
        out[c] = ''
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)

    key_path = os.path.splitext(out_path)[0] + '-CODEBOOK.md'
    with open(key_path, 'w') as fh:
        fh.write(_codebook(newspaper, len(out), population))
    return key_path


def _shorten(text):
    ''' Trim ad text for the coder without breaking the CSV.

    Newlines and quotes are handled by the csv writer; the length cap keeps a
    spreadsheet cell readable.
    '''
    t = ' '.join(str(text).split())
    return t if len(t) <= 1200 else t[:1200] + ' …'


def _fmt_addr(a):
    if not isinstance(a, dict):
        return str(a)
    parts = [a.get('housenumber'), a.get('street'), a.get('city'),
             a.get('state'), a.get('zipcode')]
    return ' '.join(str(p) for p in parts if p) or '(empty)'


def _codebook(newspaper, n, population=None):
    lines = [
        '# Coding instructions — {} validation sample (n={})'.format(newspaper, n),
        '',
        'Fill in the `truth_*` columns from the ad text ALONE. Only after that,',
        'compare against `pred_*` and fill the two judgement columns. Coding the',
        'truth first is what keeps the prediction from anchoring the judgement.',
        '',
        '**Do not sort, filter or delete rows, and do not edit the `stratum`,',
        '`stratum_size`, `stratum_drawn` or `design_weight` columns.** They record',
        'how each row was selected; scoring is wrong without them.',
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
        '- **`unclear`** is a real answer for any `truth_*` column when the OCR is',
        '  unreadable. Rows coded `unclear` are excluded from that statistic and',
        '  reported separately; guessing instead would quietly bias the result.',
        '- **Whenever `pred_addresses` is non-empty you must fill',
        '  `pred_address_correct`** (same for wages). A blank there is treated as',
        '  "not yet coded" and the row is dropped from precision, so leaving it',
        '  empty silently shrinks the sample rather than counting against the',
        '  pipeline.',
        '- Non-job ads (real estate, personals, notices) are in the sample on',
        '  purpose: the pipeline is supposed to skip them. Code',
        '  `truth_is_job_ad = n`; they are scored in their own section and are',
        '  excluded from the address precision and recall numbers.',
        '',
        '## What the numbers will mean',
        '',
        '- **Precision** = of the ads where the pipeline emitted an address, how',
        '  many were right.',
        '- **Recall** = of the ads that genuinely contain an address, how many the',
        '  pipeline found *and got right*. Measurable only because empty ads are',
        '  deliberately over-sampled.',
        '- Every figure is **design-weighted** back to the corpus, so it estimates',
        '  the newspaper, not this sample. Precision, recall and coverage are each',
        '  reported overall; recall is additionally broken out by decade.',
        '- Reported intervals are **approximate Kish-effective-n Wilson** intervals;',
        '  they do not correct for coder nonresponse or coding error.',
    ]
    if population:
        lines += [
            '',
            'For reference, measured on the full {:,}-ad file this sample was drawn'.format(
                population['n']),
            'from: the pipeline emitted an address for {:.1%} of ads.'.format(
                population['coverage']),
        ]
    lines += ['', 'Run `python scripts/validate.py score --filepath=<this csv>` when done.']
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------- scoring ---

def wilson(k, n, z=1.96):
    ''' Wilson score interval — behaves sensibly at small n and at 0/1, where
    the normal approximation does not. '''
    if n <= 0:
        return (float('nan'), float('nan'), float('nan'))
    p = min(1.0, max(0.0, k / n))
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    lo = 0.0 if p == 0.0 else max(0.0, centre - half)
    hi = 1.0 if p == 1.0 else min(1.0, centre + half)
    return p, lo, hi


def weighted_rate(mask_num, mask_den, weights):
    ''' Design-weighted ratio with an approximate Kish-n Wilson interval.

    Returns (estimate, low, high, effective_n, raw_denominator_count).
    '''
    w_den = weights[mask_den]
    if len(w_den) == 0 or w_den.sum() == 0:
        return (float('nan'), float('nan'), float('nan'), 0.0, 0)
    w_num = weights[mask_den & mask_num]
    p = min(1.0, max(0.0, w_num.sum() / w_den.sum()))
    # Kish: unequal weights cost variance, so the interval is computed at the
    # effective sample size rather than the raw row count.
    n_eff = (w_den.sum() ** 2) / (w_den ** 2).sum()
    _, lo, hi = wilson(p * n_eff, n_eff)
    return (p, lo, hi, n_eff, int(mask_den.sum()))


def _code(series):
    return series.astype(str).str.strip().str.lower()


def validate_template(df):
    ''' Fail loudly on a template that cannot be scored correctly. '''
    if df.empty:
        raise SystemExit("The coding template contains no rows.")
    missing = [c for c in DESIGN_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            "This file is missing the design columns {}.\n"
            "It was either produced by an older version of `sample`, or the "
            "columns were edited out. Scoring without them would silently "
            "report the sampling design instead of the pipeline.".format(missing))
    missing = [c for c in CODING_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit("Missing coding columns: {}".format(missing))
    w = pd.to_numeric(df.design_weight, errors='coerce')
    if w.isna().any() or (w <= 0).any():
        bad = int(w.isna().sum() + (w <= 0).sum())
        raise SystemExit(
            "{} row(s) have a missing or non-positive design_weight. A shifted "
            "column (often an unquoted comma in a free-text cell) is the usual "
            "cause; re-export from the spreadsheet with quoting enabled.".format(bad))

    if df.stratum.isna().any() or df.stratum.astype(str).str.strip().eq('').any():
        raise SystemExit("Every row must carry a nonempty stratum label.")
    sizes = pd.to_numeric(df.stratum_size, errors='coerce')
    drawn = pd.to_numeric(df.stratum_drawn, errors='coerce')
    invalid_meta = (sizes.isna() | drawn.isna() | (sizes <= 0) | (drawn <= 0)
                    | (drawn > sizes)
                    | ~drawn.map(lambda value: float(value).is_integer())
                    | ~sizes.map(lambda value: float(value).is_integer()))
    if invalid_meta.any():
        raise SystemExit(
            "{} row(s) have invalid stratum_size/stratum_drawn metadata."
            .format(int(invalid_meta.sum())))

    expected_weight = sizes / drawn
    if not np.allclose(w, expected_weight, rtol=1e-9, atol=1e-12):
        raise SystemExit(
            "design_weight must equal stratum_size / stratum_drawn on every row.")

    metadata = pd.DataFrame({
        'stratum': df.stratum.astype(str).str.strip(),
        'size': sizes.astype(int), 'drawn': drawn.astype(int), 'weight': w})
    for label, group in metadata.groupby('stratum', sort=False):
        if (group['size'].nunique() != 1 or group['drawn'].nunique() != 1
                or group['weight'].nunique() != 1):
            raise SystemExit(
                "Stratum {!r} has inconsistent design metadata.".format(label))
        expected_rows = int(group['drawn'].iloc[0])
        if len(group) != expected_rows:
            raise SystemExit(
                "Stratum {!r} records stratum_drawn={} but the template contains "
                "{} row(s); rows were deleted or duplicated.".format(
                    label, expected_rows, len(group)))
    for col in CODING_COLUMNS:
        vals = set(_code(df[col])) - {''}
        if col.startswith('truth_') and not col.endswith(('_address', '_wage')):
            unexpected = vals - YES - NO - {'unclear'}
            if unexpected:
                print("warning: {} contains unexpected codes {} — treated as "
                      "uncoded".format(col, sorted(unexpected)))
    return w


def score(df):
    ''' Report design-weighted precision/recall/coverage with intervals. '''
    weights = validate_template(df)

    pred_any = df.pred_addresses.astype(str).str.strip().ne('')
    truth = _code(df.truth_has_address)
    truth_yes, truth_no = truth.isin(YES), truth.isin(NO)
    coded = truth_yes | truth_no
    unclear = truth.eq('unclear')

    correct = _code(df.pred_address_correct)
    judged = correct.isin({'y', 'n', 'partial'})
    is_job = _code(df.truth_is_job_ad)
    job_yes, job_no = is_job.isin(YES), is_job.isin(NO)

    n_total = len(df)
    print("\n{} rows; {} coded, {} unclear, {} left blank.".format(
        n_total, int(coded.sum()), int(unclear.sum()),
        int(n_total - coded.sum() - unclear.sum())))
    print("Every figure is design-weighted back to the corpus; n_eff is Kish's "
          "effective sample size and intervals are approximate.\n")

    def line(label, num, den):
        p, lo, hi, n_eff, raw = weighted_rate(num, den, weights)
        if raw == 0 or math.isnan(p):
            print("  {:<32s}      n/a  (no rows qualify)".format(label))
        else:
            print("  {:<32s} {:>6.1%}   [{:.1%}, {:.1%}]   rows={} n_eff={:.0f}".format(
                label, p, lo, hi, raw, n_eff))

    # Non-job ads are excluded here and scored in their own section: an ad the
    # pipeline should have skipped is not evidence about address accuracy.
    base = coded & job_yes
    print("ADDRESSES  (job ads only)")
    line("precision (strict)", correct.eq('y'), pred_any & judged & base)
    line("precision (incl. partial)", correct.isin({'y', 'partial'}),
         pred_any & judged & base)
    # Recall requires the emitted address to be RIGHT: an ad where the pipeline
    # emitted something wrong has not been recalled.
    line("recall (emitted and correct)", pred_any & correct.isin({'y', 'partial'}),
         truth_yes & job_yes & (judged | ~pred_any))
    line("coverage (emitted anything)", pred_any, base)

    ws = _code(df.truth_address_is_worksite)
    if (ws.isin(YES) | ws.isin(NO)).any():
        print("\nCONSTRUCT VALIDITY")
        line("addresses that are the worksite", ws.isin(YES), ws.isin(YES) | ws.isin(NO))
        print("       (the rest are application addresses — the pipeline cannot")
        print("        distinguish these, so this bounds what the geography")
        print("        variable can mean)")

    pred_wage = df.pred_wage.astype(str).str.strip().replace('nan', '').ne('')
    tw = _code(df.truth_has_wage)
    wc = _code(df.pred_wage_correct)
    wjudged = wc.isin({'y', 'n', 'partial'})
    if (tw.isin(YES) | tw.isin(NO)).any():
        print("\nWAGES")
        line("precision (strict)", wc.eq('y'), pred_wage & wjudged & job_yes)
        line("recall (emitted and correct)", pred_wage & wc.isin({'y', 'partial'}),
             tw.isin(YES) & job_yes & (wjudged | ~pred_wage))

    if job_no.any():
        print("\nAD FILTER  (ads the pipeline should have skipped)")
        line("non-job ads given an address", pred_any, job_no)

    if 'year' in df.columns:
        print("\nRECALL BY DECADE")
        decades = sorted({d for d in df.year.apply(decade) if d is not None})
        for d in decades:
            rows = df.year.apply(decade).eq(d)
            den = rows & truth_yes & job_yes & (judged | ~pred_any)
            if den.sum() == 0:
                continue
            line("  {}s".format(d), pred_any & correct.isin({'y', 'partial'}), den)


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

    if not os.path.isfile(args.filepath):
        parser.error('Invalid --filepath.')

    if args.mode == 'sample':
        df = pd.read_parquet(args.filepath)
        if 'addresses' not in df.columns:
            raise SystemExit("Needs an extract output with `addresses`.")
        pop = {'n': len(df),
               'coverage': df.addresses.apply(emitted_address).mean()}
        sample = draw_sample(df, args.n, args.seed)
        key = write_template(sample, args.out, newspaper_from_path(args.filepath), pop)
        found = sample.design_weight[sample.addresses.apply(emitted_address)]
        print("Wrote {} rows to {} (seed {}).".format(len(sample), args.out, args.seed))
        print("  {} strata; design weights {:.1f}-{:.1f} (each row stands for that "
              "many ads).".format(sample.stratum.nunique(),
                                  sample.design_weight.min(), sample.design_weight.max()))
        print("  {} rows where the pipeline found an address, {} where it did not"
              " — the empty side is over-sampled on purpose so recall is"
              " estimable, and the weights undo it at scoring time.".format(
                  len(found), len(sample) - len(found)))
        print("Coding instructions: {}".format(key))
    else:
        score(pd.read_csv(args.filepath, keep_default_na=False))
