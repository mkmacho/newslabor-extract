"""Monte Carlo verification for validate.py's stratified weighted estimator.

Run this against the exact extract produced by the bundled 2,000-row demo:

    python scripts/verify_validation_design.py \
        --filepath=demo-output/NJG-extract-all.gzip \
        --simulations=500 --seed=20260817

The simulation repeatedly draws the implemented decade x emitted-address
sample from fixed finite populations with known rates. It verifies point bias,
RMSE, and repeated-sampling coverage of the approximate Kish-Wilson interval.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from validate import decade, draw_sample, emitted_address, weighted_rate


def build_patterns(df, seed):
    '''Construct fixed finite-population outcomes for informative edge cases.'''
    rng = np.random.default_rng(seed)
    years = df['year'].apply(decade) if 'year' in df.columns else pd.Series(
        [None] * len(df), index=df.index)
    found = df.addresses.apply(emitted_address)
    valid_decades = [value for value in years if value is not None]
    low = min(valid_decades) if valid_decades else 0
    high = max(valid_decades) if valid_decades else low
    span = max(1, high - low)
    time_position = np.array([
        0.5 if value is None else (value - low) / span for value in years])

    homogeneous = pd.Series(rng.random(len(df)) < 0.55, index=df.index)

    # Strong between-stratum heterogeneity correlated with the deliberately
    # balanced found/empty allocation and with decade.
    heterogeneous_prob = np.clip(
        np.where(found.to_numpy(), 0.82, 0.12) - 0.25 * time_position,
        0.02, 0.98)
    heterogeneous = pd.Series(
        rng.random(len(df)) < heterogeneous_prob, index=df.index)

    # Within every cell, put all successes first in source order. This checks
    # that within-cell sample errors are not synchronized across strata.
    order_adverse = pd.Series(False, index=df.index)
    design = pd.DataFrame({'decade':years, 'found':found}, index=df.index)
    for _, indices in design.groupby(['decade', 'found'], dropna=False).groups.items():
        ordered = list(indices)
        order_adverse.loc[ordered[:len(ordered) // 2]] = True

    # A ratio/domain estimand: domain prevalence is concentrated in found ads,
    # while success conditional on domain is concentrated in empty ads.
    domain_prob = np.clip(
        np.where(found.to_numpy(), 0.80, 0.20) + 0.15 * time_position,
        0.02, 0.98)
    domain = pd.Series(rng.random(len(df)) < domain_prob, index=df.index)
    conditional_prob = np.clip(
        np.where(found.to_numpy(), 0.20, 0.85) - 0.10 * time_position,
        0.02, 0.98)
    domain_success = domain & pd.Series(
        rng.random(len(df)) < conditional_prob, index=df.index)

    all_rows = pd.Series(True, index=df.index)
    return {
        'homogeneous': (homogeneous, all_rows),
        'heterogeneous': (heterogeneous, all_rows),
        'order_adverse': (order_adverse, all_rows),
        'domain_adverse': (domain_success, domain),
        'all_zero': (pd.Series(False, index=df.index), all_rows),
        'all_one': (pd.Series(True, index=df.index), all_rows),
    }


def run_simulation(df, sample_size, simulations, seed):
    patterns = build_patterns(df, seed)
    work = df.copy()
    for name, (numerator, denominator) in patterns.items():
        work['_mc_{}_num'.format(name)] = numerator
        work['_mc_{}_den'.format(name)] = denominator

    truths = {
        name: float(numerator[denominator].mean())
        for name, (numerator, denominator) in patterns.items()
    }
    estimates = {name: [] for name in patterns}
    expected_n = min(sample_size, len(df))

    for replication in range(simulations):
        sample = draw_sample(work, sample_size, seed + 100000 + replication)
        if len(sample) != expected_n:
            raise RuntimeError(
                "draw_sample returned {} rows; expected {}.".format(
                    len(sample), expected_n))
        if not math.isclose(sample.design_weight.sum(), len(df),
                            rel_tol=1e-12, abs_tol=1e-9):
            raise RuntimeError("Design weights do not reconstruct the population.")
        for name in patterns:
            result = weighted_rate(
                sample['_mc_{}_num'.format(name)].astype(bool),
                sample['_mc_{}_den'.format(name)].astype(bool),
                sample.design_weight)
            estimates[name].append(result[:3])
    return truths, estimates


def summarize(truths, estimates, bias_tolerance, minimum_coverage):
    rows, passed = [], True
    for name, values in estimates.items():
        array = np.asarray(values, dtype=float)
        truth = truths[name]
        point = array[:, 0]
        finite = np.isfinite(point) & np.isfinite(array[:, 1]) & np.isfinite(array[:, 2])
        if not finite.all():
            passed = False
        point = point[finite]
        intervals = array[finite, 1:3]
        bias = float(point.mean() - truth) if len(point) else float('nan')
        rmse = float(np.sqrt(np.mean((point - truth) ** 2))) if len(point) else float('nan')
        mcse = float(point.std(ddof=1) / math.sqrt(len(point))) if len(point) > 1 else 0.0
        coverage = float(np.mean(
            (intervals[:, 0] <= truth) & (truth <= intervals[:, 1])
        )) if len(intervals) else float('nan')
        width = float(np.mean(intervals[:, 1] - intervals[:, 0])) \
            if len(intervals) else float('nan')
        bias_limit = max(bias_tolerance, 3 * mcse)
        row_pass = (len(point) == len(values)
                    and abs(bias) <= bias_limit
                    and coverage >= minimum_coverage)
        passed &= row_pass
        rows.append({
            'pattern': name, 'truth': truth, 'mean': float(point.mean()),
            'bias': bias, 'rmse': rmse, 'coverage': coverage,
            'mean_width': width, 'bias_limit': bias_limit,
            'pass': row_pass,
        })
    return pd.DataFrame(rows), passed


def main(args):
    if args.simulations <= 0 or args.n <= 0:
        raise SystemExit('--simulations and --n must be positive.')
    df = pd.read_parquet(args.filepath)
    if 'addresses' not in df.columns:
        raise SystemExit('Input must be an extract parquet with `addresses`.')

    design = pd.DataFrame({
        'decade': df['year'].apply(decade) if 'year' in df.columns else None,
        'found': df.addresses.apply(emitted_address),
    })
    cell_sizes = design.groupby(['decade', 'found'], dropna=False).size().unstack(
        fill_value=0)
    print("Finite population: {:,} rows; {} nonempty decade x found cells; "
          "emitted-address rate {:.1%}.".format(
              len(df), int((cell_sizes > 0).sum().sum()), design.found.mean()))
    print(cell_sizes.to_string())
    reference = draw_sample(df, args.n, args.seed)
    allocation = reference.groupby(['_decade', '_found']).size().unstack(
        fill_value=0)
    print("\nImplemented sample allocation (weights sum to {:.0f}):".format(
        reference.design_weight.sum()))
    print(allocation.to_string())
    print("\nMonte Carlo: {} repeated samples of n={} (fixed seed {}).".format(
        args.simulations, min(args.n, len(df)), args.seed))

    truths, estimates = run_simulation(
        df, args.n, args.simulations, args.seed)
    report, passed = summarize(
        truths, estimates, args.bias_tolerance, args.minimum_coverage)
    print("\n" + report.to_string(index=False, formatters={
        'truth':'{:.4f}'.format, 'mean':'{:.4f}'.format,
        'bias':'{:+.4f}'.format, 'rmse':'{:.4f}'.format,
        'coverage':'{:.3f}'.format, 'mean_width':'{:.4f}'.format,
        'bias_limit':'{:.4f}'.format,
    }))
    coverage_mcse = math.sqrt(0.95 * 0.05 / args.simulations)
    print("\nPass criteria: |bias| <= max({:.3f}, 3 x point-estimate MCSE); "
          "coverage >= {:.3f}. Nominal coverage is 0.95 (MCSE near {:.3f})."
          .format(args.bias_tolerance, args.minimum_coverage, coverage_mcse))
    print("Limitations: fixed synthetic finite populations, complete coding, and "
          "simple random sampling within each implemented stratum. Kish-Wilson "
          "intervals are approximate; this does not model coder error, coder "
          "nonresponse, or external validity for historical newspaper data.")
    if not passed:
        raise SystemExit("Validation-design Monte Carlo failed its stated criteria.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filepath', required=True,
        help='Extract parquet carrying year and addresses columns.')
    parser.add_argument('-n', type=int, default=200,
        help='Validation sample size (default 200).')
    parser.add_argument('--simulations', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260817)
    parser.add_argument('--bias_tolerance', type=float, default=0.005)
    parser.add_argument('--minimum_coverage', type=float, default=0.90,
        help='Undercoverage guard for the approximate nominal-95%% interval.')
    args = parser.parse_args()

    if not os.path.isfile(args.filepath):
        parser.error('Invalid --filepath.')
    main(args)
