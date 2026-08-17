"""Re-derive county assignments WITHOUT making any API requests.

Two independent offline sources, both usable with no API key:

1. `--from_requests` (default): re-score the Geoapify responses already stored in
   the `geo_requests` column of a resolve output, using the same selection rules
   as resolve.py: the highest-confidence nearby-city feature supplies every
   output field, and postcode-to-county mapping uses `uszips.csv`.

2. `--from_addresses`: derive counties straight from the `addresses` column of an
   extract output. Candidates carrying a zipcode already contain a county taken
   from the US city table, so those ads never needed a geocoder at all. Use this
   when no resolve output exists.

What cannot be recovered offline: any address candidate that was never sent to
the provider has no stored response. Re-run extraction and live resolution when
candidate-generation rules or geographic scope change.

Usage:
    python scripts/recompute.py --filepath=<RESOLVE_OUTPUT.gzip> \
        --aux_dir=./auxiliary_files --output_dir=<DIR>

    python scripts/recompute.py --from_addresses \
        --filepath=<EXTRACT_OUTPUT.gzip> --aux_dir=./auxiliary_files \
        --output_dir=<DIR>
"""
import argparse
import os
import time

import pandas as pd
from statistics import mode

from common import USGeoData, add_filepath_suffix, newspaper_from_path, time_now
from resolve import select_geoapify, summarize


def recompute_from_requests(row_requests, US_DATA):
    ''' Re-score one ad's stored responses. Makes no network calls. '''
    addresses, counties, zipcodes = [], [], []
    n_usable = 0
    for response in (row_requests if row_requests is not None else []):
        if not isinstance(response, dict):
            continue
        if response.get('status_code') == 200:
            n_usable += 1
        address, county, zipcode = select_geoapify(response,
            US_DATA.biggest_nearby_cities)
        addresses.append(address)
        if county: counties.append(county)
        if zipcode: zipcodes.append(zipcode)
    out = summarize(addresses, counties, zipcodes, US_DATA, prefix='rc')
    out['rc_usable_responses'] = n_usable
    return out


def counties_from_addresses(address_dicts, US_DATA):
    ''' County directly from extracted candidates — no geocoder involved.

    Candidates built from a matched zipcode already carry the county the US city
    table gives for that zipcode; a zipcode alone still resolves through
    uszips.csv. Returns the modal county over whatever is available.
    '''
    counties, zipcodes = [], []
    for addr in (address_dicts if address_dicts is not None else []):
        if not isinstance(addr, dict):
            continue
        if addr.get('county'):
            counties.append(addr['county'])
        if addr.get('zipcode'):
            zipcodes.append(addr['zipcode'])
    zip_counties = US_DATA.counties_from_zips(zipcodes) or []
    zip_fips = US_DATA.fips_from_zips(zipcodes) or []
    # A county carried on the candidate is direct evidence; a county inferred
    # from a zipcode is the fallback, matching resolve()'s geo/geo_zip split.
    out = {
        'offline_county': mode(counties) if counties else None,
        'offline_zip_county': mode(zip_counties) if zip_counties else None,
        # FIPS of the ZIP-derived county specifically. Named for its source: it
        # is not the code for `offline_county`, and the two disagree on ~4% of
        # ads, so a generic name would invite joining on the wrong one.
        'offline_zip_county_fips': mode(zip_fips) if zip_fips else None,
        'offline_n_zipcodes': len(set(zipcodes)),
    }
    # Flag the disagreement rather than hiding it behind a single column.
    out['offline_county_agrees'] = (
        None if (out['offline_county'] is None or out['offline_zip_county'] is None)
        else out['offline_county'] == out['offline_zip_county'])
    return out


def unusable_responses(requests_logs):
    ''' Stored responses that carry no usable payload, so they answer nothing.

    resolve() appends exactly one log per candidate — including for timeouts and
    connection errors — so counting `len(addresses) - len(logs)` was always zero
    and the report claimed no candidate needed a new call. What actually needs
    re-fetching is a candidate whose stored response is a non-200 or has no body.
    '''
    if requests_logs is None:
        return 0
    return sum(1 for r in requests_logs
               if not isinstance(r, dict)
               or r.get('status_code') != 200
               or not r.get('content'))


def report(df, US_DATA, from_addresses=False):
    ''' Print what changed, so the effect is quantified before anything is used. '''
    n = len(df)
    print("\n=== Recompute report ({} ads) ===".format(n))
    if from_addresses:
        direct = df.offline_county.notna()
        viazip = df.offline_zip_county.notna()
        print("Counties from candidate fields : {} ({:.1%})".format(
            direct.sum(), direct.mean()))
        print("Counties via zipcode lookup    : {} ({:.1%})".format(
            viazip.sum(), viazip.mean()))
        hasfips = df.offline_zip_county_fips.notna()
        print("With a joinable county FIPS    : {} ({:.1%})".format(
            hasfips.sum(), hasfips.mean()))
        agree = df.offline_county_agrees
        if agree.notna().any():
            print("Both sources agree on county   : {} of {} ({:.1%})".format(
                int((agree == True).sum()), int(agree.notna().sum()),
                (agree == True).sum() / agree.notna().sum()))
        print("Any offline county             : {} ({:.1%})".format(
            (direct | viazip).sum(), (direct | viazip).mean()))
        return

    had = df.geo_county.notna() if 'geo_county' in df.columns else None
    now = df.rc_county.notna()
    print("Ads with a county before : {}".format(int(had.sum()) if had is not None else 'n/a'))
    print("Ads with a county after  : {}".format(int(now.sum())))
    if had is not None:
        gained = (~had & now).sum()
        lost = (had & ~now).sum()
        both = had & now
        changed = (df.geo_county[both] != df.rc_county[both]).sum()
        print("  gained : {}".format(int(gained)))
        print("  lost   : {}".format(int(lost)))
        print("  changed county (had one before and after): {} of {}".format(
            int(changed), int(both.sum())))
        if changed:
            ex = df[both & (df.geo_county != df.rc_county)].head(5)
            for _, r in ex.iterrows():
                print("    {!r} -> {!r}".format(r.geo_county, r.rc_county))
    if 'rc_unusable_responses' in df.columns:
        tot = int(df.rc_unusable_responses.sum())
        ads = int((df.rc_unusable_responses > 0).sum())
        print("Stored responses with no usable payload: {} across {} ads".format(tot, ads))
        print("  -> non-200 or empty-bodied responses. Nothing offline can resolve")
        print("     these; only a new geocoding request could.")
    print("NOTE: candidates that were never sent have no stored response and do")
    print("      not appear in this file. Re-run extraction and live resolution")
    print("      when candidate-generation rules or geographic scope change.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filepath', type=str, required=True,
        help="Resolve output (.gzip) with geo_requests, or extract output with --from_addresses.")
    parser.add_argument('--from_addresses', action='store_true',
        help="Derive counties from the `addresses` column instead of stored responses.")
    parser.add_argument('-n', '--nrows', type=int, default=None)
    parser.add_argument('-a', '--aux_dir', type=str, default='./auxiliary_files')
    parser.add_argument('--min_pop', type=int, default=USGeoData.DEFAULT_MIN_POP,
        help="Minimum city population used when interpreting stored responses.")
    parser.add_argument('-o', '--output_dir', type=str, default='./output')
    args = parser.parse_args()

    if not os.path.isfile(args.filepath):
        parser.error('Invalid filepath to data.')
    if not os.path.isdir(args.aux_dir):
        parser.error('Invalid filepath to auxiliary files.')
    # The output directory is ours to create; asserting on it only
    # made a first run fail on a path the user never chose.
    os.makedirs(args.output_dir, exist_ok=True)

    newspaper = newspaper_from_path(args.filepath)
    US_DATA = USGeoData(
        os.path.join(args.aux_dir, "states.csv"),
        os.path.join(args.aux_dir, "geo/uscities.csv"),
        os.path.join(args.aux_dir, "neighbors-states.csv"),
        os.path.join(args.aux_dir, "geo/uszips.csv")
    ).load(newspaper, min_pop=args.min_pop)

    st = time.time()
    df = pd.read_parquet(args.filepath)
    if args.nrows:
        df = df.iloc[:args.nrows]
    print("Recomputing {} ads from {} at {} — no API requests will be made.".format(
        len(df), newspaper, time_now()))

    if args.from_addresses:
        if 'addresses' not in df.columns:
            raise ValueError("No `addresses` column in this file.")
        derived = df.addresses.apply(
            lambda a: counties_from_addresses(a, US_DATA)).to_list()
        suffix = 'offline-county'
    else:
        if 'geo_requests' not in df.columns:
            raise ValueError(
                "No `geo_requests` column: this file has no stored responses. "
                "Use --from_addresses against an extract output instead.")
        derived = df.geo_requests.apply(
            lambda r: recompute_from_requests(r, US_DATA)).to_list()
        suffix = 'recomputed'

    out = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis='columns')
    if not args.from_addresses:
        out['rc_unusable_responses'] = [unusable_responses(r) for r in df.geo_requests]

    report(out, US_DATA, from_addresses=args.from_addresses)

    out.to_parquet(add_filepath_suffix(args.output_dir, newspaper, n=args.nrows,
        suffix=suffix), compression='gzip')
    print("\nCompleted in {} seconds at {}.".format(round(time.time() - st), time_now()))
