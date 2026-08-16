"""Build the final per-newspaper dataset. Makes no API requests.

This replaces the `merge_final` / `best_coordinates` snippets that used to live in
the README as prose. Keeping them there had two costs: they could not be run or
tested, and `best_coordinates` selected geocoder features by *different* rules
than resolve.py did — no confidence floor and no nearby-city filter — so an ad's
final coordinates could come from a feature its own `geo_county` had rejected.
This script calls the same `select_geoapify` the resolver uses, so the coordinate
and county columns are guaranteed to describe the same chosen feature.

Inputs (all local files; nothing is fetched):
  --geo       resolve output carrying the `geo_requests` column
  --base      the newspaper's ad-level dataset, joined on `id`
  --wage      optional extract output carrying wage columns, joined on `id`

Usage:
    python scripts/finalize.py --geo=./7-geolocation/NJG-resolve-all.gzip \
        --base=./6-final-datasets/NJG.csv --wage=./8-employer/NJG-extract-all.gzip \
        --aux_dir=./auxiliary_files --output_dir=./9-final
"""
import argparse
import os

import pandas as pd

from common import USGeoData, add_filepath_suffix, newspaper_from_path, time_now
from resolve import select_geoapify


def best_coordinates(response_list, US_DATA):
    ''' Best latitude/longitude/county for one ad, from its stored responses.

    Uses resolve.py's selection to choose the feature, then reads the coordinate
    fields off that same feature, so this cannot disagree with `geo_county`.
    '''
    out = {'address':None, 'county':None, 'postcode':None,
           'latitude':None, 'longitude':None, 'coordinates_confidence':None}
    best_conf = 0
    for response in (response_list if response_list is not None else []):
        if not isinstance(response, dict):
            continue
        address, county, postcode = select_geoapify(response,
            US_DATA.biggest_nearby_cities)
        if address is None and county is None and postcode is None:
            continue
        # Recover the confidence of the feature select_geoapify chose, so the
        # best feature ACROSS candidate queries wins (select_geoapify picks the
        # best feature WITHIN one response).
        content = response.get('content') or {}
        conf = 0
        latitude = longitude = None
        for feature in content.get('features', []):
            props = (feature or {}).get('properties') or {}
            if props.get('formatted') != address:
                continue
            conf = props.get('rank', {}).get('confidence', 0)
            latitude, longitude = props.get('lat'), props.get('lon')
            break
        if conf > best_conf:
            best_conf = conf
            out.update({'address':address, 'county':county, 'postcode':postcode,
                        'latitude':latitude, 'longitude':longitude,
                        'coordinates_confidence':conf})
    return out


def main(args):
    newspaper = newspaper_from_path(args.geo)
    US_DATA = USGeoData(
        os.path.join(args.aux_dir, "states.csv"),
        os.path.join(args.aux_dir, "simplemaps/uscities.csv"),
        os.path.join(args.aux_dir, "neighbors-states.csv"),
        os.path.join(args.aux_dir, "simplemaps/uszips.csv")
    ).load(newspaper)

    geo = pd.read_parquet(args.geo)
    assert 'geo_requests' in geo.columns, "No `geo_requests` column in --geo file."
    assert 'id' in geo.columns, "No `id` column in --geo file to join on."
    print("Read {} geolocated ads from {}.".format(len(geo), newspaper))

    coordinates = pd.DataFrame(
        [best_coordinates(r, US_DATA) for r in geo.geo_requests], index=geo.index)
    coordinates['id'] = geo['id']
    # Consolidate the two county sources, the rule that used to sit only in the
    # README's statistics snippet.
    if 'geo_county' in geo.columns:
        coordinates['geo_county'] = geo['geo_county'].values
    if 'geo_zip_county' in geo.columns:
        coordinates['geo_zip_county'] = geo['geo_zip_county'].values
    coordinates['county_final'] = coordinates.get('geo_county')
    if 'geo_zip_county' in coordinates.columns:
        coordinates['county_final'] = coordinates['county_final'].fillna(
            coordinates['geo_zip_county'])
    coordinates['county_final'] = coordinates['county_final'].fillna(
        coordinates['county'])
    del geo

    base = (pd.read_csv(args.base, index_col=[0]) if args.base.endswith('.csv')
            else pd.read_parquet(args.base))
    final = base.merge(coordinates, on='id', how='left')
    assert len(final) == len(base), (
        "Merge changed row count ({} -> {}): duplicate ids in the geo file.".format(
            len(base), len(final)))
    del base, coordinates

    if args.wage:
        wage_cols = ['id', 'wage', 'wage_amount', 'wage_period', 'wage_is_range']
        available = [c for c in wage_cols
                     if c in pd.read_parquet(args.wage).columns] or ['id', 'wage']
        wages = pd.read_parquet(args.wage, columns=available)
        n_before = len(final)
        final = final.merge(wages, on='id', how='left')
        assert len(final) == n_before, (
            "Wage merge changed row count ({} -> {}): duplicate ids.".format(
                n_before, len(final)))
        del wages

    print("Final shape: {}. Ads with any county: {} ({:.1%}).".format(
        final.shape, final.county_final.notna().sum(),
        final.county_final.notna().mean()))
    final.to_parquet(add_filepath_suffix(args.output_dir, newspaper, n=len(final),
        suffix='final'), compression='gzip')
    if args.write_csv:
        final.to_csv(add_filepath_suffix(args.output_dir, newspaper, n=len(final),
            suffix='final', ext='csv'))
    print("Completed at {}.".format(time_now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--geo', type=str, required=True,
        help="Resolve output with a geo_requests column.")
    parser.add_argument('--base', type=str, required=True,
        help="Ad-level dataset to attach the geolocation to (csv or parquet).")
    parser.add_argument('--wage', type=str, default=None,
        help="Optional extract output carrying wage columns.")
    parser.add_argument('-a', '--aux_dir', type=str, default='./auxiliary_files')
    parser.add_argument('-o', '--output_dir', type=str, default='.')
    parser.add_argument('--write_csv', type=int, default=0,
        help="Also write a CSV copy (large: these files carry full ad text).")
    args = parser.parse_args()

    assert os.path.isfile(args.geo), 'Invalid --geo filepath.'
    assert os.path.isfile(args.base), 'Invalid --base filepath.'
    assert args.wage is None or os.path.isfile(args.wage), 'Invalid --wage filepath.'
    assert os.path.isdir(args.aux_dir), 'Invalid --aux_dir.'
    assert os.path.isdir(args.output_dir), 'Invalid --output_dir.'
    main(args)
