"""Build the final per-newspaper dataset without making API requests.

Final coordinates and county fields use the same qualified Geoapify-feature
selector as resolve.py, keeping every field tied to one provider feature.

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
import pyarrow.parquet as pq

from common import USGeoData, add_filepath_suffix, newspaper_from_path, time_now
from resolve import select_geoapify_feature


def best_coordinates(response_list, US_DATA):
    ''' Best latitude/longitude/county for one ad, from its stored responses.

    Uses resolve.py's selection to return the exact feature properties, then
    reads every output field from that object.
    '''
    out = {'address':None, 'county':None, 'postcode':None,
           'latitude':None, 'longitude':None, 'coordinates_confidence':None}
    best_conf = 0
    for response in (response_list if response_list is not None else []):
        if not isinstance(response, dict):
            continue
        props = select_geoapify_feature(response, US_DATA.biggest_nearby_cities)
        if not props:
            continue
        conf = props.get('rank', {}).get('confidence', 0)
        if conf > best_conf:
            best_conf = conf
            county = props.get('county')
            if county:
                county = county.split(' County')[0]
            out.update({'address':props.get('formatted'), 'county':county,
                        'postcode':props.get('postcode'),
                        'latitude':props.get('lat'), 'longitude':props.get('lon'),
                        'coordinates_confidence':conf})
    return out


def consolidate_counties(coordinates, geo, US_DATA=None):
    '''Attach county diagnostics and choose the county matching the coordinates.

    When a feature supplied the selected latitude/longitude, `county_final` must
    come from that same feature.  The resolver's modal county is a useful
    fallback only when no coordinate feature was selected at all; otherwise it
    can silently describe a different candidate query.
    '''
    coordinates = coordinates.copy()
    for column in ('geo_county', 'geo_zip_county'):
        if column in geo.columns:
            coordinates[column] = geo[column].to_numpy()

    coordinates['county_final'] = coordinates['county']
    # Some geocoders omit a county while still returning the chosen feature's
    # postcode.  Resolving that one postcode preserves the coordinate/county
    # relationship; the modal postcode county from all queries would not.
    if US_DATA is not None and 'postcode' in coordinates.columns:
        selected_zip_county = coordinates['postcode'].map(
            lambda zipcode: (US_DATA.counties_from_zips([zipcode]) or [None])[0]
            if pd.notna(zipcode) else None)
        coordinates['county_final'] = coordinates['county_final'].fillna(
            selected_zip_county)
    no_selected_feature = coordinates['coordinates_confidence'].isna()
    if 'geo_county' in coordinates.columns:
        coordinates.loc[no_selected_feature, 'county_final'] = (
            coordinates.loc[no_selected_feature, 'county_final'].fillna(
                coordinates.loc[no_selected_feature, 'geo_county']))
    if 'geo_zip_county' in coordinates.columns:
        coordinates.loc[no_selected_feature, 'county_final'] = (
            coordinates.loc[no_selected_feature, 'county_final'].fillna(
                coordinates.loc[no_selected_feature, 'geo_zip_county']))
    return coordinates


def read_wage_columns(filepath):
    '''Read the available wage fields with one projected parquet read.'''
    available = set(pq.ParquetFile(filepath).schema_arrow.names)
    if 'id' not in available:
        raise ValueError("No `id` column in --wage file to join on.")
    wanted = ['id', 'wage', 'wage_amount', 'wage_period', 'wage_is_range',
              'wage_n_amounts']
    selected = [column for column in wanted if column in available]
    if len(selected) <= 1:
        raise ValueError("No wage columns found in --wage file.")
    return pd.read_parquet(filepath, columns=selected)


def main(args):
    newspaper = newspaper_from_path(args.geo)
    US_DATA = USGeoData(
        os.path.join(args.aux_dir, "states.csv"),
        os.path.join(args.aux_dir, "geo/uscities.csv"),
        os.path.join(args.aux_dir, "neighbors-states.csv"),
        os.path.join(args.aux_dir, "geo/uszips.csv")
    ).load(newspaper, min_pop=args.min_pop)

    geo = pd.read_parquet(args.geo)
    if 'geo_requests' not in geo.columns:
        raise ValueError("No `geo_requests` column in --geo file.")
    if 'id' not in geo.columns:
        raise ValueError("No `id` column in --geo file to join on.")
    print("Read {} geolocated ads from {}.".format(len(geo), newspaper))

    coordinates = pd.DataFrame(
        [best_coordinates(r, US_DATA) for r in geo.geo_requests], index=geo.index)
    coordinates['id'] = geo['id']
    coordinates = consolidate_counties(coordinates, geo, US_DATA)
    del geo

    base = (pd.read_csv(args.base, index_col=[0]) if args.base.endswith('.csv')
            else pd.read_parquet(args.base))
    final = base.merge(coordinates, on='id', how='left')
    if len(final) != len(base):
        raise RuntimeError(
            "Merge changed row count ({} -> {}): duplicate ids in the geo file."
            .format(len(base), len(final)))
    del base, coordinates

    if args.wage:
        wages = read_wage_columns(args.wage)
        n_before = len(final)
        final = final.merge(wages, on='id', how='left')
        if len(final) != n_before:
            raise RuntimeError(
                "Wage merge changed row count ({} -> {}): duplicate ids."
                .format(n_before, len(final)))
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
    parser.add_argument('--min_pop', type=int, default=USGeoData.DEFAULT_MIN_POP,
        help="Minimum city population used when interpreting stored responses.")
    parser.add_argument('-o', '--output_dir', type=str, default='./output')
    parser.add_argument('--write_csv', type=int, default=0,
        help="Also write a CSV copy (large: these files carry full ad text).")
    args = parser.parse_args()

    if not os.path.isfile(args.geo):
        parser.error('Invalid --geo filepath.')
    if not os.path.isfile(args.base):
        parser.error('Invalid --base filepath.')
    if args.wage is not None and not os.path.isfile(args.wage):
        parser.error('Invalid --wage filepath.')
    if not os.path.isdir(args.aux_dir):
        parser.error('Invalid --aux_dir.')
    # The output directory is ours to create; asserting on it only
    # made a first run fail on a path the user never chose.
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
