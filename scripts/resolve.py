from concurrent.futures import ThreadPoolExecutor
import re
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, ReadTimeout
from statistics import mode
from math import ceil
from urllib.parse import urlparse
import argparse
import pandas as pd
import os
import time
import numpy as np
from common import USGeoData, add_filepath_suffix, time_now, newspaper_from_path


# Shared, connection-pooling session with retry/backoff. A fresh requests.get()
# per query paid a TCP+TLS handshake for each of tens of millions of requests,
# and a single transient failure or 429 permanently lost that address.
SESSION = requests.Session()
# Only GETs are issued here, so urllib3's default allowed_methods applies (the
# keyword was renamed across urllib3 1.x/2.x and is best left unset).
_RETRY = Retry(total=4, backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    respect_retry_after_header=True)
SESSION.mount('https://', HTTPAdapter(max_retries=_RETRY, pool_maxsize=64))
SESSION.mount('http://', HTTPAdapter(max_retries=_RETRY, pool_maxsize=64))

_RATE_LOCK = threading.Lock()
_RATE_STATE = {'min_interval':0.0, 'next_time':0.0}

def set_rate_limit(requests_per_second:float):
    ''' Throttle all threads to at most `requests_per_second` API calls. '''
    if requests_per_second < 0:
        raise ValueError("requests_per_second must be non-negative")
    _RATE_STATE['min_interval'] = 1.0 / requests_per_second if requests_per_second else 0.0


def validate_base_url(url):
    '''Require an encrypted provider endpoint without embedded credentials.'''
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or not parsed.netloc or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError(
            "Geoapify base URL must be an HTTPS origin without credentials, "
            "a query, or a fragment.")
    return url.rstrip('/')

def _throttle():
    if not _RATE_STATE['min_interval']: return
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _RATE_STATE['next_time'] - now
        _RATE_STATE['next_time'] = max(now, _RATE_STATE['next_time']) + \
            _RATE_STATE['min_interval']
    if wait > 0: time.sleep(wait)

def _redact(url:str):
    ''' Strip the API key before the URL is persisted into the output files. '''
    return re.sub(r'(apiKey=)[^&]*', r'\1REDACTED', url)

def get_wrapper(url, timeout=10):
    # content stays None (not {}) until a payload arrives: an all-failed batch of
    # empty dicts has no inferable struct schema, and pyarrow refuses to write it
    # ("Cannot write struct type 'content' with no child field"), which would kill
    # the run at its checkpoint.
    output = {'url':_redact(url), 'elapsed':None, 'content':None, 'message':""}
    try:
        _throttle()
        resp = SESSION.get(url, timeout=timeout)
        output['content'] = resp.json()
    except ReadTimeout as err:
        output['message'] = _redact(str(err) or "")
        output['elapsed'] = timeout
        resp = err
    except RequestException as err:
        output['message'] = _redact(str(err) or "")
        resp = err
    finally:
        # None (not 404) when no HTTP response came back: a timeout or connection
        # error must stay distinguishable from a genuine "not found".
        output.update({
            'status_code':getattr(resp, "status_code", None),
            'type':type(resp).__name__ or ""
        })
        try:
            output['elapsed'] = resp.elapsed.total_seconds()
        except (AttributeError, TypeError):
            pass
    return output

# --- response selection -------------------------------------------------------
# Selection is kept separate from the HTTP call so that the *same* rules can be
# re-applied to responses already stored in `geo_requests`, with no new API calls
# (see scripts/recompute.py). Any change here must therefore be understood as
# changing how stored responses are read, not only how new ones are fetched.

def select_nominatum(response, biggest_nearby_cities):
    ''' Reduce a stored Nominatim response to (address, county, zipcode). '''
    counties, zipcodes, address = [], [], None
    # Nominatim returns a dict (not a list) for error payloads; treat it as no
    # result rather than a valid candidate collection.
    if response.get('status_code') == 200 and isinstance(response.get('content'), list):
        for verified in response['content']:
            if not verified.get('address'): continue
            if not verified['address'].get('city') in biggest_nearby_cities: continue
            county = verified['address'].get('county')
            if county: counties.append(county.split(' County')[0])
            zipcode = verified['address'].get('postcode')
            if zipcode: zipcodes.append(zipcode)
            if verified.get('display_name') and not address:
                # Just use first address
                address = verified['display_name']
    return address, mode(counties or [None]), mode(zipcodes or [None])

def select_geoapify_feature(response, biggest_nearby_cities):
    '''Return the exact highest-confidence qualifying feature properties.'''
    best, best_conf = None, 0
    if (response.get('status_code') != 200
            or not isinstance(response.get('content'), dict)):
        return None
    for verified in response['content'].get('features', []):
        if not verified.get('properties'):
            continue
        props = verified['properties']
        confidence = props.get('rank', {}).get('confidence', 0)
        if confidence <= 0:
            continue
        if props.get('city') not in biggest_nearby_cities:
            continue
        if confidence <= best_conf:
            continue
        best_conf, best = confidence, props
    return best


def select_geoapify(response, biggest_nearby_cities):
    '''Reduce a stored Geoapify response to (address, county, zipcode).

    Picks the single highest-confidence qualifying feature and reads every field
    off *that* feature, preventing address, county, and postcode values from being
    combined across candidates.
    '''
    best_county, best_zipcode, address = None, None, None
    best = select_geoapify_feature(response, biggest_nearby_cities)
    if best:
        county = best.get('county')
        if county:
            best_county = county.split(' County')[0]
        best_zipcode = best.get('postcode')
        address = best.get('formatted')
    return address, best_county, best_zipcode


def nominatum_request(query, biggest_nearby_cities, timeout=10):
    url = "https://nominatim.openstreetmap.org/search?addressdetails=1&q={}&format=jsonv2".format(
        requests.utils.quote(query, safe=''))
    response = get_wrapper(url, timeout=timeout)
    return select_nominatum(response, biggest_nearby_cities) + (response,)

def geoapify_request(query, biggest_nearby_cities, timeout=10):
    url = os.environ['GEOAPIFY_URL'] + "/v1/geocode/search?text={}&apiKey={}".format(
        requests.utils.quote(query, safe=''), os.environ['GEOAPIFY_API_KEY'])
    response = get_wrapper(url, timeout=timeout)
    return select_geoapify(response, biggest_nearby_cities) + (response,)

def format_str_address(address_fields:dict):
    if not isinstance(address_fields, dict):
        raise TypeError("Address fields must be a dictionary.")
    addr_str = ''
    number = address_fields.get('housenumber')
    street = address_fields.get('street')
    if street:
        addr_str = number + ' ' + street if number else street
    for field in ['city', 'state', 'zipcode']:
        if not address_fields.get(field): continue
        new_field = address_fields[field]
        addr_str = addr_str + ', ' + new_field if addr_str else new_field
    addr_str += ', USA'
    return addr_str

# Identical query strings recur heavily: repeated ads, and the candidate fan-out
# that pairs one street with every plausible city/state suffix. Geocoding each
# distinct query once cuts paid API calls substantially.
_CACHE_LOCK = threading.Lock()
_QUERY_CACHE = {}
# Queries that are currently being fetched.  The cache lookup alone is not
# enough under ThreadPoolExecutor: several ads can miss simultaneously and all
# issue the same paid request before the first response is stored.  Followers
# wait on the leader's event and reuse its result, including transient results
# that deliberately are not placed in the long-lived cache.
_INFLIGHT = {}
# Bounded so a 34M-ad run cannot accumulate every response in memory; entries are
# evicted oldest-first (dicts preserve insertion order).
CACHE_MAX_ENTRIES = 200000
_CACHE_HITS = [0]
# Status tally so a run that is quietly returning nothing (expired key, exhausted
# quota, blocked host) is visible in the log instead of finishing "successfully"
# with a column of nulls.
_STATUS_COUNTS = {}

def cached_request(request_func, query, biggest_nearby_cities, timeout=10):
    key = (request_func.__name__, query)
    with _CACHE_LOCK:
        if key in _QUERY_CACHE:
            _CACHE_HITS[0] += 1
            return _QUERY_CACHE[key]

        flight = _INFLIGHT.get(key)
        if flight is None:
            flight = {'event':threading.Event(), 'result':None, 'error':None}
            _INFLIGHT[key] = flight
            leader = True
        else:
            _CACHE_HITS[0] += 1
            leader = False

    if not leader:
        flight['event'].wait()
        if flight['error'] is not None:
            raise flight['error']
        return flight['result']

    try:
        result = request_func(query, biggest_nearby_cities, timeout=timeout)
        code = result[3].get('status_code')
        with _CACHE_LOCK:
            _STATUS_COUNTS[code] = _STATUS_COUNTS.get(code, 0) + 1
            # Only cache outcomes that a retry would not change: a timeout or
            # 5xx should be retried for the next ad rather than memoized as a
            # permanent failure. 401/402/403 are deliberately NOT cached.
            if code in (200, 400, 404):
                _QUERY_CACHE[key] = result
                while len(_QUERY_CACHE) > CACHE_MAX_ENTRIES:
                    _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
            flight['result'] = result
            _INFLIGHT.pop(key, None)
            flight['event'].set()
    except BaseException as error:
        # Always release followers, even for an interrupt or a malformed
        # response.  The exception is not cached, so a later call can retry.
        with _CACHE_LOCK:
            flight['error'] = error
            _INFLIGHT.pop(key, None)
            flight['event'].set()
        raise
    return result

def cache_stats():
    with _CACHE_LOCK:
        return len(_QUERY_CACHE), _CACHE_HITS[0]

def status_summary():
    ''' Requests by HTTP status, most frequent first (None = no response). '''
    with _CACHE_LOCK:
        counts = dict(_STATUS_COUNTS)
    return ', '.join('{}:{}'.format(k, v)
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or 'none'

def ok_share():
    ''' Share of requests that returned HTTP 200 (1.0 when none made yet). '''
    with _CACHE_LOCK:
        total = sum(_STATUS_COUNTS.values())
        return _STATUS_COUNTS.get(200, 0) / total if total else 1.0


def summarize(addresses:list, counties:list, zipcodes:list, US_DATA:object,
        prefix:str='geo'):
    ''' Aggregate an ad's per-candidate resolutions into its output columns.

    Shared by resolve() and by the offline recompute so the two cannot drift.
    Note mode() returns the first value on a tie, which is why candidate order
    is held deterministic upstream.
    '''
    zip_counties = US_DATA.counties_from_zips(zipcodes)
    return {
        '{}_addrs'.format(prefix): addresses,
        '{}_county'.format(prefix): mode(counties) if counties else None,
        '{}_zip_county'.format(prefix): mode(zip_counties or [None]),
    }


def resolve(address_dicts_list:list, US_DATA:object, nominatum=False, geoapify=True, verbose=False):
    output = {}

    if nominatum:
        nom_counties, nom_zipcodes, nom_addresses, nom_logs, nom_time = [], [], [], [], 0
    
    if geoapify:
        geo_counties, geo_zipcodes, geo_addresses, geo_logs, geo_time = [], [], [], [], 0
    
    for addr in address_dicts_list:
        query = format_str_address(addr)

        if nominatum:
            nst = time.time()
            time.sleep(1) # Avoid requests block
            address, county, zipcode, log = cached_request(nominatum_request,
                query, US_DATA.biggest_nearby_cities)
            nom_addresses.append(address)
            nom_logs.append(log)
            if county: nom_counties.append(county)
            if zipcode: nom_zipcodes.append(zipcode)
            nom_time += time.time() - nst

        if geoapify:
            gst = time.time()
            address, county, zipcode, log = cached_request(geoapify_request,
                query, US_DATA.biggest_nearby_cities)
            geo_addresses.append(address)
            geo_logs.append(log)
            if county: geo_counties.append(county)
            if zipcode: geo_zipcodes.append(zipcode)
            geo_time += time.time() - gst
           
    if len(address_dicts_list) > 0 and verbose:
        if nominatum:
            print("Nominatum API: {} seconds per request.".format(
                round(nom_time / len(address_dicts_list), 1)))
        if geoapify:
            print("GeoApify API: {} seconds per request.".format(
                round(geo_time / len(address_dicts_list), 1)))

    if geoapify:
        output.update(summarize(geo_addresses, geo_counties, geo_zipcodes, US_DATA))
        output['geo_requests'] = geo_logs
    if nominatum:
        output.update(summarize(nom_addresses, nom_counties, nom_zipcodes, US_DATA,
            prefix='nom'))
        output['nom_requests'] = nom_logs
    if nominatum and geoapify:
        output['same_county'] = output['nom_county'] == output['geo_county']
        output['same_zip_county'] = output['nom_zip_county'] == output['geo_zip_county']
    
    return output


def multithreading(func, addrs, geo, max_workers:int=None):
    with ThreadPoolExecutor(max_workers) as ex:
        res = ex.map(lambda x: func(x, geo), addrs)
    return list(res)


def validate_resume(nrows:int, batch_size:int, skip:int):
    '''Validate that a resolve run resumes only at a checkpoint boundary.'''
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if skip < 0 or skip > nrows:
        raise ValueError("skip must lie between 0 and the input row count")
    if skip % batch_size:
        raise ValueError("skip must be a whole multiple of batch_size")


def assign_batch_results(sample, results):
    '''Attach one resolved batch without retaining duplicate response payloads.'''
    locations = sample.index.get_indexer(results.index)
    if (locations < 0).any():
        raise ValueError("Resolved batch contains row ids outside the input sample.")
    for column in results.columns:
        if column not in sample.columns:
            sample[column] = pd.Series(index=sample.index, dtype='object')
        sample[column].to_numpy(copy=False)[locations] = results[column].to_numpy()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True, help="Filepath to extract output, e.g. ./output/NJG-extract-all.gzip")
    parser.add_argument('-n', '--nrows', type=int, default=None, help="Maximum number of ads.")
    parser.add_argument('-s', '--skip', type=int, default=0, help="Ads to skip at beginning.")
    # type=int, not type=bool: bool("0") is True.
    parser.add_argument('-m', '--multithreading', type=int, default=0, help="Use multithreads.")
    parser.add_argument('-w', '--nworkers', type=int, default=None, help="Number workers to use.")
    parser.add_argument('-b', '--batch_size', type=int, default=10000, help="Batch size.")
    parser.add_argument('-r', '--rate_limit', type=float, default=0,
        help="Max API requests per second across all threads (0 = unthrottled).")
    parser.add_argument('--min_pop', type=int, default=USGeoData.DEFAULT_MIN_POP,
        help="Minimum city population used when building address candidates.")
    parser.add_argument('--min_ok_share', type=float, default=0.05,
        help="Abort if the share of HTTP 200 responses falls below this (0 to disable).")
    parser.add_argument('-u', '--geoapify_url', type=str, default="https://api.geoapify.com",
        help="GeoApify URL endpoint to ping.")
    parser.add_argument('-a', '--aux_dir', type=str, help="Filepath to auxiliary files.",
        default='./auxiliary_files')
    parser.add_argument('-o', '--output_dir', type=str, help="Filepath to output directory.",
        default='./output')
    parser.add_argument('--write_csv', type=int, default=0,
        help="Also write a CSV copy containing source text and response payloads "
             "(default: disabled).")
    args = parser.parse_args()

    if not os.path.isdir(args.aux_dir):
        parser.error('Invalid filepath to auxiliary files.')
    if not os.path.isfile(args.filepath):
        parser.error('Invalid filepath to extract output.')
    # The output directory is ours to create; asserting on it only
    # made a first run fail on a path the user never chose.
    os.makedirs(args.output_dir, exist_ok=True)
    # Fail before the run rather than mid-job inside a worker thread.
    if not os.environ.get('GEOAPIFY_API_KEY'):
        parser.error('GEOAPIFY_API_KEY is not set in the environment.')

    print("Beginning geolocation validation.")

    newspaper = newspaper_from_path(args.filepath)

    sample = pd.read_parquet(args.filepath).iloc[:args.nrows]
    if not len(sample):
        raise SystemExit('Input data is empty.')
    if not sample.index.is_unique:
        raise SystemExit('Input row index contains duplicates.')
    try:
        validate_resume(len(sample), args.batch_size, args.skip)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if 'addresses' not in sample.columns or sample.addresses.isna().sum():
        raise SystemExit('Missing or null `addresses` values, exiting.')
    if not sample.addresses.map(lambda value: isinstance(value, (list, np.ndarray))).all():
        raise SystemExit('Every `addresses` value must be a list/array, exiting.')
    print("Will resolve sample of {} observations from {}.".format(len(sample), newspaper))

    # Load US geo-data
    US_DATA = USGeoData(
        os.path.join(args.aux_dir, "states.csv"),
        os.path.join(args.aux_dir, "geo/uscities.csv"),
        os.path.join(args.aux_dir, "neighbors-states.csv"),
        os.path.join(args.aux_dir, "geo/uszips.csv")
    ).load(newspaper, min_pop=args.min_pop)

    try:
        os.environ['GEOAPIFY_URL'] = validate_base_url(args.geoapify_url)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not 0 <= args.min_ok_share <= 1:
        raise SystemExit('--min_ok_share must lie between 0 and 1.')
    try:
        set_rate_limit(args.rate_limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print("Will make requests to GeoApify URL: '{}'{}".format(os.environ['GEOAPIFY_URL'],
        " at {} req/s".format(args.rate_limit) if args.rate_limit else ""))

    # Predict
    print("Beginning resolutions using {} threading ({} workers) at {}.".format(
        'multi' if args.multithreading else 'mono', args.nworkers or 1, time_now()))
    st_time = time.time()
    for batch_idx in range(ceil(len(sample) / args.batch_size)):
        if args.skip >= (batch_idx+1)*args.batch_size: continue
        batch = sample.addresses.iloc[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
        indices = sample.index[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
        if args.multithreading:
            counties_batch = pd.DataFrame(multithreading(resolve, batch.to_list(), 
                US_DATA, max_workers=args.nworkers), index=indices)
        else:
            counties_batch = pd.DataFrame(batch.apply(resolve, args=(US_DATA,)).to_list(),
                index=indices)
        # A checkpoint that cannot be written must stop the run: continuing would
        # lose that batch and also break merge-batch recovery afterwards, and the
        # unguarded final save below would fail the same way after days of work.
        counties_batch.to_parquet(add_filepath_suffix(args.output_dir, newspaper,
            n=(batch_idx+1)*args.batch_size, suffix='resolve-batch'), compression='gzip')
        assign_batch_results(sample, counties_batch)
        cached, hits = cache_stats()
        print("Processed ads {}-{} at {} ({} queries cached, {} calls saved; "
            "statuses {})...".format(
            batch_idx*args.batch_size,(batch_idx+1)*args.batch_size, time_now(),
            cached, hits, status_summary()))
        # A run that has stopped getting answers (expired key, exhausted quota,
        # blocked host) should stop burning through the corpus producing nulls.
        if ok_share() < args.min_ok_share:
            raise SystemExit("Aborting: only {:.1%} of requests returned HTTP 200 "
                "(statuses {}). Nothing after this point would resolve; batches "
                "written so far are intact.".format(ok_share(), status_summary()))
        
    # Full runs are named '-resolve-all' to match extract.py's convention (and
    # the README's statistics loop); a resumed run (--skip) carries NaN for the
    # skipped rows and so must not overwrite a complete file.
    out_n = args.nrows  # None -> 'all'
    out_suffix = 'resolve' if not args.skip else 'resolve-from{}'.format(args.skip)
    sample.to_parquet(add_filepath_suffix(args.output_dir, newspaper, n=out_n,
        suffix=out_suffix), compression='gzip')
    if args.write_csv:
        sample.to_csv(add_filepath_suffix(args.output_dir, newspaper, n=out_n,
            suffix=out_suffix, ext='csv'))
    elapsed = time.time() - st_time
    cached, hits = cache_stats()
    print("Completed resolutions at {} in {} minutes ({} seconds). "
        "{} queries cached, {} API calls saved by deduplication. "
        "Request statuses: {}.\n".format(
        time_now(), round(elapsed/60, 2), round(elapsed), cached, hits,
        status_summary()))
