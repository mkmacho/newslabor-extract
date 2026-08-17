import time
import os 
import argparse
import pandas as pd
from common import (TextWrapper, USGeoData, add_filepath_suffix, time_now,
    newspaper_from_path, first_ad)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class Newspaper(object):
    def __init__(self, newspaper, US_DATA, TEXT_HELP, min_pop=USGeoData.DEFAULT_MIN_POP):
        if newspaper not in US_DATA.NEWSPAPER_TO_STATE_ID:
            raise ValueError("Unsupported newspaper code: {!r}.".format(newspaper))
        self.newspaper = newspaper
        # Population is the largest geographic candidate restriction, so keep
        # the configured threshold explicit at load time.
        self.US_DATA = US_DATA.load(newspaper, min_pop=min_pop)
        self.TEXT_HELP = TEXT_HELP.set_state_abbreviations(
            self.US_DATA.US_STATES.state_id)
        print("Newspaper class loaded.")

    def first_ad(self, ad_text:str):
        ''' Keep only the first advertisement of a concatenated blob. '''
        return first_ad(ad_text)

    def city_state_options(self, tokens_list:list, idx:int):
        ''' Return possible city and state of address.
        If tokens following marker *seem like* potential city or state, include. 
        '''
        def get_possibles(arrays:list):
            possibles = []
            for arr in arrays:
                arr = ' '.join([word for word in arr if (len(word) > 1 and not word.isdigit())])
                if not arr: continue
                possibles.append(arr)
            # dict.fromkeys dedups while preserving order; list(set(...)) varied
            # with the process hash seed and made candidate order irreproducible.
            return list(dict.fromkeys(possibles))

        possible_cities = get_possibles([tokens_list[idx+1:idx+2],
                            tokens_list[idx+2:idx+3],tokens_list[idx+1:idx+3],
                            tokens_list[idx+2:idx+4]])
        possible_states = get_possibles([tokens_list[idx+1:idx+2],
                            tokens_list[idx+2:idx+3],tokens_list[idx+3:idx+4]])
        # Check for possible misspelled cities and states in post road marker tokens
        cities_dict_dict = self.US_DATA.check_nearby_cities(possible_cities)
        states_dict_dict = self.US_DATA.check_nearby_states(possible_states)
        return self.US_DATA.possible_city_state(self.US_DATA.state_name, 
            self.US_DATA.nearby_states, cities_dict_dict, states_dict_dict)

    def extract(self, ad_text, year=None):
        '''
        Extract possible address from tokens surrounding road markers.
        Main assumption is that for a road marker (e.g. "street") the idx token in token_list,
        token idx-2 will be the number, token idx-1 the street name, 
        token idx+1 the city, and token idx+2 the state. 

        Returns:
            address_dicts_list: list of structured addresses
        '''
        address_dicts_list = []
        if not ad_text: return address_dicts_list
        if not isinstance(ad_text, str):
            print('ERROR. Ad text supplied:', ad_text)
            return address_dicts_list

        tokens_list = self.TEXT_HELP.clean_tokenize(ad_text, self.newspaper)
        if not tokens_list: return address_dicts_list

        # Upon detecting street marker, form extracted geolocation
        for i, word in enumerate(tokens_list):
            if i == 0: continue
            if word.lower() in self.TEXT_HELP.STREET_MARKERS:
                prefix = self.TEXT_HELP.find_street(tokens_list, i)
                # Low-precision markers (see STREET_MARKERS_NEED_NUMBER) only
                # count when a house number precedes them, which separates
                # "509 Park Ct" from "Hartford Ct" and courthouse boilerplate.
                if word.lower() in self.TEXT_HELP.STREET_MARKERS_NEED_NUMBER \
                        and not prefix.get('housenumber'):
                    continue
                suffixes = self.city_state_options(tokens_list, i)
                if not suffixes:
                    raise RuntimeError("Address candidate has no geographic suffixes.")
                for suffix in suffixes:
                    address = prefix | suffix
                    if not address in address_dicts_list: 
                        address_dicts_list.append(address)

        # Complement that with zipcodes (which also lead directly to county).
        # Scoped to the first ad, as the street/city extraction above is:
        # scanning the whole blob attached later ads' zipcodes to this one.
        # Skipped entirely for ads predating ZIP codes (1963) — a 5-digit match
        # there is a price, a lot number or OCR noise, never a postal code.
        if year is not None and year < self.US_DATA.ZIPCODE_INTRODUCED:
            return address_dicts_list
        zipcode_objects = self.US_DATA.find_nearby_zipcodes(
            self.first_ad(ad_text), self.US_DATA.nearby_state_ids)
        zipcodes, added_zipcodes = sorted({z.zip for z in zipcode_objects}), []
        
        # For detected cities, check if detected zipcodes found in said cities
        for city_object in self.US_DATA.check_nearby_cities(tokens_list).values():
            for row in self.US_DATA.city_objects(city_object['name']):
                # sorted(): set iteration order varies with the process hash seed,
                # which made the emitted candidate order non-reproducible.
                for matched_zipcode in sorted(set(zipcodes) & set(row['zips'].split())):
                    added_zipcodes.append(matched_zipcode)
                    address_dicts_list.append({
                        'city':city_object['name'],
                        'state':row['state_name'],
                        'county':row['county_name'],
                        'zipcode':matched_zipcode}
                    )
        address_dicts_list.extend([{'zipcode':z} for z in zipcodes if z not in added_zipcodes])
        return address_dicts_list

    def employer_info(self, ad_text, sandbox=False, extract_employer=False):
        ''' Mirror extract, find *EMPLOYER NAME* and *OFFERED WAGE*.
        In theory would've done both at same time.
        '''
        # Seeded with the parsed-wage keys so that the three early returns below
        # (empty text, real-estate ad, empty cleaned text) still produce the full
        # column set; otherwise ~9% of records lacked them and pandas widened
        # wage_is_range from bool to object with NaNs mixed in.
        employer_dict = {'wage':None}
        employer_dict.update(self.TEXT_HELP.parse_wage(None))
        if sandbox: employer_dict.update(
            {'_wage_pred_strong':[],'_wage_pred_maybe':[],'_wage_pred_weak':[]})
        if not ad_text or not isinstance(ad_text, str): return employer_dict

        # Keep only first ad and skip over non-labor ads
        text = first_ad(ad_text)
        if any(term in text for term in self.TEXT_HELP.REAL_ESTATE) and not any(
            word in text for word in self.TEXT_HELP.NOT_RE): 
            return employer_dict

        # Try to extract employer name
        if extract_employer:
            employer_dict['employer'] = self.TEXT_HELP.extract_pos_employer(text)

        text = self.TEXT_HELP.clean_for_wage(text)
        if not text: return employer_dict

        tokens_list = text.split()
        best_candidates, potential_candidates, weak_candidates = [], [], []
        for i, word in enumerate(tokens_list):
            # When find potential salary, format
            if self.TEXT_HELP.potential_salary(word):
                best, potential, weak = self.TEXT_HELP.format_wage_candidate(tokens_list, i)
                if best: best_candidates.append(best)
                if potential: potential_candidates.append(potential)
                if weak: weak_candidates.append(weak)

        # Output best choice
        def choose_best_salary(options:list):
            salary = None
            if not options: return salary
            if len(options) > 1:
                options.sort(key=len)
                # Only keep larger substrings
                options = [wage for i, wage in enumerate(options) if not any(wage in opt for opt in options[i+1:])]
                for wage in options:
                    # Ensure best option has RATE
                    if any(rate in wage for rate in (self.TEXT_HELP.RATES_SINGLE | self.TEXT_HELP.RATES_DOUBLE)):
                        salary = wage
                        break
            return salary or options[0]
    
        employer_dict['wage'] = choose_best_salary(best_candidates or potential_candidates)
        # Parsed form for analysis: a number plus the period it is quoted per.
        employer_dict.update(self.TEXT_HELP.parse_wage(employer_dict['wage']))
        if sandbox:
            employer_dict['_wage_pred_strong'] = best_candidates
            employer_dict['_wage_pred_maybe'] = potential_candidates
            employer_dict['_wage_pred_weak'] = weak_candidates
        
        return employer_dict



_WORKER_NEWSPAPER = None

def _init_worker(newspaper:str, aux_dir:str,
        min_pop:int=USGeoData.DEFAULT_MIN_POP):
    ''' Build the Newspaper helper once per worker process.

    ProcessPoolExecutor pickles whatever callable it is given for *every* task;
    passing a bound method of Newspaper shipped the SymSpell dictionary and the
    US city table with each row, which is what exhausted memory on the cluster.
    '''
    global _WORKER_NEWSPAPER
    _WORKER_NEWSPAPER = build_newspaper(newspaper, aux_dir, min_pop=min_pop)

def _worker_extract(text_and_year):
    return _WORKER_NEWSPAPER.extract(*text_and_year)

def _worker_employer_info(ad_text):
    return _WORKER_NEWSPAPER.employer_info(ad_text)

def build_newspaper(newspaper:str, aux_dir:str, min_pop:int=USGeoData.DEFAULT_MIN_POP):
    ''' Load a Newspaper with its US-geo and text helpers from `aux_dir`. '''
    return Newspaper(
        newspaper=newspaper,
        min_pop=min_pop,
        US_DATA=USGeoData(
            os.path.join(aux_dir, "states.csv"),
            os.path.join(aux_dir, "geo/uscities.csv"),
            os.path.join(aux_dir, "neighbors-states.csv"),
            os.path.join(aux_dir, "geo/uszips.csv")
        ),
        TEXT_HELP=TextWrapper()
    )


def multiprocessing(func, args, max_workers:int=None, initializer=None,
        initargs=(), chunksize:int=1000):
    with ProcessPoolExecutor(max_workers, initializer=initializer,
            initargs=initargs) as ex:
        res = ex.map(func, args, chunksize=chunksize)
    return list(res)


def multithreading(func, args, max_workers:int=None):
    with ThreadPoolExecutor(max_workers) as ex:
        res = ex.map(func, args)
    return list(res)


def iter_batch_bounds(nrows:int, batch_size:int, skip:int=0):
    '''Yield (start, stop, checkpoint_stop) for resumable extraction batches.

    Checkpoint filenames retain the producer's regular batch boundary even for a
    short final batch. That is the convention merge-batch.py consumes. Resume is
    intentionally restricted to whole batches: accepting an arbitrary row offset
    would produce a checkpoint whose filename claims rows it does not contain.
    '''
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if skip < 0 or skip > nrows:
        raise ValueError("skip must lie between 0 and nrows")
    if skip % batch_size:
        raise ValueError("skip must be a whole multiple of batch_size")
    for start in range(0, nrows, batch_size):
        stop = min(start + batch_size, nrows)
        if stop <= skip:
            continue
        yield start, stop, start + batch_size


def extract_one_batch(sample, years, start:int, stop:int, newspaper,
        extract_address=True, extract_wage=False, executor=None,
        chunksize:int=1000):
    '''Compute the requested derived columns for one row batch.'''
    indices = sample.index[start:stop]
    raw = sample.raw_content.iloc[start:stop]
    out = pd.DataFrame(index=indices)

    if extract_address:
        args = list(zip(raw.to_list(), years[start:stop], strict=True))
        addresses = (list(executor.map(_worker_extract, args,
                         chunksize=chunksize)) if executor else
                     [newspaper.extract(text, year) for text, year in args])
        out['addresses'] = addresses

    if extract_wage:
        texts = raw.to_list()
        records = (list(executor.map(_worker_employer_info, texts,
                       chunksize=chunksize)) if executor else
                   [newspaper.employer_info(text) for text in texts])
        out = out.join(pd.DataFrame(records, index=indices))
    return out


def assign_batch_results(sample, results):
    '''Attach one derived batch without retaining a second full-size frame.'''
    locations = sample.index.get_indexer(results.index)
    if (locations < 0).any():
        raise ValueError("Batch results contain row ids outside the input sample.")
    for column in results.columns:
        if column not in sample.columns:
            sample[column] = pd.Series(index=sample.index, dtype='object')
        # Assign through the backing object array. Pandas 1.5 warns that the
        # future semantics of ``.loc[:, column] =`` will change; positional
        # assignment is both stable and O(batch) rather than copying the full
        # 34M-row column for every checkpoint.
        sample[column].to_numpy(copy=False)[locations] = results[column].to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True,
        help="Filepath to newspaper ads, e.g. ./test_data/NJG.csv")
    parser.add_argument('--extract_address', type=int, default=1)
    parser.add_argument('--extract_wage', type=int, default=0)
    parser.add_argument('-n', '--nrows', type=int, default=None, help="Maximum number of ads.")
    parser.add_argument('-m', '--multiprocessing', type=int, default=0, 
        help="Use multiprocessing.")
    parser.add_argument('-s', '--skip', type=int, default=0, help="Ads to skip at beginning.")
    parser.add_argument('-w', '--nworkers', type=int, default=None, help="Number workers to use.")
    parser.add_argument('-b', '--batch_size', type=int, default=100000, help="Batch size.")
    parser.add_argument('--min_pop', type=int, default=USGeoData.DEFAULT_MIN_POP,
        help="Minimum place population for a city to be a candidate. This is the "
             "largest sampling restriction in the design; see AUDIT.md.")
    parser.add_argument('-a', '--aux_dir', type=str, default='./auxiliary_files',
        help="Filepath to auxiliary directory.")
    parser.add_argument('-o', '--output_dir', type=str, default='./output',
        help="Filepath to output directory.")
    parser.add_argument('--write_csv', type=int, default=0,
        help="Also write a CSV copy containing source text (default: disabled).")
    args = parser.parse_args()

    if not os.path.isdir(args.aux_dir):
        parser.error('Invalid filepath to auxiliary files.')
    if not os.path.isfile(args.filepath):
        parser.error('Invalid filepath to data CSV.')
    if args.batch_size <= 0:
        raise SystemExit('--batch_size must be positive.')
    if args.skip < 0:
        raise SystemExit('--skip must be non-negative.')
    # The output directory is ours to create; asserting on it only
    # made a first run fail on a path the user never chose.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    sample = pd.read_csv(args.filepath, nrows=args.nrows, index_col=[0])
    if not len(sample):
        raise SystemExit('Input data is empty.')
    if not sample.index.is_unique:
        raise SystemExit('Input row index contains duplicates.')
    sample.raw_content = sample.raw_content.fillna('')
    print("Will process sample of {} observations.".format(len(sample)))

    # Load Newspaper class with helper classes
    paper = newspaper_from_path(args.filepath)
    NEWSPAPER = build_newspaper(paper, args.aux_dir, min_pop=args.min_pop)

    # Predict
    print("Beginning extractions using {} processing ({} workers) at {}.".format(
        'multi' if args.multiprocessing else 'serial', args.nworkers or 1, time_now()))
    start_time = time.time()
    args.batch_size = min(args.batch_size, len(sample))
    if args.skip > len(sample):
        raise SystemExit('--skip exceeds the input row count.')
    if args.skip % args.batch_size:
        raise SystemExit('--skip must be a whole multiple of --batch_size.')

    years = [None] * len(sample)
    if args.extract_address:
        print("Extracting addresses...")
        # `year` gates the zipcode scan (ZIP codes postdate 1963). Absent or
        # unparseable years leave the scan enabled, i.e. prior behaviour.
        if 'year' in sample.columns:
            years = pd.to_numeric(sample['year'], errors='coerce')
            years = [None if pd.isna(y) else int(y) for y in years]
            n_pre = sum(1 for y in years if y is not None
                        and y < NEWSPAPER.US_DATA.ZIPCODE_INTRODUCED)
            print("  {} of {} ads predate ZIP codes; skipping their zipcode scan.".format(
                n_pre, len(sample)))
        else:
            print("  No `year` column: zipcode scan enabled for all ads.")
            years = [None] * len(sample)
    if args.extract_wage:
        print("Extracting wages...")

    executor = None
    if args.multiprocessing and (args.extract_address or args.extract_wage):
        executor = ProcessPoolExecutor(max_workers=args.nworkers,
            initializer=_init_worker,
            initargs=(paper, args.aux_dir, args.min_pop))
    try:
        for batch_start, batch_stop, checkpoint_stop in iter_batch_bounds(
                len(sample), args.batch_size, args.skip):
            results = extract_one_batch(sample, years, batch_start, batch_stop,
                NEWSPAPER, extract_address=bool(args.extract_address),
                extract_wage=bool(args.extract_wage), executor=executor)
            if results.empty and not len(results.columns):
                continue
            results.to_parquet(add_filepath_suffix(args.output_dir, paper,
                n=checkpoint_stop, suffix='extract-batch'), compression='gzip')
            assign_batch_results(sample, results)
            print("Processed ads {}-{} at {}...".format(
                batch_start, batch_stop, time_now()))
    finally:
        if executor is not None:
            executor.shutdown()

    # Full runs recover native numeric/bool dtypes after the object-typed slots
    # used for incremental assignment. Partial resume outputs retain missing rows.
    if not args.skip:
        for column in sample.columns:
            if column not in ('addresses', 'wage'):
                sample[column] = sample[column].infer_objects()


    # A resumed run (--skip) holds NaN for the skipped rows, so it must not reuse
    # the filename of a complete run: merge its batch files instead.
    out_suffix = 'extract' if not args.skip else 'extract-from{}'.format(args.skip)
    sample.to_parquet(add_filepath_suffix(args.output_dir, paper, n=args.nrows,
        suffix=out_suffix), compression='gzip')
    if args.write_csv:
        sample.to_csv(add_filepath_suffix(args.output_dir, paper, n=args.nrows,
            suffix=out_suffix, ext='csv'))
    elapsed = time.time() - start_time
    print("Completed extractions at {} in {} minutes ({} seconds).".format(
        time_now(), round(elapsed / 60, 2), round(elapsed)))
