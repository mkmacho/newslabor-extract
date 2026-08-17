import os
import re
from importlib import resources
import pandas as pd
from pyzipcode import ZipCodeDatabase
from symspellpy import SymSpell
# from jamspell import TSpellCorrector
from thefuzz import process, fuzz
from spacy.lang.en.stop_words import STOP_WORDS
from string import capwords
from pytz import timezone
from datetime import datetime


def add_filepath_suffix(dirpath:str, newspaper:str, suffix:str='extract', n:int=None, ext:str='gzip'):
    filename = '{}-{}-{}.{}'.format(newspaper, suffix, str(n or 'all'), ext)
    filepath = os.path.join(dirpath, filename)
    print("Will save to '{}'.".format(filepath))
    return filepath

def time_now(tz:str='America/New_York'):
    return datetime.now(timezone(tz)).strftime("%m/%d/%Y %H:%M:%S")

def newspaper_from_path(filepath:str):
    ''' Newspaper abbreviation from a data filename.

    Handles both plain inputs ("NJG.csv") and derived ones
    ("NJG-extract-all.gzip"). Splitting on '-' alone left the extension attached
    for un-suffixed names, so a CSV template never matched its batch files.
    '''
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return stem.split('-')[0]

# Token separating concatenated advertisements inside one `raw_content` blob.
# It is stored WITHOUT a newspaper prefix ("._classifiedad_19791130_1"), and is a
# substring of any prefixed form, so splitting on it works for every paper.
# Splitting on "{newspaper}_classifiedad_" instead matched nothing at all in the
# NJG data (0 of 10,994 rows), so the address path silently scanned whole blobs.
AD_SEPARATOR = '_classifiedad_'

def first_ad(text:str):
    ''' Keep only the first advertisement of a concatenated blob.

    The separator is fused to the preceding token, so the first segment keeps a
    trailing fragment of that token; that is harmless for tokenised matching and
    is how the original wage path already behaved.
    '''
    return text.split(AD_SEPARATOR)[0] if text else text

def first_digit(word:str):
    for ch in word: 
        if ch.isdigit(): return ch
    return None

# Stop words that carry rate information. "a"/"an" are only kept when scanning
# *forward* from the amount, where RATES_DOUBLE ("a week", "an hour") is matched;
# keeping them in the backward scan only injected stray tokens into the emitted
# string (e.g. "salary $12,307 a 858").
RATE_STOP_WORDS = set(["per", "every"])
RATE_STOP_WORDS_PREFIX = RATE_STOP_WORDS | set(["a", "an"])

def _wage_candidate_array(tokens, start, end, prefix=True):
    keep = RATE_STOP_WORDS_PREFIX if prefix else RATE_STOP_WORDS
    candidate_arr = [token.lower() for token in tokens[start:end] if \
        token.lower() not in (STOP_WORDS - keep)]
    if prefix and "hours" in candidate_arr: 
        return None # signifies schedule, not wage
    if len(candidate_arr) == 1: # If all stop words minus wage
        assert candidate_arr[0] == tokens[start if prefix else end-1]
        return None
    if prefix and candidate_arr[1] in ['dollars', 'cash'] and end+1 <= len(tokens):
        candidate_arr.append(tokens[end].lower())
    return candidate_arr


class TextWrapper(object):
    """Text cleaning backed by SymSpell's packaged English dictionary.

    ``symspellpy`` ships the frequency dictionary used by its own examples. Using
    that installed resource keeps the lexicon tied to the pinned package without
    redistributing a separate frequency dictionary. An explicit path remains
    available for controlled private runs.
    """

    DEFAULT_DICTIONARY = "frequency_dictionary_en_82_765.txt"

    def __init__(self, dictionary_filepath=None):
        self.checker = SymSpell()
        if dictionary_filepath is None:
            dictionary_resource = resources.files("symspellpy").joinpath(
                self.DEFAULT_DICTIONARY)
            with resources.as_file(dictionary_resource) as dictionary_path:
                loaded = self.checker.load_dictionary(dictionary_path, 0, 1)
            dictionary_label = "symspellpy:{}".format(self.DEFAULT_DICTIONARY)
        else:
            loaded = self.checker.load_dictionary(dictionary_filepath, 0, 1)
            dictionary_label = str(dictionary_filepath)
        assert loaded, "SymSpell dictionary not loaded from '{}'.".format(
            dictionary_label)
        self.dictionary = self.checker.words
        assert self.dictionary, "SymSpell dictionary loaded but empty."
        self.dictionary_source = dictionary_label
        self.CARDINAL_DIRECTIONS = ["east","e","west","w","north","n","south","s"]
        self.REAL_ESTATE = ["decorated","refurbish","remodel","bedroom","bathroom", 
                         "tenant","furniture","deluxe","furnish","apartment",
                         "realtor","realty","garage","backyard","vacant","for sale"]
        self.STREET_MARKERS_ABBREV = ["rd","blvd","st","ct","ave","av","ln","pl"]
        # NOTE: "dr" is deliberately excluded — it collides with the "Dr." title.
        self.STREET_MARKERS_FULL = ["road","boulevard","street","circuit","avenue",
                                    "lane","court","drive","place"]
        self.STREET_MARKERS = self.STREET_MARKERS_ABBREV + self.STREET_MARKERS_FULL
        # Markers that are common English words or state abbreviations ("ct" is
        # also Connecticut; "court"/"circuit" appear in courthouse boilerplate).
        # Measured on NJG, only 4-22% of their matches carry a house number,
        # against 61-93% for every other marker, so they are accepted only with
        # one. "circuit" was already in the list and is the worst of them at 5%.
        self.STREET_MARKERS_NEED_NUMBER = {"ct", "pl", "ln", "court", "place", "circuit"}
        # Short tokens that must survive clean_tokenize's length filter regardless
        # of whether they appear in the English dictionary. Without this, markers
        # like "rd"/"ct" (absent from the frequency dictionary) were deleted before
        # STREET_MARKERS was ever consulted. State abbreviations are added by
        # set_state_abbreviations() once the US state table is available.
        self.SHORT_KEEP = set(self.STREET_MARKERS_ABBREV) | set(self.CARDINAL_DIRECTIONS)
        self.NUMBERS_SUFFIX = {"1":"st", "2":"nd", "3":"rd"}
        self.WAGE_MARKERS = {"salary","sal","pays","pay","payment","rate","start",
                                "starting","earn","begins","beginning"}
        self.RATES_DOUBLE = {"per annum","per year","per yr","a year","a yr",
                           "per mo","a mo","a month","per month",
                           "per week","per wk","a week","a wk","every week","every wk",
                           "per day","a day","every day",
                           "per hr","per hour","an hour","an hr"}
        self.RATES_SINGLE = {"annually","yearly","monthly","weekly","daily","hourly"}
        self.TIMES = {'hour','week','day','daily','month','year'}
        self.TIMES_ABBREV = {'hr','wk','mo','yr'}
        self.NOT_RE = ["hiring", "salary", "equal opportunity", "employer", "employee"]
        print("Loaded text functions.")

    def _correct_street(self, addr:list):    
        # Spell check   
        corrected = self._correct_sentence(' '.join(addr)).split() or addr
        assert corrected, "Original: '{}' and corrected: '{}'.".format(addr, corrected)
        # Correct numbered street
        if corrected[-1][0].isdigit():
            ndigits = len([d for d in corrected[-1] if d.isdigit()])
            # If majority, assume supposed to be, e.g. '5th'
            if ndigits / len(corrected[-1]) > 0.5:
                corrected[-1] = corrected[-1][:ndigits]
                corrected[-1] += self.NUMBERS_SUFFIX.get(corrected[-1][-1], "th")
        return capwords(' '.join(corrected))

    def _correct_sentence(self, words:str, edit_dist=2, ignore_non_words=False):
        return self.checker.lookup_compound(words, split_by_space=ignore_non_words,
                    max_edit_distance=edit_dist, ignore_non_words=ignore_non_words,
                    ignore_term_with_digits=ignore_non_words)[0].term

    def set_state_abbreviations(self, state_ids):
        ''' Register US state abbreviations as short tokens worth keeping.

        Two-letter abbreviations such as "tx", "md", "dc" are absent from the
        English dictionary and were otherwise dropped by clean_tokenize, making
        abbreviation-based state detection unreachable for those states.
        '''
        self.SHORT_KEEP |= {str(sid).lower() for sid in state_ids}
        return self

    def _is_word(self, word:str):
        return word.lower() in self.dictionary or word.title() in self.dictionary

    def potential_salary(self, word:str):
        # Decimal part is a single optional group: a bare '\d{1,2}?' still
        # requires one digit, which rejected all single-digit wages ("$8 a day").
        if not re.findall(r'^\$?\d+(?:[.,]\d{1,2})?\$?[-\s]', word + " "):
            return False
        if first_digit(word) == '0': 
            return False
        if re.findall(r'\d{0,3}-?\s?\d{3}-?\s?\d{4}', word):
            return False
        return True

    def clean_tokenize(self, text:str, newspaper:str, exclude_RE:bool=True, min_token_length:int=3):
        ''' Basic ad text cleaning. Firstly ensures that we consider only
        first ad, then removes punctuation and extra whitespace. 
        '''
        first = first_ad(text)
        if exclude_RE and any(re in first for re in self.REAL_ESTATE): return None
        cleaned = re.sub(' +', ' ', re.sub(r'[^\w\s]', ' ', first)).strip().split()
        return [token for token in cleaned if (len(token) >= min_token_length or
                self._is_word(token) or token.isdigit() or token.lower() in self.SHORT_KEEP)]

    def extract_pos_employer(self, text):
        ''' TODO: find employer names from text. '''
        employers = []
        # orgs_nlp = [ent.text for ent in nlp_large(text).ents if ent.label_ == 'ORG'] 
        # orgs_wiki = [ent.text for ent in nlp_wiki(text).ents if ent.label_ == 'ORG' and not 
        #                 any(ent.text in org for org in orgs_nlp)]
        # employers = [o for o in orgs_nlp if not any(o in org for org in orgs_wiki)] + orgs_wiki
        return employers

    def format_wage_from_number_words(self, tokens:list, idx:int):
        ''' TODO: turn words, e.g. 'one hundred a week' into output. '''
        # from word2number import w2n
        return None

    def format_wage_candidate(self, tokens:list, idx:int):
        best_candidate = potential_candidate = weak_candidate = None
        # First, try to find rate (e.g. hourly) following potential salary
        for i in range(idx+2, idx+4):
            if i > len(tokens): continue
            candidate_arr = _wage_candidate_array(tokens, idx, i)
            if not candidate_arr: continue
            candidate = ' '.join(candidate_arr)
            # Case when e.g. "$500 WEEKLY" or e.g. "$500 PER WEEK"
            if (i == idx+2 and candidate_arr[-1] in self.RATES_SINGLE) or \
                (i == idx+3 and ' '.join(candidate_arr[-2:]) in self.RATES_DOUBLE):
                # SymSpell can turn an OCR-damaged street name immediately
                # before "St" into a rate word (for example, "3296 Berkly St"
                # became "3296 weekly st"). A bare number followed by that
                # synthetic rate is an address, not pay. Dollar-marked amounts
                # remain eligible because they carry independent wage evidence.
                next_token = tokens[idx+2].lower() if idx+2 < len(tokens) else None
                if '$' not in tokens[idx] and next_token in self.STREET_MARKERS:
                    break
                if '$' in tokens[idx]: 
                    best_candidate = candidate
                else: 
                    potential_candidate = candidate
                break
            # Case when e.g. "$50 hour"
            if any(time in candidate_arr for time in (self.TIMES | self.TIMES_ABBREV)):
                if '$' in tokens[idx]: 
                    potential_candidate = potential_candidate or candidate
                else: 
                    weak_candidate = candidate
                break
        # Second, if prior text indicates a salary (though no rate) consider
        for i in range(idx-1, idx-4, -1):
            if i < 0: continue
            candidate_arr = _wage_candidate_array(tokens, i, idx+1, prefix=False)
            if not candidate_arr: continue
            candidate = ' '.join(candidate_arr)
            if candidate_arr[0] in self.WAGE_MARKERS:
                potential = candidate
                if idx+2 < len(tokens):
                    if tokens[idx+2] in (self.TIMES | self.TIMES_ABBREV):
                        potential = ' '.join(tokens[i:idx+3])
                elif idx+1 < len(tokens):
                    if tokens[idx+1] in (self.TIMES | self.TIMES_ABBREV):
                        potential = ' '.join(tokens[i:idx+2])
                # A bare "<marker> <amount>" (no rate, no '$') is only credible
                # when the amount has two or more digits: allowing single digits
                # here promotes noise like "pay 1" / "salary 2" to an output wage.
                marker_and_amount = len(candidate_arr) == 2 and \
                    sum(ch.isdigit() for ch in tokens[idx]) > 1
                if '$' in potential or marker_and_amount:
                    potential_candidate = potential_candidate or potential
                else:
                    weak_candidate = weak_candidate or potential
        # Finally, if have dollar wage consider (weak)
        if '$' in tokens[idx]: 
            weak_candidate = weak_candidate or tokens[idx]
        return best_candidate, potential_candidate, weak_candidate

    # Rate words -> the period the amount is quoted per. Used by parse_wage.
    WAGE_PERIODS = {
        'hour':'hour', 'hr':'hour', 'hourly':'hour',
        'day':'day', 'daily':'day',
        'week':'week', 'wk':'week', 'weekly':'week',
        'month':'month', 'mo':'month', 'monthly':'month',
        'year':'year', 'yr':'year', 'yearly':'year', 'annually':'year', 'annum':'year',
    }

    def parse_wage(self, wage:str):
        ''' Split an extracted wage string into (amount, period, is_range).

        The pipeline emits strings such as "$60 per hour"; analysis needs a
        number and a period. Returns None for amount and/or period when the
        string does not carry them, rather than guessing.

        `amount` is the FIRST number in the string (the low end of a range).

        `is_range` requires an explicit range separator as well as a second
        number: OCR routinely splits one amount into two tokens ("$9 75 hour" is
        $9.75, not a range), and keying only on "more than one number" made 63%
        of the flags false positives on real output.

        `wage_n_amounts` exposes how many numbers were seen, so a caller can
        exclude ambiguous strings without re-parsing. Amounts are unreliable
        where OCR split a thousands group; see AUDIT.md for the measured rate.
        '''
        out = {'wage_amount':None, 'wage_period':None, 'wage_is_range':False,
               'wage_n_amounts':0}
        if not wage or not isinstance(wage, str):
            return out
        text = wage.lower()
        # Amounts: allow thousands separators and decimals.
        amounts = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        if amounts:
            try:
                out['wage_amount'] = float(amounts[0].replace(',', ''))
            except ValueError:
                out['wage_amount'] = None
            out['wage_n_amounts'] = len(amounts)
            out['wage_is_range'] = len(amounts) > 1 and bool(
                re.search(r'\d\s*(?:-|–|to\b)\s*\$?\d', text))
        for token in re.findall(r'[a-z]+', text):
            if token in self.WAGE_PERIODS:
                out['wage_period'] = self.WAGE_PERIODS[token]
                break
        return out

    def clean_for_wage(self, text:str):
        # Addl spaces
        spaces = ' ' + re.sub(' {2,}', ' ', text).strip() + ' '
        # Consecutive digits
        d = re.sub(r'(?<=\s\d)\s+(?=\d+\s)', '', spaces)
        d = re.sub(r'(?<=\s\d\d)\s+(?=\d+\s)', '', d)
        d = re.sub(r'(?<=\s\d\d\d)\s+(?=\d+\s)', '', d)
        d = re.sub(r'(\s\$?\s?\d+)\s?(\d+\$?\s)', r'\1\2', d)
        # Decimals
        x = re.sub(r'(\s\$?\s?\d+)\s?(\.|,)\s?(\d{1,3}\$?\s)', r'\1\2\3', d)
        # Dollar digits
        x = re.sub(r'\s[s|t|f|F|S|\$]\s?(\d+[,|\.]?\d*)\$?\s',r' $\1 ', x)
        x = re.sub(r'\s(\d+[,|\.]?\d*)[s|t|f|F|S|\$]\s',r' \1$ ', x)
        # Colons
        x = re.sub(r'\s-\s|\s-\$?\d+|\d+-\s', '-', x)
        # Extra punctuation
        punct = x.translate(str.maketrans('', '', '!"#%&\'()*+/:;<>?@[\\]^_`{|}~'))
        punct = re.sub(r'\s\.\s|\s,\s|\s-|-\s',' ',punct)
        return self._correct_sentence(punct.lower(), ignore_non_words=True)

    def find_street(self, tokens_list:str, idx:int):
        ''' Return (house)number and street.

        Arguments
            idx: index of street marker
        Returns:
            dict: containing 'housenumber', 'street' fields
        '''
        addr = tokens_list[max(idx-3,0):idx]
        assert addr, "Address malformed: '{}'".format(addr)
        assert all(comp for comp in addr), "Address malformed: '{}'".format(addr)
        marker = capwords(tokens_list[idx])
        if marker == "Av": marker = "Ave"

        if addr[0][0] == '0':
            if len(addr[0]) == 1: 
                addr.pop(0)
            else: 
                addr[0] = addr[0][1:]
        if not addr: return {}
        if len(addr) == 3:
            if addr[0][0].isdigit():
                if not (addr[1][0].isdigit() or addr[2][0].isdigit()): 
                    pass  # e.g. "100 This That"
                elif addr[1].lower() in self.CARDINAL_DIRECTIONS:
                    pass # e.g. "100 E 4th"
                else:
                    addr.pop(0)
            elif addr[0].lower() in self.CARDINAL_DIRECTIONS: 
                pass   # e.g. East North London
            else: 
                addr.pop(0)
        while len(addr) > 1:
            if addr[0][0].isdigit() or (addr[0].lower() in self.CARDINAL_DIRECTIONS and 
                    addr[1] not in STOP_WORDS): 
                break
            addr.pop(0)
        
        structured = {}
        if len(addr) > 1 and addr[0][0].isdigit():
            number = addr.pop(0)
            structured['housenumber'] = ''.join([d for d in number if d.isdigit()])

        assert addr, "Address malformed: '{}'".format(addr)
        structured['street'] = self._correct_street(addr) + ' ' + marker
        return structured

    def find_street_markers(self, text:str, short_thresh:int=100, long_thresh:int=80):
        ''' Identifies possible street markers. '''
        street_tokens = []
        matches = process.extract(text, self.STREET_MARKERS_FULL, scorer=fuzz.partial_ratio)
        for match in matches:
            if match[1] >= long_thresh:
                street_tokens.append(match[0])
        matches = process.extract(text, self.STREET_MARKERS_ABBREV, scorer=fuzz.token_set_ratio)
        for match in matches:
            if match[1] >= short_thresh:
                street_tokens.append(match[0])
        return street_tokens 


class USGeoData(object):
    def __init__(self, states_fp, cities_fp, nearby_fp, zips_fp=None):
        # Database of US states and state abbreviations
        self.US_STATES = pd.read_csv(states_fp).rename(
            {"State":"state_name","Abbreviation":"state_id"}, axis='columns')
        # Derived city/county reference built from Census and GeoNames sources.
        self.US_CITIES = pd.read_csv(cities_fp, dtype={'county_fips':str})[
            ['city','state_id','state_name','county_name','county_fips','zips','population']
        ]
        self.US_CITIES.zips = self.US_CITIES.zips.fillna('')
        # Neighboring state IDs mapping
        self.NEIGHBOR_STATES = pd.read_csv(nearby_fp).rename(
            {"StateCode":"state_id","NeighborStateCode":"neighbor_id"}, axis='columns')
        self.NEWSPAPER_TO_STATE_ID = {"ASA":"TX","ATC":"GA","ATL":"GA","BaS":"MD",
            "BoG":"MA","ChT":"IL","HaC":"CT","LAS":"CA","LAT":"CA","NJG":"VA",
            "NYr":"NY","NYT":"NY","WaP":"DC"}
        # City-name indexes, built once. Both replace repeated full-table scans
        # in the per-advertisement path; iteration follows the source file so the
        # emitted candidate order is unchanged.
        self.CITY_INDEX, self.CITY_STATES = {}, {}
        for r in self.US_CITIES.itertuples(index=False):
            self.CITY_INDEX.setdefault(r.city, []).append({
                'city': r.city, 'state_id': r.state_id, 'state_name': r.state_name,
                'county_name': r.county_name, 'county_fips': r.county_fips,
                'zips': r.zips if isinstance(r.zips, str) else '',
                'population': r.population})
            states = self.CITY_STATES.setdefault(r.city, [])
            if r.state_name not in states:
                states.append(r.state_name)

        self.ZIPCODE_DB = ZipCodeDatabase()
        # ZIP-to-county lookups are built from both derived reference tables.
        # `uszips.csv` is the primary mapping; its Census component covers ZCTAs,
        # while GeoNames fallback rows and the city table retain some PO-box-only
        # and unique codes (Norfolk's 23501 among them). City rows are applied
        # first and `uszips.csv` is overlaid wherever both carry the code.
        # County FIPS is the joinable key analysis actually needs: county *names*
        # are not unique nationally (1,910 names span 3,207 name+state pairs).
        self.ZIP_TO_COUNTY, self.ZIP_TO_FIPS = {}, {}
        # Where two city rows claim the same zipcode, the MORE SPECIFIC row wins:
        # rows are written in order of decreasing zip-list length, so the shortest
        # (most specific) list is applied last. Ranking by population instead let
        # a consolidated "New York" row — population 18.9M, 308 zips,
        # county recorded as Queens — overwrite 96 Manhattan zipcodes that the
        # borough's own row labels correctly. Benchmarked against uszips, this
        # ordering agrees on 95.9% of shared zips versus 95.2% for population.
        by_specificity = self.US_CITIES.assign(
            _nzips=self.US_CITIES.zips.fillna('').str.split().apply(len)
        ).sort_values(['_nzips', 'population'], ascending=[False, True])
        for row in by_specificity.itertuples():
            if not isinstance(row.zips, str): continue
            for z in row.zips.split():
                self.ZIP_TO_COUNTY[z] = row.county_name
                if isinstance(row.county_fips, str):
                    self.ZIP_TO_FIPS[z] = row.county_fips.zfill(5)
        if zips_fp:
            uszips = pd.read_csv(zips_fp, usecols=['zip','county_name','county_fips'],
                dtype={'zip':str, 'county_name':str, 'county_fips':str})
            zips5 = uszips.zip.str.zfill(5)
            self.ZIP_TO_COUNTY.update(dict(
                zip(zips5, uszips.county_name, strict=True)))
            have_fips = uszips.county_fips.notna()
            self.ZIP_TO_FIPS.update(dict(zip(zips5[have_fips],
                uszips.county_fips[have_fips].str.zfill(5), strict=True)))
        print("Loaded USA geo-data.")

    # Population threshold for a place to be a candidate city, expressed in ACS
    # *place* population rather than metropolitan-area population. A higher gate
    # can sharply tighten the largest sampling restriction in the design, so the
    # 15,000 default remains explicit and configurable. See AUDIT.md.
    DEFAULT_MIN_POP = 15000

    def load(self, newspaper:str, min_pop=DEFAULT_MIN_POP):
        # The compute_* methods are deliberately named apart from the attributes
        # they fill: assigning results onto the method names made load() a
        # one-shot operation (a second call raised "'list' object is not callable").
        self.state_id = self.NEWSPAPER_TO_STATE_ID[newspaper]
        self.state_name = self.state_id_to_state_name(self.state_id)
        self.nearby_state_ids = self.compute_nearby_state_ids(self.state_id)
        self.nearby_states = self.nearby_state_names(self.nearby_state_ids)
        self.biggest_nearby_cities = self.compute_biggest_nearby_cities(
            self.nearby_state_ids, min_pop=min_pop)
        # Deterministic iteration order for fuzzy matching (see above).
        self.sorted_nearby_cities = sorted(self.biggest_nearby_cities)
        # Fuzzy matching is the dominant remaining cost once the city lookups are
        # indexed, and ad vocabulary repeats heavily across a corpus, so results
        # are memoised per token. Cleared here because the city set it depends on
        # is rebuilt by every load().
        self._city_match_memo = {}
        print("Loaded newspaper-state data.")
        return self

    def counties_from_zips(self, zipcodes:list):
        ''' Map each zipcode to its county.

        Returns one county per resolvable zipcode so the caller can select the
        modal county across every mapped postcode.
        '''
        if not zipcodes: return None
        normalized = [str(z).strip()[:5] for z in zipcodes if z]
        counties = [self.ZIP_TO_COUNTY[z] for z in normalized
                    if z in self.ZIP_TO_COUNTY]
        return counties or None

    def fips_from_zips(self, zipcodes:list):
        ''' Map each zipcode to its 5-digit county FIPS code.

        Unlike a county name this is a unique, joinable identifier, so it is the
        column downstream analysis should merge on.
        '''
        if not zipcodes or not self.ZIP_TO_FIPS: return None
        codes = [self.ZIP_TO_FIPS[z] for z in
                 (str(x).strip()[:5] for x in zipcodes if x)
                 if z in self.ZIP_TO_FIPS]
        return codes or None

    # US ZIP codes were introduced on 1 July 1963; a 5-digit match in an earlier
    # ad is necessarily something else (a price, a lot number, OCR noise).
    ZIPCODE_INTRODUCED = 1963

    def state_id_to_state_name(self, state_id:str):
        assert state_id in self.US_STATES.state_id.to_list()
        return self.US_STATES.state_name[self.US_STATES.state_id == state_id].iloc[0]

    def compute_nearby_state_ids(self, state_id:str):
        ''' Adjacent (and home newspaper) state IDs, i.e. abbreviations.

        neighbors-states.csv stores each adjacency ONCE, alphabetically ordered
        ("DC,MD" but never "MD,DC"), so selecting only rows whose state_id matches
        returned just the alphabetically-later neighbours: Virginia's neighbours
        were WV alone, losing DC, KY, MD, NC and TN. Both directions are unioned
        here so the adjacency is symmetric, as the name and the README imply.
        '''
        forward = self.NEIGHBOR_STATES.loc[
            self.NEIGHBOR_STATES.state_id == state_id].neighbor_id.to_list()
        reverse = self.NEIGHBOR_STATES.loc[
            self.NEIGHBOR_STATES.neighbor_id == state_id].state_id.to_list()
        # sorted() keeps the ordering deterministic across runs
        return sorted(set(forward) | set(reverse) | {state_id})

    def nearby_state_names(self, nearby_state_ids:list):
        return self.US_STATES.loc[
            self.US_STATES.state_id.isin(nearby_state_ids)].state_name.to_list()

    def big_cities_in_state(self, state_name:str, min_pop:int=DEFAULT_MIN_POP):
        return self.US_CITIES[(self.US_CITIES.state_name == state_name) & (
            self.US_CITIES.population >= min_pop)].sort_values(
                by=['population'], ascending=False).city.to_list()

    def compute_biggest_nearby_cities(self, nearby_state_ids:list, min_pop:int=DEFAULT_MIN_POP):
        ''' Return the set of biggest cities in given states.

        A set is returned for O(1) membership tests, but it must never be
        *iterated* for matching: set iteration order depends on Python's
        per-process string hash seed, which made candidate ordering (and hence
        mode() tie-breaking on the resolved county) vary between runs and
        between worker processes. Use `sorted_nearby_cities` for iteration.
        '''
        biggest_cities = []
        for state_id in nearby_state_ids:
            biggest_cities.extend(self.US_CITIES[(self.US_CITIES.state_id == state_id) &
                    (self.US_CITIES.population >= min_pop)].city.to_list())
        return set(biggest_cities)

    def find_nearby_zipcodes(self, text:str, nearby_state_ids:list):
        ''' Matches 5-digit to plausible (nearby-state) zipcodes. '''
        # Lookarounds rather than consuming boundary characters: "23501 23502"
        # otherwise yielded only the first zipcode.
        zips = re.findall(r"(?<!\d)(\d{5})(?!\d)", " " + text + " ")
        return [self.ZIPCODE_DB[z] for z in zips if (self.ZIPCODE_DB.get(z) and 
            self.ZIPCODE_DB[z].state in nearby_state_ids)]

    def city_objects(self, city:str):
        ''' Rows for a city name, as plain dicts in source-file order.

        CITY_INDEX is built once at load so candidate lookup avoids repeated
        full-table string comparisons.
        '''
        return self.CITY_INDEX.get(city, ())

    def states_for_city(self, city:str):
        ''' State names in which a city name occurs. O(1) index lookup. '''
        return self.CITY_STATES.get(city, ())

    def possible_city_state(self, state_name:str, nearby_states:list, cities_dict_dict:dict, states_dict_dict:dict):
        ''' Return possible city and state of address.
        If tokens following marker *seem like* potential city or state, include.
        '''
        suffixes = []
        added_city_state = False
        for city_object in cities_dict_dict.values():
            added_city = False
            city_states = self.states_for_city(city_object['name'])
            for state in nearby_states:
                if state in city_states:
                    suffixes.append({'city':city_object['name'], 'state':state})
                    added_city = True
                    added_city_state = True
            if not added_city:
                suffixes.append({'city':city_object['name']})
        if not added_city_state:
            for state in list(states_dict_dict.keys()) + [state_name]:
                if not any(state == suffix['state'] for suffix in suffixes):
                    suffixes.append({'state':state})
        return suffixes

    def check_nearby_cities(self, tokens:list, threshold:int=70):
        ''' Given list of potential cities, return possible true cities. 

        Returns
            matches: dict from words in tokens to dicts of correct word and confidence
        '''
        matches = {}
        for token in tokens:
            assert token, tokens
            # exact match by priority
            if token.title() in self.biggest_nearby_cities:
                matches[token] = {'name':token.title(), 'conf':100}
                continue
            # Otherwise take the best probable match; process.extract is sorted
            # by descending confidence.
            if not self.sorted_nearby_cities: continue
            (city, score) = self._best_city_match(token.title())
            if score >= threshold:
                matches[token] = {'name':city, 'conf':score}
        return matches

    # Bounded so a 34M-ad run cannot retain every distinct token; ad vocabulary
    # is far smaller than this in practice, so the cap is a safety net rather
    # than a working constraint.
    CITY_MEMO_MAX = 300000

    def _best_city_match(self, title_token:str):
        ''' Memoised best fuzzy city match for one token.

        The candidate list is fixed for the loaded newspaper, so the answer
        depends only on the token — and tokens recur constantly across a corpus.
        '''
        hit = self._city_match_memo.get(title_token)
        if hit is None:
            hit = process.extractOne(title_token, self.sorted_nearby_cities,
                scorer=fuzz.ratio)
            if len(self._city_match_memo) >= self.CITY_MEMO_MAX:
                self._city_match_memo.clear()
            self._city_match_memo[title_token] = hit
        return hit

    def check_nearby_states(self, tokens_list:list, name_thresh:int=80, id_thresh:int=90):
        ''' Given list of potential states, return possible true states
        as dict of dicts mapping state name to state name and confidence. 
        '''
        matches = {}
        for token in tokens_list:
            assert token, tokens_list
            # exact (nearby state) matches
            if token.title() in set(self.nearby_states):
                matches[token.title()] = {'name':token.title(),'conf':100,'type':'name'}
            if token.upper() in set(self.nearby_state_ids):
                token_name = self.state_id_to_state_name(token.upper())
                if not token_name in matches: 
                    matches[token_name] = {'name':token_name,'conf':100,'type':'id'}
            # probable matches
            (state, score) = process.extractOne(token.title(), self.nearby_states, scorer=fuzz.ratio)
            if score >= name_thresh and not state in matches: 
                matches[state] = {'name':state,'conf':score,'type':'name'}
            (abbrev, score) = process.extractOne(token.upper(), self.nearby_state_ids, scorer=fuzz.ratio)
            if score >= id_thresh: 
                abbrev_name = self.state_id_to_state_name(abbrev)
                if not abbrev_name in matches: 
                    matches[abbrev_name] = {'name':abbrev_name,'conf':score,'type':'id'}
        return matches
