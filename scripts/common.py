import os
import re
import pandas as pd
from statistics import mode
from pyzipcode import ZipCodeDatabase
from nltk import ConditionalFreqDist, pos_tag, word_tokenize
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

def first_digit(word:str):
    for ch in word: 
        if ch.isdigit(): return ch
    return None

def _wage_candidate_array(tokens, start, end, prefix=True):
    candidate_arr = [token.lower() for token in tokens[start:end] if \
        token.lower() not in (STOP_WORDS - set(["per", "every"]))]
    if prefix and "hours" in candidate_arr: 
        return None # signifies schedule, not wage
    if len(candidate_arr) == 1: # If all stop words minus wage
        assert candidate_arr[0] == tokens[start if prefix else end-1]
        return None
    if prefix and candidate_arr[1] in ['dollars', 'cash'] and end+1 <= len(tokens):
        candidate_arr.append(tokens[end].lower())
    return candidate_arr


class TextWrapper(object):
    def __init__(self, dictionary_filepath):
        self.checker = SymSpell()
        self.checker.load_dictionary(dictionary_filepath, 0, 1)
        assert self.checker, "SymSpell not loaded."
        self.dictionary = self.checker.words
        self.CARDINAL_DIRECTIONS = ["east","e","west","w","north","n","south","s"]
        self.REAL_ESTATE = ["decorated","refurbish","remodel","bedroom","bathroom", 
                         "tenant","furniture","deluxe","furnish","apartment",
                         "realtor","realty","garage","backyard","vacant","for sale"]
        self.STREET_MARKERS_ABBREV = ["rd","blvd","st","ct","ave","av"]
        self.STREET_MARKERS_FULL = ["road","boulevard","street","circuit","avenue","lane"]
        self.STREET_MARKERS = self.STREET_MARKERS_ABBREV + self.STREET_MARKERS_FULL
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

    def _is_word(self, word:str):
        return word.lower() in self.dictionary or word.title() in self.dictionary

    def potential_salary(self, word:str):
        if not re.findall('^\$?\d+\.?\d{1,2}?\$?[-\s]', word + " "):
            return False       
        if first_digit(word) == '0': 
            return False
        if re.findall('\d{0,3}-?\s?\d{3}-?\s?\d{4}', word): 
            return False
        return True

    def clean_tokenize(self, text:str, newspaper:str, exclude_RE:bool=True, min_token_length:int=3):
        ''' Basic ad text cleaning. Firstly ensures that we consider only
        first ad, then removes punctuation and extra whitespace. 
        '''
        first = text.split("{}_classifiedad_".format(newspaper))[0]
        if exclude_RE and any(re in first for re in self.REAL_ESTATE): return None
        cleaned = re.sub(' +', ' ', re.sub(r'[^\w\s]', ' ', first)).strip().split()
        return [token for token in cleaned if (len(token) >= min_token_length or 
                self._is_word(token) or token.isdigit() or token.lower() in self.CARDINAL_DIRECTIONS)]

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
                    if any(tokens[idx+2] in time for time in self.TIMES) or \
                        tokens[idx+2] in self.TIMES_ABBREV:
                        potential = ' '.join(tokens[i:idx+3])
                elif idx+1 < len(tokens):
                    if any(tokens[idx+1] in time for time in self.TIMES) or \
                        tokens[idx+1] in self.TIMES_ABBREV:
                        potential = ' '.join(tokens[i:idx+2])
                if '$' in potential or len(candidate_arr) == 2: 
                    potential_candidate = potential_candidate or potential
                else: 
                    weak_candidate = weak_candidate or potential
        # Finally, if have dollar wage consider (weak)
        if '$' in tokens[idx]: 
            weak_candidate = weak_candidate or tokens[idx]
        return best_candidate, potential_candidate, weak_candidate

    def clean_for_wage(self, text:str):
        # Addl spaces
        spaces = ' ' + re.sub(' {2,}', ' ', text).strip() + ' '
        # Consecutive digits
        d = re.sub('(?<=\s\d)\s+(?=\d+\s)', '', spaces)
        d = re.sub('(?<=\s\d\d)\s+(?=\d+\s)', '', d)
        d = re.sub('(?<=\s\d\d\d)\s+(?=\d+\s)', '', d)
        d = re.sub('(\s\$?\s?\d+)\s?(\d+\$?\s)', r'\1\2', d)
        # Decimals
        x = re.sub('(\s\$?\s?\d+)\s?(\.|,)\s?(\d{1,3}\$?\s)', r'\1\2\3', d)
        # Dollar digits
        x = re.sub('\s[s|t|f|F|S|\$]\s?(\d+[,|\.]?\d*)\$?\s',r' $\1 ', x)
        x = re.sub('\s(\d+[,|\.]?\d*)[s|t|f|F|S|\$]\s',r' \1$ ', x)
        # Colons
        x = re.sub('\s-\s|\s-\$?\d+|\d+-\s', '-', x)
        # Extra punctuation
        punct = x.translate(str.maketrans('', '', '!"#%&\'()*+/:;<>?@[\\]^_`{|}~'))
        punct = re.sub('\s\.\s|\s,\s|\s-|-\s',' ',punct)
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
        matches = process.extract(text, STREET_MARKERS_FULL, scorer=fuzz.partial_ratio)
        for match in matches:
            if match[1] >= long_thresh:
                street_tokens.append(match[0])
        matches = process.extract(text, STREET_MARKERS_ABBREV, scorer=fuzz.token_set_ratio)
        for match in matches:
            if match[1] >= short_thresh:
                street_tokens.append(match[0])
        return street_tokens 

    def find_tags(self, tag_prefix:str, tagged_text:list):
        ''' Find tokens matching the specified tag_prefix. '''
        cfd = ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                      if tag.startswith(tag_prefix))
        return dict((tag, list(cfd[tag].keys())) for tag in cfd.conditions())


class USGeoData(object):
    def __init__(self, states_fp, cities_fp, nearby_fp):
        # Database of US states and state abbreviations
        self.US_STATES = pd.read_csv(states_fp).rename(
            {"State":"state_name","Abbreviation":"state_id"}, axis='columns')
        # Database of US cities and city-level information from SimpleMaps
        self.US_CITIES = pd.read_csv(cities_fp)[
            ['city','state_id','state_name','county_name','zips','population']
        ]
        self.US_CITIES.zips = self.US_CITIES.zips.fillna('')
        # Neighboring state IDs mapping
        self.NEIGHBOR_STATES = pd.read_csv(nearby_fp).rename(
            {"StateCode":"state_id","NeighborStateCode":"neighbor_id"}, axis='columns')
        self.NEWSPAPER_TO_STATE_ID = {"ASA":"TX","ATC":"GA","ATL":"GA","BaS":"MD",
            "BoG":"MA","ChT":"IL","HaC":"CT","LAS":"CA","LAT":"CA","NJG":"VA",
            "NYr":"NY","NYT":"NY","WaP":"DC"}
        self.ZIPCODE_DB = ZipCodeDatabase()
        print("Loaded USA geo-data.")

    def load(self, newspaper:str, min_pop=50000):
        self.state_id = self.NEWSPAPER_TO_STATE_ID[newspaper] 
        self.state_name = self.state_id_to_state_name(self.state_id)
        self.nearby_state_ids = self.nearby_state_ids(self.state_id)
        self.nearby_states = self.nearby_state_names(self.nearby_state_ids)
        self.biggest_nearby_cities = self.biggest_nearby_cities(
            self.nearby_state_ids, min_pop=min_pop)
        print("Loaded newspaper-state data.")
        return self

    def counties_from_zips(self, zipcodes:list):
        if not zipcodes: return None
        return list(self.US_CITIES.loc[self.US_CITIES.zips.str.contains(
            mode(zipcodes)), 'county_name'].values) 

    def state_id_to_state_name(self, state_id:str):
        assert state_id in self.US_STATES.state_id.to_list()
        return self.US_STATES.state_name[self.US_STATES.state_id == state_id].iloc[0]

    def nearby_state_ids(self, state_id:str):
        # Adjacent (and home newspaper) state IDs (i.e. abbreviations)
        return self.NEIGHBOR_STATES.loc[
            self.NEIGHBOR_STATES.state_id == state_id].neighbor_id.to_list() + [state_id]

    def nearby_state_names(self, nearby_state_ids:list):
        return self.US_STATES.loc[
            self.US_STATES.state_id.isin(nearby_state_ids)].state_name.to_list()

    def big_cities_in_state(self, state_name:str, min_pop:int=50000):
        return self.US_CITIES[(self.US_CITIES.state_name == state_name) & (
            self.US_CITIES.population >= min_pop)].sort_values(
                by=['population'], ascending=False).city.to_list()

    def biggest_nearby_cities(self, nearby_state_ids:list, min_pop:int=50000):
        ''' Return list of biggest cities in given states. '''
        biggest_cities = []
        for state_id in nearby_state_ids:
            biggest_cities.extend(self.US_CITIES[(self.US_CITIES.state_id == state_id) & 
                    (self.US_CITIES.population >= min_pop)].city.to_list())
        return set(biggest_cities)

    def find_nearby_zipcodes(self, text:str, nearby_state_ids:list):
        ''' Matches 5-digit to plausible (nearby-state) zipcodes. '''
        zips = re.findall(r"\D(\d{5})\D", " " + text + " ")
        return [self.ZIPCODE_DB[z] for z in zips if (self.ZIPCODE_DB.get(z) and 
            self.ZIPCODE_DB[z].state in nearby_state_ids)]

    def city_objects(self, city:str):
        return self.US_CITIES[self.US_CITIES.city == city]

    def possible_city_state(self, state_name:str, nearby_states:list, cities_dict_dict:dict, states_dict_dict:dict):
        ''' Return possible city and state of address.
        If tokens following marker *seem like* potential city or state, include. 
        '''
        suffixes = []
        added_city_state = False
        for city_object in cities_dict_dict.values():
            added_city = False
            for state in nearby_states:
                if state in self.US_CITIES[
                        self.US_CITIES.city == city_object['name']].state_name.to_list():
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
            # otherwise probable matches
            cities = process.extract(token.title(), self.biggest_nearby_cities, 
                scorer=fuzz.ratio, limit=5)
            for (city, score) in cities:
                if score < threshold: break
                matches[token] = {'name':city, 'conf':score}
        return matches

    def check_nearby_states(self, tokens_list:list, name_thresh:int=80, id_thresh:int=90):
        ''' Given list of potential states, return possible true states
        as dict of dicts mapping state name to state name and confidence. 
        '''
        matches = {}
        for token in tokens_list:
            assert token, tokens
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

