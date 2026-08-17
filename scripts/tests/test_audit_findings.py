"""Regression tests for the pipeline's documented correctness safeguards.

Each test asserts current intended behavior at a boundary where silent errors
would materially change extracted research variables.

Run from the repo root:
    python -m pytest scripts/tests/test_audit_findings.py -v

No network access: the Geoapify test stubs out the HTTP layer.
"""
import math
import os
import sys

import pandas as pd
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
AUX_DIR = os.path.join(REPO_ROOT, "auxiliary_files")
sys.path.insert(0, SCRIPTS_DIR)

import common  # noqa: E402
import build_geo_reference  # noqa: E402
import resolve  # noqa: E402
from extract import Newspaper  # noqa: E402


@pytest.fixture(scope="module")
def us_data():
    return common.USGeoData(
        os.path.join(AUX_DIR, "states.csv"),
        os.path.join(AUX_DIR, "geo/uscities.csv"),
        os.path.join(AUX_DIR, "neighbors-states.csv"),
    ).load("NJG")


@pytest.fixture(scope="module")
def us_data_with_zips():
    """Loaded the way the scripts load it in production — with uszips.csv."""
    return common.USGeoData(
        os.path.join(AUX_DIR, "states.csv"),
        os.path.join(AUX_DIR, "geo/uscities.csv"),
        os.path.join(AUX_DIR, "neighbors-states.csv"),
        os.path.join(AUX_DIR, "geo/uszips.csv"),
    ).load("NJG")


@pytest.fixture(scope="module")
def text_help():
    return common.TextWrapper()


def test_packaged_dictionary_is_loaded(text_help):
    """The public build uses symspellpy's versioned, MIT-licensed resource."""
    assert text_help.dictionary_source == (
        "symspellpy:frequency_dictionary_en_82_765.txt")
    assert len(text_help.dictionary) == 82_834


@pytest.fixture(scope="module")
def newspaper(text_help):
    us = common.USGeoData(
        os.path.join(AUX_DIR, "states.csv"),
        os.path.join(AUX_DIR, "geo/uscities.csv"),
        os.path.join(AUX_DIR, "neighbors-states.csv"),
    )
    return Newspaper("NJG", us, text_help)


def test_a1_geoapify_keeps_best_confidence_feature(us_data, monkeypatch):
    """A1: the highest-confidence qualifying feature must win, not the last one."""
    payload = {
        "features": [
            {"properties": {"rank": {"confidence": 0.95}, "city": "Norfolk",
                            "county": "Norfolk County", "postcode": "23501",
                            "formatted": "45 Ocean St, Norfolk, VA"}},
            {"properties": {"rank": {"confidence": 0.30}, "city": "Chesapeake",
                            "county": "Chesapeake County", "postcode": "23320",
                            "formatted": "45 Ocean St, Chesapeake, VA"}},
        ]
    }
    monkeypatch.setenv("GEOAPIFY_URL", "https://api.example.invalid")
    monkeypatch.setenv("GEOAPIFY_API_KEY", "dummy")
    monkeypatch.setattr(
        resolve, "get_wrapper",
        lambda url, timeout=10: {"url": url, "status_code": 200, "content": payload,
                                 "elapsed": 0.0, "message": "", "type": "Response"},
    )
    address, county, zipcode, _ = resolve.geoapify_request(
        "45 Ocean St, Norfolk, Virginia, USA", us_data.biggest_nearby_cities)
    assert county == "Norfolk", (
        "geoapify_request returned the county of a lower-confidence feature "
        "(best_conf is never updated)")
    assert zipcode == "23501"
    assert address == "45 Ocean St, Norfolk, VA"


@pytest.fixture(scope="module")
def us_data_lat():
    return common.USGeoData(
        os.path.join(AUX_DIR, "states.csv"),
        os.path.join(AUX_DIR, "geo/uscities.csv"),
        os.path.join(AUX_DIR, "neighbors-states.csv"),
    ).load("LAT")


def test_a2_fuzzy_city_match_keeps_best_candidate(us_data_lat):
    """A2: a misspelled city must map to its best fuzzy match, not the worst >= 70.

    On the LAT city set, 'santa clar' scores [Santa Clara 95, Santa Clarita 87,
    Santa Cruz 80, Santa Maria 76, Santa Ana 74]; pre-fix code keeps Santa Ana.
    (NJG's 25-city set has no pair similar enough to trigger the overwrite,
    which is why this test uses LAT.)
    """
    matches = us_data_lat.check_nearby_cities(["santa clar"])
    assert matches, "expected a fuzzy match for 'santa clar'"
    assert matches["santa clar"]["name"] == "Santa Clara", (
        "check_nearby_cities kept a lower-scoring candidate: "
        f"{matches['santa clar']}")


def test_a3_tokenizer_keeps_rd_and_ct_street_markers(text_help):
    """A3: 'rd' and 'ct' must survive tokenization so STREET_MARKERS can match."""
    tokens = [t.lower() for t in text_help.clean_tokenize(
        "Cook wanted apply 45 Ocean Rd Norfolk", "NJG")]
    assert "rd" in tokens, "'rd' was dropped by clean_tokenize (not in dictionary)"
    tokens = [t.lower() for t in text_help.clean_tokenize(
        "Maid wanted 1010 Park Ct Norfolk", "NJG")]
    assert "ct" in tokens, "'ct' was dropped by clean_tokenize (not in dictionary)"


def test_a3_rd_address_is_extracted(newspaper):
    """A3 end-to-end: an address on a 'Rd' street must be extractable."""
    addresses = newspaper.extract("Cook wanted apply 45 Ocean Rd Norfolk")
    streets = [a.get("street") for a in addresses if a.get("street")]
    assert any("Ocean" in s for s in streets), (
        f"no street extracted from a 'Rd' address; got {addresses}")


def test_a4_zipcodes_scoped_to_first_ad(newspaper):
    """A4: a zipcode appearing only in a later concatenated ad must not attach
    to the first ad's extraction.

    Uses the REAL separator. The blobs store it without a newspaper prefix and
    fused to the previous token ("...sold 13 ._classifiedad_19791130_1 CAL..."),
    so an earlier version of this test that split on "NJG_classifiedad_" passed
    while the behaviour it names occurred on none of the 10,994 NJG rows.
    """
    blob = ("HELP WANTED cook apply 45 Ocean Street Norfolk "
            "13 ._classifiedad_19791130_1 FOR SALE house lovely Norfolk 23502 call now")
    addresses = newspaper.extract(blob)
    zips = {a.get("zipcode") for a in addresses if a.get("zipcode")}
    assert "23502" not in zips, (
        "zipcode from the second ad leaked into the first ad's addresses")


def test_a4_separator_matches_the_real_data(newspaper):
    """A4: guard the separator itself against the prefixed form regressing in."""
    assert common.AD_SEPARATOR == "_classifiedad_"
    real = "cook wanted 45 Ocean Street Norfolk 13 ._classifiedad_19791130_1 tail text"
    assert newspaper.first_ad(real) != real, (
        "first_ad did not split on the separator the data actually uses")
    # a prefixed form still splits, since the bare token is a substring of it
    prefixed = "cook wanted NYT_classifiedad_19701219_198 tail text"
    assert newspaper.first_ad(prefixed) != prefixed


def test_a6_single_digit_wage_is_potential_salary(text_help):
    """A6: single-digit dollar amounts ('$8 a day') must qualify as salaries."""
    assert text_help.potential_salary("$8"), "single-digit wage rejected"
    assert text_help.potential_salary("$50"), "regression: two-digit wage"
    assert text_help.potential_salary("$5.50"), "regression: decimal wage"
    assert not text_help.potential_salary("757-555-1234"), \
        "regression: phone numbers must stay rejected"


def test_a6_a_week_rate_is_best_candidate(text_help):
    """A6: '$500 a week' must be recognized via RATES_DOUBLE as a *best*
    candidate ('a' is stripped as a stop word pre-fix)."""
    tokens = "pay $500 a week steady".split()
    best, potential, weak = text_help.format_wage_candidate(tokens, 1)
    assert best is not None and "week" in best, (
        f"'$500 a week' not recognized as best candidate: "
        f"best={best!r} potential={potential!r} weak={weak!r}")


def test_a6_substring_time_inversion(text_help):
    """A6: tokens that are substrings of time words ('our' in 'hour') must not
    be treated as rate words when extending a wage candidate."""
    tokens = "salary 500 for our crew".split()
    best, potential, weak = text_help.format_wage_candidate(tokens, 1)
    for cand in (best, potential, weak):
        assert cand is None or "our" not in cand.split(), (
            f"substring time-match extended the wage with 'our': {cand!r}")


def test_packaged_dictionary_does_not_turn_street_number_into_wage(text_help):
    """A spell-corrected street name beside ``St`` is not wage evidence.

    The packaged SymSpell dictionary corrects the OCR-like ``Berkly`` to
    ``weekly``. The following street number therefore looked exactly like a
    non-dollar weekly wage during the dictionary transition.
    """
    tokens = text_help.clean_for_wage(
        "seamstress apply 3296 Berkly St Norfolk").split()
    assert tokens[2:5] == ["3296", "weekly", "st"]
    assert text_help.format_wage_candidate(tokens, 2) == (None, None, None)

    # A dollar sign is independent wage evidence, even if an address follows.
    marked = "$500 weekly St Louis".split()
    best, potential, _ = text_help.format_wage_candidate(marked, 0)
    assert best == "$500 weekly"
    assert potential is None


def test_geo_builder_rejects_state_specific_truncated_names():
    """Builder canaries reject old LSAD bugs without rejecting real places."""
    norfolk_zips = "23501 23506 23514 23515 23519 23529 23541"
    rows = [
        ("Washington", "DC", "11001", "District of Columbia", ""),
        ("Norfolk", "VA", "51710", "Norfolk", norfolk_zips),
        ("Jersey City", "NJ", "34017", "Hudson", "07302"),
        ("Panama City", "FL", "12005", "Bay", "32401"),
        ("Winston-Salem", "NC", "37067", "Forsyth", "27101"),
        # These short names are legitimate Census places outside the states
        # where repeated suffix stripping formerly created false rows.
        ("Jersey", "GA", "13297", "Walton", "30014"),
        ("Panama", "IA", "19165", "Shelby", "51562"),
        ("Winston", "OR", "41019", "Douglas", "97496"),
    ]
    cities = pd.DataFrame(rows, columns=[
        "city", "state_id", "county_fips", "county_name", "zips"])
    zips = pd.DataFrame([("23501", "51710")],
                        columns=["zip", "county_fips"])
    assert build_geo_reference.sanity_check(cities, zips) == []

    broken = pd.concat([
        cities,
        pd.DataFrame([("Jersey", "NJ", "34017", "Hudson", "07302")],
                     columns=cities.columns),
    ], ignore_index=True)
    assert build_geo_reference.sanity_check(broken, zips) == [
        "name normalisation introduced truncated 'Jersey' in NJ"]


def test_b6_adjacent_zipcodes_both_found(us_data):
    """B6: two space-adjacent zipcodes must both be extracted."""
    found = us_data.find_nearby_zipcodes(
        "Norfolk VA 23501 23502", us_data.nearby_state_ids)
    zips = sorted(z.zip for z in found)
    assert zips == ["23501", "23502"], (
        f"adjacent zipcode lost to boundary consumption: {zips}")


def test_b7_find_street_markers_callable(text_help):
    """B7: find_street_markers must not NameError (bare globals pre-fix)."""
    try:
        text_help.find_street_markers("apply 45 ocean street norfolk")
    except NameError as err:
        pytest.fail(f"find_street_markers raised NameError: {err}")


def test_b1_check_nearby_states_assert_message(us_data):
    """B1: the assert's failure path must not itself NameError."""
    with pytest.raises(AssertionError):
        us_data.check_nearby_states([""])


def test_b1_nominatim_error_dict_is_not_fatal(monkeypatch):
    """B1: a Nominatim error payload (a dict, not a list) must yield no result
    rather than raising and killing the worker's whole batch."""
    monkeypatch.setattr(
        resolve, "get_wrapper",
        lambda url, timeout=10: {"url": url, "status_code": 200,
                                 "content": {"error": "Unable to geocode"},
                                 "elapsed": 0.0, "message": "", "type": "Response"},
    )
    address, county, zipcode, _ = resolve.nominatum_request("nowhere, USA", {"Norfolk"})
    assert (address, county, zipcode) == (None, None, None)


def test_d1_identical_queries_are_geocoded_once(us_data, monkeypatch):
    """D1: repeated candidate queries must cost a single API call."""
    calls = []

    def fake_request(query, cities, timeout=10):
        calls.append(query)
        return ("1 Main St, Norfolk, VA", "Norfolk", "23501",
                {"status_code": 200, "url": "stub"})

    resolve._QUERY_CACHE.clear()
    monkeypatch.setattr(resolve, "geoapify_request", fake_request)
    addrs = [{"street": "Main Street", "housenumber": "1", "city": "Norfolk",
              "state": "Virginia"}] * 3
    out = resolve.resolve(addrs, us_data)
    assert len(calls) == 1, f"expected 1 API call for 3 identical queries, got {len(calls)}"
    assert len(out["geo_addrs"]) == 3, "every ad-level address must still get a result"
    assert out["geo_county"] == "Norfolk"


def test_a12_extraction_is_reproducible_across_hash_seeds():
    """A12: candidate ordering must not depend on Python's hash seed.

    `biggest_nearby_cities` is a set; iterating it for fuzzy matching made the
    emitted candidate order vary per process, which in turn changed mode()
    tie-breaking on the resolved county.
    """
    import subprocess
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from extract import build_newspaper\n"
        "paper = build_newspaper('NJG', %r)\n"
        "print(paper.extract('cook wanted apply 1151 Dune Street Norfolk Virginia'))\n"
        % (SCRIPTS_DIR, AUX_DIR)
    )
    outputs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        assert res.returncode == 0, res.stderr
        outputs.append(res.stdout.strip().splitlines()[-1])
    assert outputs[0] == outputs[1], (
        "extraction output differs between hash seeds:\n"
        f"seed 1: {outputs[0]}\nseed 2: {outputs[1]}")


# --- second-pass findings (post-audit review) ---------------------------------

def test_p1_neighbor_states_are_symmetric(us_data):
    """P1: neighbors-states.csv lists each adjacency once, alphabetically ordered,
    so selecting only forward rows gave Virginia just WV — dropping DC, MD, NC,
    KY and TN from every NJG candidate address and from the accepted-city filter.
    """
    va = set(us_data.compute_nearby_state_ids("VA"))
    assert {"DC", "MD", "NC", "KY", "TN", "WV", "VA"} <= va, (
        "VA neighbours are not symmetric: {}".format(sorted(va)))
    # adjacency must be mutual for every newspaper's home state
    for paper, sid in us_data.NEWSPAPER_TO_STATE_ID.items():
        for other in us_data.compute_nearby_state_ids(sid):
            if other == sid:
                continue
            assert sid in us_data.compute_nearby_state_ids(other), (
                "{} lists {} as a neighbour but not vice versa".format(sid, other))


def test_p2_geoapify_reads_all_fields_from_the_winning_feature(us_data, monkeypatch):
    """P2: fields must come from the single best feature. Assigning each field
    independently dropped a county that only a lower-confidence feature carried,
    and could mix address/county/postcode across different features.
    """
    payload = {"features": [
        {"properties": {"rank": {"confidence": 0.95}, "city": "Norfolk",
                        "postcode": "23510", "formatted": "A st, Norfolk"}},
        {"properties": {"rank": {"confidence": 0.60}, "city": "Norfolk",
                        "county": "Norfolk County", "postcode": "23511",
                        "formatted": "B st, Norfolk"}},
    ]}
    monkeypatch.setenv("GEOAPIFY_URL", "https://api.example.invalid")
    monkeypatch.setenv("GEOAPIFY_API_KEY", "dummy")
    monkeypatch.setattr(resolve, "get_wrapper",
        lambda url, timeout=10: {"url": url, "status_code": 200, "content": payload,
                                 "elapsed": 0.0, "message": "", "type": "Response"})
    address, county, zipcode, _ = resolve.geoapify_request("q", us_data.biggest_nearby_cities)
    assert (address, zipcode) == ("A st, Norfolk", "23510"), \
        "fields must come from the highest-confidence feature"
    assert county is None, \
        "county must not be borrowed from a different (lower-confidence) feature"


def test_p3_auth_failures_are_not_cached(us_data, monkeypatch):
    """P3: caching a 401 would memoize a permanent negative for every subsequent
    query, so a key that expires mid-run yields a 'successful' run of nulls."""
    calls = []

    def failing(query, cities, timeout=10):
        calls.append(query)
        return (None, None, None, {"status_code": 401, "url": "stub"})

    resolve._QUERY_CACHE.clear()
    monkeypatch.setattr(resolve, "geoapify_request", failing)
    for _ in range(3):
        resolve.cached_request(resolve.geoapify_request, "same query", {"Norfolk"})
    assert len(calls) == 3, "401 responses must not be cached"


def test_p4_all_failed_batch_is_still_serializable():
    """P4: a batch in which every request failed must still checkpoint. An empty
    dict for `content` has no inferable struct schema and pyarrow refuses it —
    which, now that checkpoint failure is fatal, would end a multi-day job."""
    import tempfile
    failed = {'url': 'u', 'elapsed': 10, 'content': None, 'message': 'timeout',
              'status_code': None, 'type': 'ReadTimeout'}
    frame = pd.DataFrame({'geo_addrs': [[None]], 'geo_requests': [[failed]]})
    with tempfile.NamedTemporaryFile(suffix='.gzip') as fh:
        frame.to_parquet(fh.name, compression='gzip')  # must not raise
        assert pd.read_parquet(fh.name).shape == (1, 2)


def test_p5_low_precision_markers_require_a_housenumber(newspaper):
    """P5: 'ct'/'court'/'place'/'circuit' match courthouse boilerplate and the
    Connecticut abbreviation. Measured on NJG only 4-22% of their matches carry a
    house number (61-93% for every other marker), so they need one."""
    # "Hartford Ct" is the state, not a street
    addrs = newspaper.extract("Secretary wanted apply 24 Main St Hartford Ct")
    streets = [a.get("street") for a in addrs if a.get("street")]
    assert not any(s.endswith("Ct") for s in streets), (
        f"bare 'Ct' accepted as a street marker: {streets}")
    assert any("Main" in s for s in streets), f"lost the real street: {streets}"
    # a numbered Court address is still extracted
    addrs = newspaper.extract("Cook wanted apply 509 Park Court Norfolk")
    assert any("Park" in (a.get("street") or "") for a in addrs), (
        f"numbered Court address was dropped: {addrs}")


# --- offline (no-API) capability ----------------------------------------------

def _feature(conf, city, county=None, postcode=None, formatted=None):
    props = {"rank": {"confidence": conf}, "city": city}
    if county: props["county"] = county
    if postcode: props["postcode"] = postcode
    if formatted: props["formatted"] = formatted
    return {"properties": props}


def _stored(features, status=200):
    """A response log shaped like the ones resolve.py persists to geo_requests."""
    return {"url": "https://api.geoapify.com/v1/geocode/search?text=x&apiKey=REDACTED",
            "status_code": status, "content": {"features": features} if status == 200 else None,
            "message": "", "elapsed": 0.2, "type": "Response"}


def test_o1_recompute_recovers_best_match_offline(us_data):
    """O1: re-scoring a STORED response must apply the corrected selection, so
    the county the buggy code got wrong is fixable with no new API call."""
    import recompute
    stored = [_stored([
        _feature(0.95, "Norfolk", "Norfolk County", "23501", "45 Ocean St"),
        _feature(0.30, "Chesapeake", "Chesapeake County", "23320", "45 Ocean St"),
    ])]
    out = recompute.recompute_from_requests(stored, us_data)
    assert out["rc_county"] == "Norfolk", out
    assert out["rc_usable_responses"] == 1


def test_o2_recompute_handles_failed_and_empty_responses(us_data):
    """O2: stored failures must yield no county rather than raising."""
    import recompute
    for stored in ([_stored([], status=None)], [], None, [_stored([])]):
        out = recompute.recompute_from_requests(stored, us_data)
        assert out["rc_county"] is None
        assert out["rc_addrs"] == [] or all(a is None for a in out["rc_addrs"])


def test_o3_counties_derived_from_addresses_without_any_api(us_data):
    """O3: candidates carrying a zipcode already imply a county, so those ads
    never needed a geocoder at all."""
    import recompute
    addrs = [{"city": "Norfolk", "state": "Virginia", "county": "Norfolk",
              "zipcode": "23501"}]
    out = recompute.counties_from_addresses(addrs, us_data)
    assert out["offline_county"] == "Norfolk"
    assert out["offline_n_zipcodes"] == 1
    # a bare zipcode with no county field still resolves through uszips
    out = recompute.counties_from_addresses([{"zipcode": "23501"}], us_data)
    assert out["offline_zip_county"], "zipcode did not resolve to a county"


def test_o4_zip_to_fips_is_joinable(us_data_with_zips):
    """O4: county names are not unique nationally, so the offline path must be
    able to emit a FIPS code."""
    fips = us_data_with_zips.fips_from_zips(["23501"])
    assert fips and len(fips[0]) == 5 and fips[0].isdigit(), fips


def test_o5_pre_1963_ads_get_no_zipcode(newspaper):
    """O5: ZIP codes were introduced in 1963; a 5-digit match in a 1940s ad is
    a price or OCR noise, and it feeds the offline county path directly."""
    text = "COOK wanted apply 45 Ocean Street Norfolk 23501"
    assert any(a.get("zipcode") for a in newspaper.extract(text, year=1975)), \
        "post-1963 ad lost its zipcode"
    assert not any(a.get("zipcode") for a in newspaper.extract(text, year=1941)), \
        "pre-1963 ad was given a ZIP code that could not have existed"
    # unknown year keeps prior behaviour
    assert any(a.get("zipcode") for a in newspaper.extract(text))


def test_o6_wage_parses_to_amount_and_period(text_help):
    """O6: analysis needs a number and a period, not a string."""
    assert text_help.parse_wage("$60 per hour") == {
        "wage_amount": 60.0, "wage_period": "hour", "wage_is_range": False,
        "wage_n_amounts": 1}
    assert text_help.parse_wage("$800 a month")["wage_period"] == "month"
    assert text_help.parse_wage("salary $12,307")["wage_amount"] == 12307.0
    # ranges are flagged, not silently read as a point value
    assert text_help.parse_wage("$7.50-$9.00")["wage_is_range"] is True
    assert text_help.parse_wage("$5 to $9 hour")["wage_is_range"] is True
    # ...but an OCR-split single amount is NOT a range ("$9 75 hour" = $9.75)
    assert text_help.parse_wage("$9 75 hour")["wage_is_range"] is False
    assert text_help.parse_wage("$9 75 hour")["wage_n_amounts"] == 2
    # no invention when the string carries neither
    assert text_help.parse_wage(None)["wage_amount"] is None
    assert text_help.parse_wage("salary 2500")["wage_period"] is None


def test_o10_every_record_carries_the_parsed_wage_columns(newspaper):
    """O10: employer_info's early returns (empty text, real-estate ad, empty
    cleaned text) must still emit the full column set, or pandas widens
    wage_is_range from bool to object with NaNs mixed in."""
    keys = {"wage", "wage_amount", "wage_period", "wage_is_range", "wage_n_amounts"}
    for text in ("", None, 123,
                 "LOVELY 3 bedroom apartment for sale realtor",   # real-estate drop
                 "cook wanted $8 a day apply within"):            # normal path
        out = newspaper.employer_info(text)
        assert keys <= set(out), "missing {} for {!r}".format(keys - set(out), text)
        assert isinstance(out["wage_is_range"], bool), out


def test_o7_selection_shared_between_live_and_offline_paths(us_data, monkeypatch):
    """O7: the live request path and the offline recompute must apply the very
    same selection, or the two will drift apart as the README snippet did."""
    payload_features = [_feature(0.9, "Norfolk", "Norfolk County", "23501", "A st")]
    monkeypatch.setenv("GEOAPIFY_URL", "https://api.example.invalid")
    monkeypatch.setenv("GEOAPIFY_API_KEY", "dummy")
    monkeypatch.setattr(resolve, "get_wrapper",
        lambda url, timeout=10: _stored(payload_features))
    live = resolve.geoapify_request("q", us_data.biggest_nearby_cities)[:3]
    offline = resolve.select_geoapify(_stored(payload_features),
        us_data.biggest_nearby_cities)
    assert live == offline, f"live {live} != offline {offline}"


def test_c6_newspaper_name_derivation(us_data):
    """C6: the newspaper abbreviation must be recovered from both plain and
    derived filenames, and identically across scripts."""
    assert common.newspaper_from_path("./test_data/NJG.csv") == "NJG"
    assert common.newspaper_from_path("./test_data/NJG-extract-all.gzip") == "NJG"
    assert common.newspaper_from_path("/a/b/LAT-resolve-8200000.gzip") == "LAT"
    # every newspaper in the mapping survives a round trip through both forms
    for paper in us_data.NEWSPAPER_TO_STATE_ID:
        assert common.newspaper_from_path("{}.csv".format(paper)) == paper
        assert common.newspaper_from_path("{}-extract-all.gzip".format(paper)) == paper


def test_o9_final_coordinates_match_the_chosen_county(us_data):
    """O9: the README's best_coordinates used different selection rules than
    resolve.py — no confidence floor, no nearby-city filter — so an ad's final
    lat/lon could come from a feature its own geo_county had rejected. finalize.py
    must pick the same feature the resolver would."""
    import finalize
    stored = [_stored([
        # highest raw confidence, but a city the resolver does not accept
        {"properties": {"rank": {"confidence": 0.99}, "city": "Nowheresville",
                        "county": "Elsewhere", "formatted": "9 Far Rd",
                        "lat": 1.0, "lon": 2.0}},
        {"properties": {"rank": {"confidence": 0.70}, "city": "Norfolk",
                        "county": "Norfolk County", "postcode": "23501",
                        "formatted": "45 Ocean St", "lat": 36.85, "lon": -76.28}},
    ])]
    out = finalize.best_coordinates(stored, us_data)
    assert out["county"] == "Norfolk", out
    assert out["latitude"] == 36.85 and out["longitude"] == -76.28, (
        "coordinates came from a feature the county selection rejected: %r" % out)


def test_e1_city_index_matches_the_dataframe_scan(us_data):
    """E1: the prebuilt city index replaced a per-ad full-table pandas scan.
    It must return exactly what that scan returned, in the same order."""
    for city in ["Norfolk", "Richmond", "Baltimore", "Nowhere-At-All"]:
        scan = us_data.US_CITIES[us_data.US_CITIES.city == city]
        idx = us_data.city_objects(city)
        assert len(idx) == len(scan), city
        assert [r["state_name"] for r in idx] == scan.state_name.to_list(), city
        assert [r["county_name"] for r in idx] == scan.county_name.to_list(), city
        # states_for_city must agree with the scan, deduplicated in first-seen order
        expected, seen = [], set()
        for s in scan.state_name:
            if s not in seen:
                seen.add(s)
                expected.append(s)
        assert list(us_data.states_for_city(city)) == expected, city


def test_e2_memoised_fuzzy_match_is_transparent(us_data_lat):
    """E2: memoising the per-token city match must not change any answer."""
    tokens = ["santa clar", "norfok", "los angles", "zzzznotacity", "Pasadena"]
    us_data_lat._city_match_memo.clear()
    fresh = [us_data_lat._best_city_match(t.title()) for t in tokens]
    cached = [us_data_lat._best_city_match(t.title()) for t in tokens]
    assert fresh == cached, "memo returned different answers on the second call"
    # and it agrees with calling thefuzz directly
    from thefuzz import process, fuzz
    direct = [process.extractOne(t.title(), us_data_lat.sorted_nearby_cities,
                                 scorer=fuzz.ratio) for t in tokens]
    assert fresh == direct, "memo diverges from an uncached lookup"
    # the cap must not be able to grow without bound
    assert us_data_lat.CITY_MEMO_MAX > 0


def test_v1_wilson_interval_behaves_at_the_edges():
    """V1: the normal approximation gives a zero-width interval at k=0 and k=n,
    which would understate uncertainty exactly where a small sample is weakest."""
    import validate
    p, lo, hi = validate.wilson(0, 20)
    assert p == 0.0 and lo == 0.0 and hi > 0.15, (p, lo, hi)
    p, lo, hi = validate.wilson(20, 20)
    assert p == 1.0 and hi == 1.0 and lo < 0.9, (p, lo, hi)
    p, lo, hi = validate.wilson(5, 10)
    assert lo < 0.5 < hi
    assert all(math.isnan(x) for x in validate.wilson(0, 0))


def _fake_corpus(n=400, coverage=0.6):
    """A corpus where the pipeline's hit rate differs from 50%, so an unweighted
    estimator drawn from a 50/50 sample is provably wrong."""
    rows = []
    for i in range(n):
        found = i < int(n * coverage)
        rows.append({
            'id': i,
            'year': 1930 + (i % 8) * 10,
            'raw_content': 'cook wanted apply 5 Main Street Norfolk',
            'addresses': [{'street': 'Main Street'}] if found else [],
            'wage': None,
        })
    return pd.DataFrame(rows)


def test_v2_sample_includes_ads_the_pipeline_missed(tmp_path):
    """V2: a sample drawn only from ads that produced an address cannot contain
    a false negative, so recall would be unmeasurable. Both sides must appear,
    and each row must carry the weight needed to undo the over-sampling."""
    import validate
    df = _fake_corpus(40, coverage=0.5)
    sample = validate.draw_sample(df, 20, seed=1)
    found = sample.addresses.apply(lambda a: len(a) > 0)
    assert found.any(), "sample contains no ads with a prediction"
    assert (~found).any(), "sample contains no empty ads — recall unmeasurable"
    for col in validate.DESIGN_COLUMNS:
        assert col in sample.columns, "missing design column %s" % col
    assert (sample.design_weight > 0).all()
    out = tmp_path / "s.csv"
    key = validate.write_template(sample, str(out), "NJG")
    written = pd.read_csv(out, keep_default_na=False)
    for col in list(validate.CODING_COLUMNS) + validate.DESIGN_COLUMNS:
        assert col in written.columns
    assert 'ad_text' in written.columns and written.ad_text.str.len().gt(0).all()
    assert os.path.isfile(key), "codebook not written"


def test_v4_design_weights_recover_the_population_rate():
    """V4: the harness deliberately over-samples the empty side, so the raw
    sample mean estimates accuracy under a 50/50 mixture rather than the
    corpus. The weights must undo that."""
    import validate
    df = _fake_corpus(400, coverage=0.75)
    sample = validate.draw_sample(df, 100, seed=3)
    pred = sample.addresses.apply(lambda a: len(a) > 0)
    w = sample.design_weight
    unweighted = pred.mean()
    weighted = (w * pred).sum() / w.sum()
    assert abs(weighted - 0.75) < 0.02, (
        "weighted coverage %.3f should recover the corpus rate 0.75" % weighted)
    assert abs(unweighted - 0.75) > 0.05, (
        "test is not exercising the bias: unweighted came out at %.3f" % unweighted)


def test_v5_sample_size_is_honoured():
    """V5: --n is a promise. Earlier allocation under-delivered badly (200 -> 188,
    50 -> 40) and two different --n values could yield the same sample."""
    import validate
    df = _fake_corpus(2000, coverage=0.55)
    for n in (50, 100, 200):
        got = len(validate.draw_sample(df, n, seed=2))
        assert abs(got - n) <= 0.1 * n, "asked for %d, got %d" % (n, got)


def test_v3_scoring_separates_precision_from_recall(capsys):
    """V3: an emitted-but-wrong address must hurt precision, and a missed
    address must hurt recall; the two must not be conflated. An emitted address
    the coder marked wrong must NOT count as a successful recall."""
    import validate
    df = pd.DataFrame({
        'year': [1970] * 4,
        'pred_addresses': ['5 Main St', '', '9 Oak St', ''],
        'pred_wage': [''] * 4,
        'stratum': ['1970|found', '1970|empty', '1970|found', '1970|empty'],
        'stratum_size': [10] * 4,
        'stratum_drawn': [2] * 4,
        'design_weight': [5.0] * 4,
        'truth_has_address': ['y', 'y', 'y', 'n'],
        'truth_is_job_ad': ['y'] * 4,
        'truth_address_is_worksite': ['y', '', '', ''],
        'truth_has_wage': ['n'] * 4,
        'truth_address': [''] * 4, 'truth_wage': [''] * 4, 'coder_notes': [''] * 4,
        # row 2 emitted an address the coder judged WRONG
        'pred_address_correct': ['y', '', 'n', ''],
        'pred_wage_correct': [''] * 4,
    })
    validate.score(df)
    out = capsys.readouterr().out
    # emitted: rows 0 and 2; correct: row 0 -> precision 50%
    assert "precision (strict)" in out and "50.0%" in out
    # truly have an address: rows 0,1,2; recalled correctly: only row 0 -> 33.3%
    assert "recall (emitted and correct)" in out and "33.3%" in out


def test_v6_blank_judgement_is_not_scored_as_wrong(capsys):
    """V6: an uncoded judgement cell means "not yet coded", not "the pipeline
    got it wrong". Scoring blanks as failures reported a perfect extraction as
    0% precision."""
    import validate
    df = pd.DataFrame({
        'year': [1970] * 2,
        'pred_addresses': ['5 Main St', '9 Oak St'],
        'pred_wage': [''] * 2,
        'stratum': ['1970|found'] * 2, 'stratum_size': [10] * 2,
        'stratum_drawn': [2] * 2, 'design_weight': [5.0] * 2,
        'truth_has_address': ['y', 'y'],
        'truth_is_job_ad': ['y'] * 2,
        'truth_address_is_worksite': [''] * 2,
        'truth_has_wage': ['n'] * 2,
        'truth_address': [''] * 2, 'truth_wage': [''] * 2, 'coder_notes': [''] * 2,
        'pred_address_correct': ['y', ''],   # second row simply not coded yet
        'pred_wage_correct': [''] * 2,
    })
    validate.score(df)
    out = capsys.readouterr().out
    assert "100.0%" in out, "the one coded row was correct; precision must be 100%"


def test_v7_template_without_design_columns_is_refused():
    """V7: scoring a template that lost its design columns would silently report
    the sampling design instead of the pipeline. It must refuse."""
    import validate
    df = pd.DataFrame({
        'pred_addresses': ['5 Main St'], 'pred_wage': [''],
        'truth_has_address': ['y'], 'truth_is_job_ad': ['y'],
        'truth_address_is_worksite': [''], 'truth_has_wage': ['n'],
        'truth_address': [''], 'truth_wage': [''],
        'pred_address_correct': ['y'], 'pred_wage_correct': [''],
        'coder_notes': [''],
    })
    with pytest.raises(SystemExit, match="design columns"):
        validate.score(df)


def test_s1_sample_corpus_exercises_the_audited_paths():
    """S1: the bundled corpus is synthetic, because the real advertisements are
    third-party copyright. It is only useful if it still contains the shapes the
    audit found bugs in, so pin them."""
    path = os.path.join(REPO_ROOT, "test_data", "NJG-sample.csv")
    assert os.path.isfile(path), "bundled sample corpus is missing"
    df = pd.read_csv(path, index_col=[0])
    text = df.raw_content.fillna("")

    assert len(df) >= 500, "corpus too small to be a useful demo"
    assert (df.year < 1963).any() and (df.year >= 1963).any(), \
        "corpus must span both sides of the ZIP-code gate"
    assert text.str.contains(common.AD_SEPARATOR, regex=False).any(), \
        "no concatenated multi-ad records, so first_ad is never exercised"
    # the low-precision markers must appear, since they are why the gate exists
    for marker in ("Court", "Circuit"):
        assert text.str.contains(marker, regex=False).any(), \
            "corpus never uses the %r marker" % marker
    # PO-box ZIPs are the reason GeoNames is a dependency at all
    assert text.str.contains("23501", regex=False).any(), \
        "corpus contains no PO-box ZIP code"
    # ground truth is what lets the harness be demonstrated without a coder
    for col in ("_truth_is_job_ad", "_truth_has_address", "_truth_has_wage"):
        assert col in df.columns, "missing ground-truth column %s" % col


def test_s2_sample_corpus_is_reproducible():
    """S2: same seed, same corpus — otherwise the documented demo numbers drift."""
    import make_sample_corpus as mk
    import random
    a = [mk.make_row(random.Random(7), i) for i in range(25)]
    b = [mk.make_row(random.Random(7), i) for i in range(25)]
    assert a == b, "generator is not deterministic for a fixed seed"


def test_o8_suite_cannot_reach_the_network():
    """O8: the autouse fixture in conftest.py must make a real request impossible,
    so running the suite with a live key in the environment cannot spend credit."""
    with pytest.raises(AssertionError, match="never hit the network"):
        resolve.SESSION.get("https://api.geoapify.com/v1/geocode/search")


def test_b5_api_key_not_persisted_in_logs():
    """B5: the API key must not survive into the stored request URL."""
    redacted = resolve._redact(
        "https://api.geoapify.com/v1/geocode/search?text=X&apiKey=SUPERSECRET")
    assert "SUPERSECRET" not in redacted
    assert "apiKey=REDACTED" in redacted
