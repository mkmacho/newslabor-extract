"""Build the US city and ZIP reference tables from public sources.

The pipeline previously shipped SimpleMaps' commercial city and ZIP databases,
whose licence forbids public redistribution. This script rebuilds equivalent
tables from sources that may be redistributed, and — more usefully for a reader —
turns "trust these two CSVs" into "here is exactly where every column came from".

SOURCES

  U.S. Census Bureau (public domain, 17 U.S.C. 105):
    ACS 2019-2023 5-year summary file, table B01003   place population
      (plus the 2022 vintage for three NY places the 2023 file drops)
    ACS geography lookup Geos20235YR                  place / place-in-county spine
    2020 ZCTA-to-county, -place, -county-subdivision  ZIP -> county, city -> ZIPs
    2023 Gazetteer, counties and county subdivisions  county names, CT regions

  GeoNames (CC BY 4.0, https://www.geonames.org/):
    US postal codes                                   ZIPs that have no ZCTA

The GeoNames file is used ONLY as a fallback, for ZIP codes the Census does not
model. That gap is not cosmetic: ZCTAs are built from populated census blocks, so
PO-box-only and "unique" ZIPs are absent — Norfolk's 23501 among them — while the
pipeline's `find_nearby_zipcodes` emits real USPS ZIPs from pyzipcode. Dropping
GeoNames costs 42 of 139 resolved counties on the CI slice. Because it is applied
only where the Census has nothing, Census values always win where both exist.

OUTPUT  auxiliary_files/geo/{uscities.csv,uszips.csv}  (~2.4 MB, vs 11.5 MB)

USAGE
    python scripts/build_geo_reference.py --download      # fetch, then build
    python scripts/build_geo_reference.py                 # rebuild from cache

Downloads total ~163 MB into a cache directory and are not committed; only the
two derived CSVs are.
"""
import argparse
import collections
import csv
import os
import re
import sys
import urllib.request
import zipfile

import pandas as pd

CENSUS = "https://www2.census.gov"
ACS = CENSUS + "/programs-surveys/acs/summary_file"
REL = CENSUS + "/geo/docs/maps-data/data/rel2020/zcta520"
GAZ = CENSUS + "/geo/docs/maps-data/data/gazetteer/2023_Gazetteer"

SOURCES = {
    "acsdt5y2023-b01003.dat": ACS + "/2023/table-based-SF/data/5YRData/acsdt5y2023-b01003.dat",
    "acsdt5y2022-b01003.dat": ACS + "/2022/table-based-SF/data/5YRData/acsdt5y2022-b01003.dat",
    "Geos20235YR.txt": ACS + "/2023/table-based-SF/documentation/Geos20235YR.txt",
    "tab20_zcta520_county20_natl.txt": REL + "/tab20_zcta520_county20_natl.txt",
    "tab20_zcta520_place20_natl.txt": REL + "/tab20_zcta520_place20_natl.txt",
    "tab20_zcta520_cousub20_natl.txt": REL + "/tab20_zcta520_cousub20_natl.txt",
    "2023_Gaz_counties_national.zip": GAZ + "/2023_Gaz_counties_national.zip",
    "2023_Gaz_cousubs_national.zip": GAZ + "/2023_Gaz_cousubs_national.zip",
    "geonames_US.zip": "https://download.geonames.org/export/zip/US.zip",
}

# Place-name suffixes. Stripped ONCE, never in a loop: repeated stripping turns
# "Jersey City city" into "Jersey" and "Panama City city" into "Panama".
LSAD = re.compile(r"\s+(CDP|city and borough|municipality and borough|charter township|"
                  r"city|town|village|borough|municipality|township|comunidad|"
                  r"zona urbana|corporation|county)$")
CONSOL = re.compile(r"\s+(unified|consolidated|metropolitan|metro|urban)"
                    r"(\s+county)?(\s+government)?$", re.I)
COUNTY_SUFFIX = re.compile(r"\s+(County|Parish|Borough|Census Area|Municipality|city|"
                           r"City and Borough|Municipio|Planning Region|District)$")
PLACE_SUFFIX = re.compile(r"\s+(city|town|village|borough|CDP|municipality|township|"
                          r"comunidad|zona urbana|urbana|plantation)$")

# New York city's largest place-part is Kings (Brooklyn); the pipeline wants the
# county the city is named for. SimpleMaps recorded Queens here, which AUDIT.md
# P8 identifies as an error, so this is a correction rather than a divergence.
COUNTY_OVERRIDE = {"3651000": "36061"}
BOROUGHS = {"Manhattan": "36061", "Brooklyn": "36047", "Queens": "36081",
            "Bronx": "36005", "Staten Island": "36085"}


def fetch(cache, force=False):
    os.makedirs(cache, exist_ok=True)
    for name, url in SOURCES.items():
        dest = os.path.join(cache, name)
        if os.path.exists(dest) and not force:
            continue
        print("  downloading {} ...".format(name), flush=True)
        urllib.request.urlretrieve(url, dest)
    for name in ("2023_Gaz_counties_national.zip", "2023_Gaz_cousubs_national.zip",
                 "geonames_US.zip"):
        with zipfile.ZipFile(os.path.join(cache, name)) as z:
            z.extractall(os.path.join(cache, "geonames" if "geonames" in name else "."))


def _pipe_rows(path):
    """Census pipe-delimited files carry a UTF-8 BOM."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="|"):
            yield row


def load_population(cache):
    def one(fn):
        pop = {}
        with open(os.path.join(cache, fn), encoding="utf-8-sig") as fh:
            reader = csv.reader(fh, delimiter="|")
            next(reader)
            for row in reader:
                try:
                    pop[row[0]] = int(row[1])
                except (ValueError, IndexError):
                    pass
        return pop
    current, previous = one("acsdt5y2023-b01003.dat"), one("acsdt5y2022-b01003.dat")
    # The older vintage covers three NY places the 2023 file drops, one of which
    # (Brentwood) sits above the population gate.
    return lambda geoid: current.get(geoid, previous.get(geoid))


def load_geography(cache):
    places, parts, ny_cousubs = [], [], []
    for row in _pipe_rows(os.path.join(cache, "Geos20235YR.txt")):
        level = row["SUMLEVEL"]
        if level == "160":
            places.append((row["STUSAB"], row["STATE"], row["PLACE"],
                           row["GEO_ID"], row["NAME"]))
        elif level == "155":
            parts.append((row["STATE"], row["COUNTY"], row["PLACE"], row["GEO_ID"]))
        elif level == "060" and row["STUSAB"] == "NY":
            ny_cousubs.append((row["STATE"], row["COUNTY"], row["GEO_ID"], row["NAME"]))
    return places, parts, ny_cousubs


def names_for(name_field, state_name):
    """Every spelling of a place an advertisement might plausibly use.

    Emits a SET, and the caller writes one row per spelling, because
    check_nearby_cities does an exact `token.title() in ...` test before it ever
    falls back to fuzzy matching — a single canonical spelling would lose the
    alternates entirely.
    """
    name = name_field
    # Strip the trailing ", <State>" by exact match. A regex on the last comma
    # breaks on "Lynchburg, Moore County metropolitan government, Tennessee",
    # and a Title-Case pattern silently drops "District of Columbia".
    if state_name and name.endswith(", " + state_name):
        name = name[: -len(", " + state_name)]
    else:
        name = re.sub(r",\s*[A-Z][a-z]+(\s[A-Za-z]+)*$", "", name)
    name = re.sub(r"\s+\(balance\)$", "", name.strip())
    name = re.sub(r"\s+\([A-Za-z .'\-]+ County\)$", "", name)

    out = set()
    dual = re.match(r"^(.*?)\s+\(([^)]+)\)\s+(city|town|village|CDP|borough)$", name)
    if dual:                                   # "San Buenaventura (Ventura) city"
        out |= {dual.group(1).strip(), dual.group(2).strip()}
        name = dual.group(1) + " " + dual.group(3)
    name = CONSOL.sub("", LSAD.sub("", name).strip()).strip()

    # Consolidated city-counties keep the city half ("Nashville-Davidson
    # metropolitan government" -> Nashville), but "Winston-Salem city" must not
    # be split, hence the guard on the ORIGINAL field naming a county.
    hyphen = re.match(r"^(.+?)[-/,]\s*[A-Za-z .'\-]+?(\s+County)?$", name)
    if hyphen and re.search(r"[-/,]", name) and \
            re.search(r"County|Davidson|Fayette", name_field):
        out.add(hyphen.group(1).strip())
    else:
        out.add(name)
    if name.startswith("Urban "):              # "Urban Honolulu CDP"
        out.add(name[6:])
    return {n for n in out if n and not n.endswith(" County")}


def county_lookups(cache):
    gaz = pd.read_csv(os.path.join(cache, "2023_Gaz_counties_national.txt"),
                      sep="\t", dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    # Strip the type word: without this, county_name agreement with the previous
    # data falls from 99.7% to 0.7%, because everything reads "Suffolk County".
    fips_to_name = {r.GEOID: COUNTY_SUFFIX.sub("", r.NAME.strip()) for r in gaz.itertuples()}
    fips_to_state = {r.GEOID[:2]: r.USPS for r in gaz.itertuples()}
    state_to_fips = {r.USPS: r.GEOID[:2] for r in gaz.itertuples()}
    return fips_to_name, fips_to_state, state_to_fips


def build_city_zips(cache, fips_to_state):
    """Map (city, state) -> the ZIP codes lying in it."""
    zips = collections.defaultdict(set)
    for row in _pipe_rows(os.path.join(cache, "tab20_zcta520_place20_natl.txt")):
        zcta = (row["GEOID_ZCTA5_20"] or "").strip()
        place = (row["GEOID_PLACE_20"] or "").strip()
        if not zcta or not place:
            continue
        area_z = int(row["AREALAND_ZCTA5_20"] or 0)
        area_p = int(row["AREALAND_PLACE_20"] or 0)
        part = int(row["AREALAND_PART"] or 0)
        # Require 1% of either side, which discards spatial slivers: ZCTA 23462
        # touches Norfolk at 0.01% while 99.99% of it lies in Virginia Beach.
        if (area_z and part / area_z >= 0.01) or (area_p and part / area_p >= 0.01):
            state = fips_to_state.get(place[:2])
            if state:
                zips[(PLACE_SUFFIX.sub("", row["NAMELSAD_PLACE_20"]), state)].add(zcta)

    # USPS place names close the gap for cities that mail under another name.
    gn = read_geonames(cache)
    for zipcode, place, state in zip(gn.zip.str.zfill(5), gn.place, gn.admin1code):
        zips[(place, state)].add(zipcode)
    return zips


def read_geonames(cache):
    cols = ["country", "zip", "place", "admin1", "admin1code", "admin2",
            "admin2code", "admin3", "admin3code", "lat", "lng", "acc"]
    return pd.read_csv(os.path.join(cache, "geonames", "US.txt"), sep="\t",
                       names=cols, dtype=str, keep_default_na=False)


def build_uszips(cache, fips_to_name, state_to_fips):
    """ZIP -> county, by dominant land area, with a CT fix and a GeoNames fallback."""
    # Connecticut replaced counties with planning regions in 2022. The 2020
    # relationship file still uses the old counties while the 2023 Gazetteer only
    # names the new regions, so 288 CT ZIPs would have an unnameable FIPS. Town
    # codes are stable across the change, so aggregate through those instead.
    cousubs = pd.read_csv(os.path.join(cache, "2023_Gaz_cousubs_national.txt"),
                          sep="\t", dtype=str)
    cousubs.columns = [c.strip() for c in cousubs.columns]
    town_to_region = {r.GEOID[5:]: r.GEOID[:5] for r in cousubs.itertuples()
                      if r.USPS == "CT"}
    ct_area = collections.defaultdict(lambda: collections.defaultdict(int))
    for row in _pipe_rows(os.path.join(cache, "tab20_zcta520_cousub20_natl.txt")):
        zcta = (row["GEOID_ZCTA5_20"] or "").strip()
        cousub = (row["GEOID_COUSUB_20"] or "").strip()
        if zcta and cousub.startswith("09") and cousub[5:] in town_to_region:
            ct_area[zcta][town_to_region[cousub[5:]]] += int(row["AREALAND_PART"] or 0)

    area = collections.defaultdict(lambda: collections.defaultdict(int))
    for row in _pipe_rows(os.path.join(cache, "tab20_zcta520_county20_natl.txt")):
        zcta = (row["GEOID_ZCTA5_20"] or "").strip()
        county = (row["GEOID_COUNTY_20"] or "").strip()
        if zcta:
            area[zcta][county] += int(row["AREALAND_PART"] or 0)

    # 30% of ZCTAs cross a county line and the pipeline wants exactly one, so a
    # tie-break is mandatory; ties break on the FIPS string to stay deterministic.
    zip_to_fips = {}
    for zcta, counties in area.items():
        chosen = ct_area[zcta] if zcta in ct_area else counties
        zip_to_fips[zcta] = max(chosen.items(), key=lambda kv: (kv[1], kv[0]))[0]
    census_zips = len(zip_to_fips)

    gn = read_geonames(cache)
    gn["fips"] = [(state_to_fips.get(a, "") + b.zfill(3)) if a in state_to_fips and b
                  else "" for a, b in zip(gn.admin1code, gn.admin2code)]
    gn = gn[(gn.fips.str.len() == 5) & (gn.fips.isin(fips_to_name))].drop_duplicates("zip")
    added = 0
    for zipcode, fips in zip(gn.zip.str.zfill(5), gn.fips):
        if zipcode not in zip_to_fips:        # fallback only; Census always wins
            zip_to_fips[zipcode] = fips
            added += 1

    out = pd.DataFrame({"zip": sorted(zip_to_fips)})
    out["county_fips"] = out.zip.map(zip_to_fips)
    out["county_name"] = out.county_fips.map(fips_to_name)
    out = out.dropna(subset=["county_name"])
    print("  uszips: {} from Census ZCTAs + {} GeoNames-only = {} rows".format(
        census_zips, added, len(out)))
    return out


def build_uscities(cache, aux_dir, population, fips_to_name, city_zips):
    places, parts, ny_cousubs = load_geography(cache)
    states = pd.read_csv(os.path.join(aux_dir, "states.csv"))
    id_to_state = dict(zip(states.Abbreviation, states.State))

    # county_fips per place: the county part holding the most people wins.
    best_part = {}
    for state, county, place, geoid in parts:
        pop = population(geoid)
        if pop is None:
            continue
        key = state + place
        if pop > best_part.get(key, (None, -1))[1]:
            best_part[key] = (state + county, pop)

    rows, seen = [], set()
    for stusab, state, place, geoid, name in places:
        pop = population(geoid)
        if pop is None:
            continue
        state_name = id_to_state.get(stusab)
        if state_name is None:
            continue                      # territories have no states.csv row
        fips = COUNTY_OVERRIDE.get(state + place) or \
            best_part.get(state + place, (None, None))[0]
        for spelling in names_for(name, state_name):
            if (spelling, stusab, state + place) in seen:
                continue
            seen.add((spelling, stusab, state + place))
            rows.append({"city": spelling, "state_id": stusab, "state_name": state_name,
                         "county_name": fips_to_name.get(fips) if fips else None,
                         "county_fips": fips,
                         "zips": " ".join(sorted(city_zips.get((spelling, stusab), ()))),
                         "population": pop})

    # The five New York boroughs are county subdivisions, not places. Without
    # them both New York papers — and, through state adjacency, Boston and
    # Hartford — lose the single largest block of candidate cities.
    boro_pop, boro_cousub = {}, {}
    for state, county, geoid, name in ny_cousubs:
        base = re.sub(r"\s+(borough|city|town|village|CDP)$", "",
                      name.split(",", 1)[0]).strip()
        if base in BOROUGHS and BOROUGHS[base] == state + county:
            pop = population(geoid)
            if pop:
                boro_pop[base] = max(boro_pop.get(base, 0), pop)
                boro_cousub[geoid[-10:]] = base
    boro_zips = collections.defaultdict(set)
    for row in _pipe_rows(os.path.join(cache, "tab20_zcta520_cousub20_natl.txt")):
        zcta = (row["GEOID_ZCTA5_20"] or "").strip()
        cousub = (row["GEOID_COUSUB_20"] or "").strip()
        if zcta and cousub in boro_cousub:
            boro_zips[boro_cousub[cousub]].add(zcta)
    for boro, fips in BOROUGHS.items():
        if boro in boro_pop:
            rows.append({"city": boro, "state_id": "NY", "state_name": "New York",
                         "county_name": fips_to_name.get(fips), "county_fips": fips,
                         "zips": " ".join(sorted(boro_zips.get(boro, set()) |
                                                 city_zips.get((boro, "NY"), set()))),
                         "population": boro_pop[boro]})
    print("  boroughs recovered: {}".format(sorted(boro_pop)))

    df = pd.DataFrame(rows)
    before = len(df)
    # Drop rows that can never influence behaviour, which is most of the table.
    df = df[(df.population.fillna(0) > 0) | (df.zips.str.len() > 0)]
    print("  uscities: {} rows -> {} after trim".format(before, len(df)))
    return df.sort_values(["state_id", "city"]).reset_index(drop=True)


def sanity_check(cities, zips):
    """Assert the canaries that caught real bugs while this was being built."""
    problems = []
    dc = cities[(cities.city == "Washington") & (cities.state_id == "DC")]
    if dc.empty:
        problems.append("Washington DC missing (the ', <State>' strip dropped it)")
    elif dc.county_fips.iloc[0] != "11001":
        problems.append("Washington DC county_fips is %r" % dc.county_fips.iloc[0])

    norfolk = cities[(cities.city == "Norfolk") & (cities.state_id == "VA")]
    if norfolk.empty:
        problems.append("Norfolk VA missing")
    else:
        have = set(norfolk.zips.iloc[0].split())
        want = {"23501", "23506", "23514", "23515", "23519", "23529", "23541"}
        if not want <= have:
            problems.append("Norfolk VA missing PO-box ZIPs %s" % sorted(want - have))

    z = dict(zip(zips.zip, zips.county_fips))
    if z.get("23501") != "51710":
        problems.append("ZIP 23501 -> %r, expected 51710" % z.get("23501"))
    if not cities.county_fips.dropna().map(len).eq(5).all():
        problems.append("some county_fips are not 5 characters")
    if cities.county_name.str.endswith(" County").any():
        problems.append("county_name still carries its type suffix")
    for bad, good in (("Jersey", "Jersey City"), ("Panama", "Panama City"),
                      ("Winston", "Winston-Salem")):
        if good not in set(cities.city):
            problems.append("name normalisation lost %r" % good)
    return problems


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true",
        help="Fetch the source files (~163 MB) before building.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    parser.add_argument("--cache", default=os.path.join(repo, "build-cache"),
        help="Where source downloads are kept (not committed).")
    parser.add_argument("--out", default=os.path.join(repo, "auxiliary_files", "geo"))
    parser.add_argument("--aux_dir", default=os.path.join(repo, "auxiliary_files"))
    args = parser.parse_args()

    if args.download or args.force:
        print("Fetching sources into {}".format(args.cache))
        fetch(args.cache, force=args.force)
    missing = [n for n in SOURCES if not os.path.exists(os.path.join(args.cache, n))]
    if missing:
        sys.exit("Missing sources {}. Run with --download.".format(missing))

    print("Building reference tables...")
    population = load_population(args.cache)
    fips_to_name, fips_to_state, state_to_fips = county_lookups(args.cache)
    city_zips = build_city_zips(args.cache, fips_to_state)
    zips = build_uszips(args.cache, fips_to_name, state_to_fips)
    cities = build_uscities(args.cache, args.aux_dir, population, fips_to_name, city_zips)

    problems = sanity_check(cities, zips)
    if problems:
        sys.exit("Sanity checks FAILED:\n  " + "\n  ".join(problems))
    print("  sanity checks passed")

    os.makedirs(args.out, exist_ok=True)
    cpath = os.path.join(args.out, "uscities.csv")
    zpath = os.path.join(args.out, "uszips.csv")
    cities.to_csv(cpath, index=False)
    zips.to_csv(zpath, index=False)
    total = os.path.getsize(cpath) + os.path.getsize(zpath)
    print("\nWrote {} ({:,} rows) and {} ({:,} rows) — {:.1f} MB total.".format(
        cpath, len(cities), zpath, len(zips), total / 1e6))
    print("Places >= 50,000: {}. NOTE these are ACS *place* populations, not the"
          " urban-agglomeration figures the previous data used.".format(
              int((cities.population >= 50000).sum())))
