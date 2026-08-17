"""Generate the synthetic advertisement corpus the demo and tests run on.

The pipeline was built against ~34M real classified advertisements digitised by
ProQuest. That text is third-party copyright — the newspapers' — licensed for
research, not redistributable, so it is not in this repository. This script
generates a stand-in with the same shape and the same awkwardness, so the demo,
the CI smoke test and the validation harness all still run end to end for anyone
who clones the repo.

Nothing here is copied from a real advertisement. Every ad is composed from
templates and word lists written for this file, then deliberately degraded to
imitate what OCR does to eighty-year-old newsprint.

The corpus is built to exercise the paths that actually broke during the audit,
so it doubles as a fixture:

  * every street marker, including the low-precision ones (Court, Ct, Pl, Ln,
    Circuit) both with and without a house number, since they are only accepted
    with one
  * the bare `_classifiedad_<date>_` separator fused to the previous token, which
    is how concatenated multi-ad records actually look
  * years either side of 1963, so the ZIP-code gate is exercised in both states
  * PO-box ZIP codes such as 23501 that have no census ZCTA
  * wage phrasings that each failed differently: "$8 a day" (single digit),
    "$500 a week" (rate word after a stop word), "500 per week" (no dollar sign),
    OCR-split decimals like "$9 75 hour", and explicit ranges
  * real-estate and non-job advertisements, which the filter is meant to skip

Deterministic: same seed, same corpus.

    python scripts/make_sample_corpus.py --n 2000

The filename prefix selects the geography profile the pipeline loads, so the
output is named NJG-sample.csv: the ads are Virginia-flavoured and NJG is the
Norfolk paper. They are synthetic, not Norfolk Journal & Guide advertisements.
"""
import argparse
import os
import random

import pandas as pd

STREETS = ["Granby", "Church", "Main", "Bute", "Freemason", "Monticello", "Ocean",
           "Park", "Colley", "Hampton", "Berkley", "Tidewater", "Chesapeake",
           "Princess Anne", "Little Creek", "Maple", "Cedar", "Walnut", "High"]
MARKERS = ["Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd",
           "Drive", "Lane"]
# Accepted only with a house number, so the generator emits both forms.
GATED_MARKERS = ["Court", "Ct", "Place", "Pl", "Circuit", "Ln"]
CITIES = ["Norfolk", "Portsmouth", "Hampton", "Newport News", "Suffolk",
          "Richmond", "Chesapeake", "Virginia Beach"]
ZIPS = ["23501", "23502", "23504", "23507", "23510", "23517", "23523",
        "23601", "23605", "23701", "23704", "23320"]
JOBS = ["cook", "porter", "maid", "driver", "clerk", "waiter", "laborer",
        "seamstress", "mechanic", "janitor", "nurse", "typist", "carpenter",
        "shipfitter", "stevedore", "bellhop", "presser", "barber"]
OPENERS = ["WANTED", "HELP WANTED", "WANTED AT ONCE", "MEN WANTED",
           "WOMEN WANTED", "EXPERIENCED", "RELIABLE"]
CLOSERS = ["apply in person", "apply at once", "steady work", "good wages",
           "references required", "no experience necessary", "see manager",
           "call after six", "write Box 44", "must be neat"]
WAGES = [
    "${a} a day", "${a} a week", "${a} per week", "{a} per week",
    "${a} an hour", "${a} per hour", "${a} a month", "salary ${a}",
    "${a} weekly", "${a} monthly", "pays ${a} a week", "${a} to ${b} a week",
    "${a}-${b} per week", "${a} {c} hour",   # the last imitates a split decimal
]
REAL_ESTATE = [
    "FOR SALE lovely 3 bedroom home with garage, {c} area, see realtor",
    "APARTMENT for rent, furnished, near {c}, call landlord",
    "vacant lot for sale, {c}, terms arranged",
]
NON_JOB = [
    "LOST small brown dog near {s} {m}, reward offered",
    "PUBLIC NOTICE the annual meeting will be held at the courthouse",
    "FOUND set of keys on {s} {m}, owner may claim",
]


def ocr_noise(text, rng, rate=0.04):
    """Imitate the specific ways this corpus is damaged: dropped and doubled
    characters, l/1 and O/0 confusion, and stray spaces inside numbers."""
    out = []
    for ch in text:
        r = rng.random()
        if r < rate * 0.30 and ch.isalpha():
            out.append({"l": "1", "I": "l", "O": "0", "o": "e", "s": "a",
                        "e": "c", "n": "m"}.get(ch, ch))
        elif r < rate * 0.45 and ch == " ":
            continue                              # words run together
        elif r < rate * 0.60 and ch.isalpha():
            continue                              # dropped character
        elif r < rate * 0.70 and ch.isdigit():
            out.append(ch + " ")                  # digits split by a space
        else:
            out.append(ch)
    return "".join(out)


def make_ad(rng, year):
    """One advertisement, plus the ground truth about what it contains."""
    kind = rng.random()
    has_zip = year >= 1963 and rng.random() < 0.45
    city = rng.choice(CITIES)

    if kind < 0.08:
        text = rng.choice(REAL_ESTATE).format(c=city)
        return text, {"job": False, "address": False, "wage": False}
    if kind < 0.13:
        text = rng.choice(NON_JOB).format(s=rng.choice(STREETS),
                                          m=rng.choice(MARKERS))
        return text, {"job": False, "address": True, "wage": False}

    parts = [rng.choice(OPENERS), rng.choice(JOBS)]
    if rng.random() < 0.5:
        parts.append(rng.choice(["experienced", "colored", "young", "steady",
                                 "part time", "full time"]))

    has_address = rng.random() < 0.62
    if has_address:
        marker = (rng.choice(GATED_MARKERS) if rng.random() < 0.18
                  else rng.choice(MARKERS))
        street = rng.choice(STREETS)
        # Gated markers appear both with and without a number on purpose: the
        # numberless form is what the audit found matching courthouse boilerplate.
        if rng.random() < 0.75:
            parts.append("apply {} {} {}".format(
                rng.randint(2, 3999), street, marker))
        else:
            parts.append("apply {} {}".format(street, marker))
        parts.append(city)
        if rng.random() < 0.35:
            parts.append(rng.choice(["Virginia", "Va", "VA"]))

    has_wage = rng.random() < 0.30
    if has_wage:
        parts.append(rng.choice(WAGES).format(
            a=rng.choice([5, 8, 9, 12, 18, 35, 40, 60, 75, 100, 125, 300, 500]),
            b=rng.choice([15, 45, 90, 150, 600]),
            c=rng.choice([25, 50, 75])))

    parts.append(rng.choice(CLOSERS))
    if has_zip:
        parts.append(rng.choice(ZIPS))
    text = " ".join(str(p) for p in parts)
    return text, {"job": True, "address": has_address, "wage": has_wage}


def make_row(rng, index):
    """One CSV row, matching the schema the pipeline reads."""
    year = rng.choices(
        [rng.randint(1916, 1929), rng.randint(1930, 1949), rng.randint(1950, 1962),
         rng.randint(1963, 1979), rng.randint(1980, 2003)],
        weights=[5, 12, 10, 33, 40])[0]
    text, truth = make_ad(rng, year)
    text = ocr_noise(text, rng)

    # Roughly 6% of real records fuse several advertisements from different
    # dates, separated by a bare token welded to the previous word.
    if rng.random() < 0.06:
        other_year = rng.randint(1916, 2003)
        tail, _ = make_ad(rng, other_year)
        text = "{} .{}_{}{:02d}{:02d}_1 {}".format(
            text, "_classifiedad", other_year, rng.randint(1, 12),
            rng.randint(1, 28), ocr_noise(tail, rng))

    ad_id = "_classifiedad_{}{:02d}{:02d}_{}_{}".format(
        year, rng.randint(1, 12), rng.randint(1, 28),
        rng.randint(1, 9), rng.randint(1, 40))
    job = rng.choice(JOBS)
    return {
        "id": ad_id, "X.1": index + 1, "X": index + 1,
        "zip": None, "telephone": None, "states": None,
        "adstitles": job, "original_titles": job,
        "soc": rng.randint(11, 53) * 1000, "soctitles": job,
        "cs": None, "year": year, "newspaper": "NJG",
        "NonroutineAnalytics": round(rng.random(), 3),
        "NonroutineInteractive": round(rng.random(), 3),
        "RoutineCognitive": round(rng.random(), 3),
        "RoutineManual": round(rng.random(), 3),
        "NonroutineManual": round(rng.random(), 3),
        "racialTermraw": None, "racialTermboolean": False,
        "racialTermparsed": None,
        "clean_content": text.lower(), "raw_content": text,
        # Ground truth, which the real data does not have. Kept so the
        # validation harness can be demonstrated without a human coding pass.
        "_truth_is_job_ad": truth["job"],
        "_truth_has_address": truth["address"],
        "_truth_has_wage": truth["wage"],
    }


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", default=os.path.join(repo, "test_data", "NJG-sample.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [make_row(rng, i) for i in range(args.n)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out)

    print("Wrote {} ads to {} ({:.0f} KB, seed {}).".format(
        len(df), args.out, os.path.getsize(args.out) / 1024, args.seed))
    print("  years {}-{}, {} before ZIP codes existed".format(
        df.year.min(), df.year.max(), int((df.year < 1963).sum())))
    print("  ground truth: {} job ads, {} with an address, {} with a wage".format(
        int(df._truth_is_job_ad.sum()), int(df._truth_has_address.sum()),
        int(df._truth_has_wage.sum())))
    print("  {} records fuse more than one advertisement".format(
        int(df.raw_content.str.contains("_classifiedad_", regex=False).sum())))
