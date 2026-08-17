# Current audit and verification record

This document describes the state of `newslabor-extract` at the public-release
audit dated 2026-08-17. It records what was checked, the current safeguards, and
what remains a methodological or empirical limitation. It is not an
independent certification of the software or the paper's results.

## Responsibility and assistance

Miguel Camacho Horvitz is responsible for this repository's public release and
for deciding which audit changes to accept. OpenAI Codex and Anthropic Claude
assisted with code review, refactoring, test construction, documentation, and
release checks. Miguel reviewed the resulting work and accepts responsibility for
the code and claims retained here. The AI-assisted review should not be described
as an independent third-party audit.

Miguel contributed research software and data engineering to historical
job-advertisement research led by Joan Martínez and Ellora Derenoncourt. He is not
a paper coauthor, and this repository is not the paper's replication package.

## Scope

The review covered the executable pipeline and its public-release boundary:

- extraction, geocoding, stored-response re-scoring, checkpoint merging, and
  final assembly;
- geography and OCR helper functions;
- the synthetic-corpus generator and offline demonstration;
- validation-sample design and scoring;
- test isolation from the network;
- API-key handling and persisted request logs;
- bundled-data provenance and documentation consistency.

The review did **not** reproduce a paper estimation sample, re-run a production
corpus, make paid Geoapify requests, resolve ownership or publication
authorization for repository-authored code, or validate substantive paper
results.

## Verification performed

| Check | Current result | Limitation |
|---|---|---|
| Python test suite | 73 tests collected and passing under Python 3.10.20 and 3.11 | Tests cover selected invariants, not every branch or empirical failure mode. |
| Dependency advisories | `pip-audit -r requirements.txt` reports no known vulnerabilities | Advisory databases and dependency state can change after the audit date. |
| Network isolation | Test session replaces HTTP requests with a raising stub | Live provider behavior is not exercised. |
| Offline smoke test | Extraction and ZIP-based county assignment run end to end on 2,000 synthetic ads | Synthetic difficulty is chosen by the generator. |
| Synthetic reproducibility | Generator output is pinned by seed and regression tests | It is not derived from historical ads. |
| Synthetic label regression | Address precision is 85.27% and recall 87.86%; wage precision is 100% and recall 92.51% on the deterministic 2,000-row fixture | These are generator-label checks, not estimates of performance on historical newspapers. |
| Geography rebuild | SHA-256 values verify the ten audited source files; from those cached bytes the builder reproduces `states.csv`, `uscities.csv`, and `uszips.csv` byte for byte | The GeoNames URL is mutable and its archive is not bundled, so the checksum detects drift but cannot guarantee future retrieval of the audited bytes. |
| Determinism | Candidate ordering and fuzzy-city lookup are regression-tested across hash seeds | External API results may still change over time. |
| Validation estimator Monte Carlo | With seed `20260817`, 500 implemented samples of 200 rows across 20 strata pass the stated bias and 0.900 coverage guards in six synthetic patterns; interior-pattern coverage is 0.952–0.986 and boundary coverage is 1.000 | Intervals are approximate Kish-effective-n Wilson intervals on a fixed synthetic population; the check omits finite-population correction, coder error, and coder nonresponse. |
| Credential persistence | Current request logging redacts the API key and a test pins that behavior | Git history and external logs require separate release-time secret scanning. |

The exact commands are in [README.md](README.md). A check counts as verified only
when its command completes successfully; documentation alone is not verification.

## Current safeguards

| Topic | Verified behavior |
|---|---|
| Geocoder selection | Each candidate response contributes its highest-confidence qualifying feature; address, county, postcode, and coordinates remain tied to the selected feature. |
| Candidate extraction | Address and wage paths use the same `_classifiedad_` boundary and process the first ad. Short street and state tokens remain reachable, while lower-precision street markers require a house number. |
| Candidate ordering | Address ordering and fuzzy-city selection are deterministic across Python hash seeds; indexed lookup and bounded memoization preserve the tested results. |
| Wage fields | Single-digit dollar wages and `a/an` rate phrases are reachable; parsed outputs retain amount, period, range, and ambiguity fields. |
| Geographic scope | State adjacency is symmetric, adjacent ZIP codes are matched independently, and every population-sensitive command exposes the same `--min_pop` control. |
| Process execution | Serial and worker modes share batch boundaries; worker-count controls are typed consistently; resumed runs use distinct output names. |
| Checkpoints | Batch-size assumptions are checked, checkpoint write failures stop the run, merge inputs are validated by identifiers, and checkpoint deletion is opt-in. |
| Network behavior | Provider and geography-source endpoints require HTTPS, and provider URLs reject embedded credentials. Connections are pooled; retry, timeout, rate-limit, and low-success abort behavior are explicit. Repeated queries are cached and concurrent duplicates are coalesced. |
| Credentials and logs | The API key is required at startup, read from the environment, and redacted from persisted URLs. Authentication failures are not retained in the query cache. |
| Offline paths | Stored responses can be re-scored without new requests; ZIP-bearing candidates can yield county and FIPS fields without geocoding; live and offline response selection share helpers. |
| Final assembly | Geolocation and wage inputs join on stable identifiers; duplicate-induced row growth is rejected; final county values follow the selected coordinates where available. |
| Output exposure | Extract, resolve, and final CSV copies are opt-in; Parquet is the default. CSV copies retain source text, and resolve outputs retain provider response payloads. |
| Human validation | Sampling reaches the requested feasible size, covers emitted and missed addresses, preserves design weights, rejects edited design fields, separates precision and recall, and distinguishes incomplete coding and non-job ads. |
| Public fixture | The 2,000-row synthetic sample is deterministic and exercises documented edge cases without reproducing historical source text. |

## Methodological limitations that remain

These are not resolved by software tests and must be handled in research design,
manual validation, sensitivity analysis, or interpretation.

### Unit of analysis

One `raw_content` value can contain several ads joined by `_classifiedad_`. The
pipeline keeps the first segment to avoid cross-ad contamination. That choice can
discard genuine trailing text that is not represented elsewhere.

### Address construct

The extractor finds addresses, not worksites. Personnel offices, employment
agencies, application addresses, and worksites are observationally similar. The
human-validation template therefore asks whether an address is the worksite; the
result should bound interpretation of any geography variable.

### Geographic sampling and time

Candidate cities are restricted to the newspaper's home and adjacent states and
to present-day ACS places above `--min_pop` (15,000 by default). ZIP codes are
ignored before 1963. Present-day ZIP and county relationships are then applied to
historical text. These choices can create time- and place-dependent coverage.

### OCR and heuristic parsing

Street spell correction uses an English frequency dictionary and can rewrite
proper nouns. Address rules can miss unusual formats or accept application
addresses. Wage OCR can split decimals and thousands groups; `wage_amount` is
therefore not automatically analysis-ready, and amounts written entirely in
words are not parsed. Inspect the source wage string and ambiguity fields and
validate on the target corpus.

### Provider dependence

Live validation depends on Geoapify's API, ranking, coverage, availability, terms,
and pricing. Stored responses make selection reproducible after a run, but a new
run may not reproduce historical provider results. Full response payloads may
also be sensitive research data.

### Accuracy and external validity

The bundled synthetic sample has generator-known labels and is useful for
regression testing. It cannot estimate accuracy on historical newspapers. A
credible production estimate requires a documented, independently coded sample
that includes both emitted and missed addresses, design weights, uncertainty, and
negative cases such as non-job ads.

## Release checks

Before publishing or linking a release:

1. Run the 73-test suite and offline smoke test from a clean environment.
2. Scan the complete Git history for credentials, private paths, licensed source
   text, and removed binary artifacts; rotate any exposed credential before
   rewriting history.
3. Confirm that every public data file is either redistributable or explicitly
   excluded from the project license in [NOTICE](NOTICE).
4. Confirm collaborator permission for public code, names, project framing, and
   any statistics derived from non-public data.
5. Check that CV and profile links resolve to the current repository.

Passing the software tests does not satisfy the provenance, permission, or
empirical-validation checks.
