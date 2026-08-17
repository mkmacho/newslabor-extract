"""Focused regressions for resumability, concurrency, and final assembly."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
import runpy
import sys
import threading
import time

import pandas as pd
import pytest

import build_geo_reference
import extract
import finalize
import resolve


@pytest.mark.parametrize("script", [
    "extract.py", "resolve.py", "recompute.py", "finalize.py"])
def test_all_population_sensitive_commands_expose_min_pop(
        script, monkeypatch, capsys):
    path = os.path.join(os.path.dirname(extract.__file__), script)
    monkeypatch.setattr(sys, "argv", [path, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(path, run_name="__main__")
    assert exit_info.value.code == 0
    assert "--min_pop" in capsys.readouterr().out


def test_worker_initializer_forwards_min_pop(monkeypatch):
    seen = {}
    sentinel = object()

    def fake_build(newspaper, aux_dir, min_pop):
        seen.update(newspaper=newspaper, aux_dir=aux_dir, min_pop=min_pop)
        return sentinel

    monkeypatch.setattr(extract, "build_newspaper", fake_build)
    monkeypatch.setattr(extract, "_WORKER_NEWSPAPER", None)
    extract._init_worker("NJG", "/aux", 4321)

    assert seen == {"newspaper": "NJG", "aux_dir": "/aux", "min_pop": 4321}
    assert extract._WORKER_NEWSPAPER is sentinel


@pytest.mark.parametrize("nrows,batch_size,skip", [
    (250, 100, -1), (250, 100, 251), (250, 100, 150), (250, 0, 0)])
def test_resolve_rejects_invalid_resume_boundaries(nrows, batch_size, skip):
    with pytest.raises(ValueError):
        resolve.validate_resume(nrows, batch_size, skip)
    resolve.validate_resume(250, 100, 200)


def test_extraction_batches_are_resumable_and_combine_address_and_wage():
    assert list(extract.iter_batch_bounds(250, 100, skip=100)) == [
        (100, 200, 200), (200, 250, 300)]
    with pytest.raises(ValueError, match="whole multiple"):
        list(extract.iter_batch_bounds(250, 100, skip=150))

    class FakeNewspaper:
        def extract(self, text, year):
            return [{"text": text, "year": year}]

        def employer_info(self, text):
            return {"wage": "$8 hour", "wage_amount": 8.0,
                    "wage_period": "hour", "wage_is_range": False,
                    "wage_n_amounts": 1}

    sample = pd.DataFrame(
        {"raw_content": ["first", "second", "third"]}, index=[10, 11, 12])
    result = extract.extract_one_batch(
        sample, [1950, 1960, 1970], 1, 3, FakeNewspaper(),
        extract_address=True, extract_wage=True)
    extract.assign_batch_results(sample, result)

    assert result.index.to_list() == [11, 12]
    assert {"addresses", "wage", "wage_n_amounts"} <= set(result.columns)
    assert sample.loc[11, "addresses"][0]["year"] == 1960
    assert sample.loc[12, "wage_n_amounts"] == 1
    assert pd.isna(sample.loc[10, "addresses"]), "skipped row was overwritten"

    with pytest.raises(ValueError, match="outside the input sample"):
        extract.assign_batch_results(
            sample, pd.DataFrame({"wage": ["$9 hour"]}, index=[99]))


def test_threaded_cache_is_single_flight():
    workers = 8
    barrier = threading.Barrier(workers)
    calls = []
    calls_lock = threading.Lock()

    def slow_request(query, cities, timeout=10):
        with calls_lock:
            calls.append(query)
        time.sleep(0.05)
        return ("1 Main St", "Norfolk", "23501",
                {"status_code": 200, "url": "stub"})

    def fetch(_):
        barrier.wait()
        return resolve.cached_request(
            slow_request, "one shared query", {"Norfolk"})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch, range(workers)))

    assert len(calls) == 1, "concurrent cache misses issued duplicate requests"
    assert all(result == results[0] for result in results)
    assert resolve.cache_stats() == (1, workers - 1)
    assert resolve.status_summary() == "200:1"


def test_resolved_batches_attach_without_duplicate_payload_frame():
    sample = pd.DataFrame({"raw_content": ["a", "b", "c"]}, index=[10, 11, 12])
    results = pd.DataFrame({
        "geo_county": ["Norfolk", "Richmond"],
        "geo_requests": [[{"status_code": 200}], [{"status_code": 404}]],
    }, index=[11, 12])

    resolve.assign_batch_results(sample, results)

    assert pd.isna(sample.loc[10, "geo_county"])
    assert sample.loc[11, "geo_county"] == "Norfolk"
    assert sample.loc[12, "geo_requests"][0]["status_code"] == 404
    with pytest.raises(ValueError, match="outside the input sample"):
        resolve.assign_batch_results(
            sample, pd.DataFrame({"geo_county": ["Nowhere"]}, index=[99]))


def test_single_flight_releases_waiters_after_request_error():
    workers = 4
    barrier = threading.Barrier(workers)
    calls = []

    def failing_request(query, cities, timeout=10):
        calls.append(query)
        time.sleep(0.05)
        raise RuntimeError("temporary failure")

    def fetch(_):
        barrier.wait()
        return resolve.cached_request(failing_request, "shared failure", set())

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, i) for i in range(workers)]
        for future in futures:
            with pytest.raises(RuntimeError, match="temporary failure"):
                future.result(timeout=2)

    assert len(calls) == 1
    assert not resolve._INFLIGHT
    with pytest.raises(RuntimeError, match="temporary failure"):
        resolve.cached_request(failing_request, "shared failure", set())
    assert len(calls) == 2, "a failed request was incorrectly cached"


def test_request_exception_message_redacts_api_key(monkeypatch):
    secret = "DUMMY-SECRET-NEVER-PERSIST"

    def fail(*args, **kwargs):
        raise resolve.RequestException(
            "request failed for https://example.invalid/?apiKey={}&q=x".format(
                secret))

    monkeypatch.setattr(resolve.SESSION, "get", fail)
    output = resolve.get_wrapper(
        "https://example.invalid/?apiKey={}&q=x".format(secret))
    assert secret not in repr(output)
    assert "apiKey=REDACTED" in output["url"]
    assert "apiKey=REDACTED" in output["message"]


def test_provider_base_url_requires_https_without_embedded_credentials():
    assert resolve.validate_base_url("https://api.geoapify.com/") == (
        "https://api.geoapify.com")
    for unsafe in (
            "http://api.geoapify.com",
            "https://user:pass@api.geoapify.com",
            "https://api.geoapify.com?redirect=elsewhere"):
        with pytest.raises(ValueError, match="must be an HTTPS origin"):
            resolve.validate_base_url(unsafe)


def test_geo_reference_download_rejects_non_https_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        build_geo_reference, "SOURCES", {"unsafe.txt": "file:///tmp/source"})
    with pytest.raises(ValueError, match="Refusing non-HTTPS"):
        build_geo_reference.fetch(tmp_path)


def test_best_coordinates_uses_exact_qualified_feature_with_duplicate_address():
    class Geo:
        biggest_nearby_cities = {"Norfolk"}

    response = {"status_code": 200, "content": {"features": [
        {"properties": {"rank": {"confidence": 0.99},
                        "city": "Nowheresville", "county": "Wrong County",
                        "formatted": "Same address",
                        "lat": 1.0, "lon": 2.0}},
        {"properties": {"rank": {"confidence": 0.9}, "city": "Norfolk",
                        "county": "Right County", "formatted": "Same address",
                        "lat": 3.0, "lon": 4.0}},
    ]}}
    out = finalize.best_coordinates([response], Geo())
    assert out["county"] == "Right"
    assert (out["latitude"], out["longitude"]) == (3.0, 4.0)


def test_county_final_matches_selected_coordinates_then_uses_fallbacks():
    class Geo:
        def counties_from_zips(self, zipcodes):
            return {"23501": ["Norfolk"]}.get(zipcodes[0])

    coordinates = pd.DataFrame({
        "county": ["Chosen", None, None, None, None],
        "postcode": [None, "23501", None, None, None],
        "coordinates_confidence": [0.9, 0.8, None, None, 0.7],
    }, index=[5, 6, 7, 8, 9])
    geo = pd.DataFrame({
        "geo_county": ["Wrong", "Wrong", "Modal", None, "Wrong"],
        "geo_zip_county": ["Wrong zip", "Wrong zip", "Zip", "Zip fallback",
                           "Wrong zip"],
    }, index=coordinates.index)

    out = finalize.consolidate_counties(coordinates, geo, Geo())
    assert out.county_final.to_list()[:4] == [
        "Chosen", "Norfolk", "Modal", "Zip fallback"]
    assert pd.isna(out.loc[9, "county_final"]), (
        "a modal county from other queries was attached to selected coordinates")


def test_wage_projection_keeps_count_and_reads_parquet_once(tmp_path, monkeypatch):
    path = tmp_path / "wage.gzip"
    pd.DataFrame({
        "id": [1], "wage": ["$8 hour"], "wage_amount": [8.0],
        "wage_period": ["hour"], "wage_is_range": [False],
        "wage_n_amounts": [1], "raw_content": ["large text"],
    }).to_parquet(path)

    real_read = pd.read_parquet
    calls = []

    def tracking_read(*args, **kwargs):
        calls.append(kwargs.get("columns"))
        return real_read(*args, **kwargs)

    monkeypatch.setattr(finalize.pd, "read_parquet", tracking_read)
    wages = finalize.read_wage_columns(path)

    assert len(calls) == 1
    assert calls[0] == ["id", "wage", "wage_amount", "wage_period",
                        "wage_is_range", "wage_n_amounts"]
    assert "wage_n_amounts" in wages.columns
    assert "raw_content" not in wages.columns


def test_wage_projection_rejects_missing_runtime_schema(tmp_path):
    no_id = tmp_path / "no-id.gzip"
    pd.DataFrame({"wage": ["$8 hour"]}).to_parquet(no_id)
    with pytest.raises(ValueError, match="No `id` column"):
        finalize.read_wage_columns(no_id)

    no_wage = tmp_path / "no-wage.gzip"
    pd.DataFrame({"id": [1], "raw_content": ["text"]}).to_parquet(no_wage)
    with pytest.raises(ValueError, match="No wage columns"):
        finalize.read_wage_columns(no_wage)


def _load_merge_batch_module():
    path = os.path.join(os.path.dirname(extract.__file__), "merge-batch.py")
    spec = importlib.util.spec_from_file_location("merge_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_batch_validates_ids_not_only_row_count():
    merge_batch = _load_merge_batch_module()
    template = pd.DataFrame({"raw": ["a", "b"]}, index=[10, 11])
    merge_batch.validate_batch_indices(
        template, pd.DataFrame({"wage": [1, 2]}, index=[11, 10]))

    with pytest.raises(ValueError, match="row ids do not match"):
        merge_batch.validate_batch_indices(
            template, pd.DataFrame({"wage": [1, 2]}, index=[10, 12]))
    with pytest.raises(ValueError, match="duplicate row ids"):
        merge_batch.validate_batch_indices(
            template, pd.DataFrame({"wage": [1, 2]}, index=[10, 10]))
    with pytest.raises(ValueError, match="Found 1 extraction rows"):
        merge_batch.validate_batch_indices(
            template, pd.DataFrame({"wage": [1]}, index=[10]))
