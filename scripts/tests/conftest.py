"""Shared test setup.

Two jobs: put `scripts/` on the import path, and make it *impossible* for the
test suite to reach the network. resolve.py builds a real `requests.Session` at
import time and the README tells people to export a live GEOAPIFY_API_KEY, so the
natural habit is to run pytest with a working key in the environment. Every test
monkeypatches its own request path, but a future test that forgets would spend
real API credit. The autouse fixture below removes that possibility.

It also resets resolve.py's module-level state between tests: the query cache,
the cache-hit counter, the status histogram and the rate limiter are all globals,
so without this a test's cached responses and counters leak into the next one.
"""
import os
import sys

import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
AUX_DIR = os.path.join(REPO_ROOT, "auxiliary_files")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


class _NoNetwork:
    """Stands in for requests.Session and fails loudly rather than dialling out."""

    def get(self, *args, **kwargs):
        raise AssertionError(
            "A test attempted a real HTTP request. Monkeypatch get_wrapper or the "
            "request function instead — this suite must never hit the network."
        )

    request = post = get


@pytest.fixture(autouse=True)
def _offline_and_clean_globals():
    import resolve

    saved_session = resolve.SESSION
    resolve.SESSION = _NoNetwork()
    resolve._QUERY_CACHE.clear()
    resolve._CACHE_HITS[0] = 0
    resolve._STATUS_COUNTS.clear()
    resolve._RATE_STATE.update({"min_interval": 0.0, "next_time": 0.0})
    try:
        yield
    finally:
        resolve.SESSION = saved_session
        resolve._QUERY_CACHE.clear()
        resolve._CACHE_HITS[0] = 0
        resolve._STATUS_COUNTS.clear()
