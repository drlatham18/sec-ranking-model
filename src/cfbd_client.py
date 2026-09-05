"""Thin CollegeFootballData API client with on-disk caching.

Cache is keyed by endpoint+params, with six-hour expiry for changing seasons
and a 30-day expiry for historical seasons. Legacy untimed entries refresh.
Key resolution order: CFBD_API_KEY env var -> ~/.cfbd_key file -> config.
"""
import os, json, time, hashlib, pathlib
from datetime import datetime, timezone
import tempfile
import requests

BASE = os.environ.get("CFBD_BASE", "https://api.collegefootballdata.com")
ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)


def _key():
    k = os.environ.get("CFBD_API_KEY")
    if k:
        return k.strip()
    f = pathlib.Path.home() / ".cfbd_key"
    if f.exists():
        return f.read_text().strip()
    raise RuntimeError(
        "No CFBD API key. Set CFBD_API_KEY env var or write it to ~/.cfbd_key "
        "(free key: https://collegefootballdata.com/key)"
    )


def get(endpoint, use_cache=True, refresh=False, **params):
    """GET an endpoint, returning parsed JSON (list of dicts). Cached to disk."""
    params = {k: v for k, v in params.items() if v is not None}
    sig = hashlib.md5(
        (endpoint + json.dumps(params, sort_keys=True)).encode()
    ).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    path = CACHE / f"{safe}__{sig}.json"

    now = datetime.now(timezone.utc)
    last_closed = now.year - (1 if now.month >= 2 else 2)
    try:
        historical = int(params.get("year", now.year)) < last_closed
    except (TypeError, ValueError):
        historical = False
    ttl = 30 * 86400 if historical else 6 * 3600
    if use_cache and not refresh and os.environ.get("CFBD_REFRESH") != "1" and path.exists():
        try:
            cached = json.loads(path.read_text())
            if (isinstance(cached, dict) and cached.get("_cache_version") == 1
                    and 0 <= time.time() - cached["fetched_at"] < ttl):
                return cached["data"]
        except (ValueError, KeyError, TypeError):
            pass  # Missing provenance or corrupt cache: fetch, never silently use it.


    url = f"{BASE}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {_key()}", "Accept": "application/json"}

    for attempt in range(5):
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code == 200:
            data = r.json()
            CACHE.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=CACHE)
            try:
                with os.fdopen(fd, "w") as out:
                    json.dump({"_cache_version": 1, "fetched_at": time.time(), "data": data}, out)
                os.replace(temporary, path)
            finally:
                pathlib.Path(temporary).unlink(missing_ok=True)
            return data
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"CFBD {r.status_code} on {url} {params}: {r.text[:300]}")
    raise RuntimeError(f"CFBD retries exhausted on {url} {params}")


def ping():
    """Live check that the key works. Returns number of SEC teams found."""
    teams = get("teams", conference="SEC", use_cache=False)
    return len(teams)
