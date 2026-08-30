"""Thin CollegeFootballData API client with on-disk caching.

Cache is keyed by endpoint+params so re-runs never re-hit the API.
Key resolution order: CFBD_API_KEY env var -> ~/.cfbd_key file -> config.
"""
import os, json, time, hashlib, pathlib
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


def get(endpoint, use_cache=True, **params):
    """GET an endpoint, returning parsed JSON (list of dicts). Cached to disk."""
    params = {k: v for k, v in params.items() if v is not None}
    sig = hashlib.md5(
        (endpoint + json.dumps(params, sort_keys=True)).encode()
    ).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    path = CACHE / f"{safe}__{sig}.json"

    if use_cache and path.exists():
        return json.loads(path.read_text())

    url = f"{BASE}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {_key()}", "Accept": "application/json"}

    for attempt in range(5):
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code == 200:
            data = r.json()
            path.write_text(json.dumps(data))
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
