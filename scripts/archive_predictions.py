"""Persist predictions in the prediction-history branch via GitHub Contents API.

The history branch is separate from main so scheduled snapshots do not trigger
source deployments. Timestamped files are immutable; latest is updated.
"""
import base64
import json
import os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
BRANCH = "prediction-history"
API = f"https://api.github.com/repos/{REPO}"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
           "X-GitHub-Api-Version": "2022-11-28"}

def ensure_branch():
    r = requests.get(f"{API}/git/ref/heads/{BRANCH}", headers=HEADERS, timeout=30)
    if r.status_code == 200:
        return
    r.raise_for_status() if r.status_code != 404 else None
    base = requests.get(f"{API}/git/ref/heads/main", headers=HEADERS, timeout=30)
    base.raise_for_status()
    created = requests.post(f"{API}/git/refs", headers=HEADERS, timeout=30,
                            json={"ref": f"refs/heads/{BRANCH}",
                                  "sha": base.json()["object"]["sha"]})
    created.raise_for_status()

def put(path, data, immutable):
    url = f"{API}/contents/{path}"
    current = requests.get(url, headers=HEADERS, params={"ref": BRANCH}, timeout=30)
    if current.status_code == 200 and immutable:
        raise RuntimeError(f"Snapshot already exists: {path}")
    if current.status_code not in (200, 404):
        current.raise_for_status()
    payload = {"message": f"Archive prediction snapshot: {path}", "branch": BRANCH,
               "content": base64.b64encode(data).decode("ascii")}
    if current.status_code == 200:
        payload["sha"] = current.json()["sha"]
    response = requests.put(url, headers=HEADERS, json=payload, timeout=60)
    response.raise_for_status()

def main():
    ensure_branch()
    latest = ROOT / "output" / "predictions_latest.json"
    data = latest.read_bytes()
    stamp = json.loads(data)["generated_at"].replace(":", "-").replace("+00:00", "Z")
    put(f"predictions/{stamp}.json", data, immutable=True)
    put("predictions/latest.json", data, immutable=False)
    print(f"[archive] predictions/{stamp}.json committed to {BRANCH}")

if __name__ == "__main__":
    main()
