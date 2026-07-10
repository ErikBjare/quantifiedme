"""
Loads Whoop data from the official WHOOP API (v2).

Unlike :mod:`.whoop` (which reads static CSV/GDPR exports that go stale the
moment they're downloaded), this loader pulls fresh data over OAuth2 so daily
exports keep advancing without manual re-downloads.

Setup
-----
1. Create an app at https://developer-dashboard.whoop.com/ with redirect URL
   ``http://localhost:8888``.
2. Put credentials in ``.env.whoop`` at the repo root (or set the
   ``WHOOP_CLIENT_ID``/``WHOOP_CLIENT_SECRET`` env vars)::

       CLIENT_ID=...
       CLIENT_SECRET=...

3. Run ``python -m quantifiedme.load.whoop_api auth`` and approve in the
   browser. Tokens are persisted and refreshed automatically from then on.

Notes
-----
- WHOOP rotates refresh tokens: every refresh invalidates the previous
  refresh token, so the rotated token is saved to disk immediately.
- Records are cached on disk (JSON, keyed by record id) and fetched
  incrementally with a re-fetch overlap window, since scores can be
  updated after the fact.
- The journal (self-reports) is not exposed by the API; use the CSV export
  loader in :mod:`.whoop` for that.
"""

import argparse
import json
import logging
import os
import secrets
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import platformdirs
import requests

from ..config import rootdir

logger = logging.getLogger(__name__)

AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE = "https://api.prod.whoop.com/developer/v2"

# `offline` is required to get a refresh token at all
SCOPES = (
    "offline read:recovery read:cycles read:sleep read:workout "
    "read:profile read:body_measurement"
)

REDIRECT_URI = os.environ.get("WHOOP_REDIRECT_URI", "http://localhost:8888")

# Scores can be revised after initial scoring; re-fetch this far back past the
# newest cached record on incremental updates.
FETCH_OVERLAP = timedelta(days=14)

KCAL_PER_KILOJOULE = 1 / 4.184


# ── Credentials & token persistence ───────────────────────────────────────────


def _load_credentials() -> tuple[str, str]:
    """Client id/secret from env vars, falling back to ``.env.whoop``."""
    client_id = os.environ.get("WHOOP_CLIENT_ID")
    client_secret = os.environ.get("WHOOP_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    env_file = rootdir / ".env.whoop"
    if env_file.exists():
        env: dict[str, str] = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
        if "CLIENT_ID" in env and "CLIENT_SECRET" in env:
            return env["CLIENT_ID"], env["CLIENT_SECRET"]

    raise RuntimeError(
        "Whoop API credentials not found. Set WHOOP_CLIENT_ID/WHOOP_CLIENT_SECRET "
        f"or create {env_file} with CLIENT_ID=... and CLIENT_SECRET=..."
    )


def _token_path() -> Path:
    return Path(platformdirs.user_data_dir("quantifiedme")) / "whoop_token.json"


def _save_token(token: dict[str, Any]) -> None:
    """Persist a token response, converting expires_in → absolute expires_at."""
    if "expires_in" in token and "expires_at" not in token:
        token["expires_at"] = time.time() + token["expires_in"]
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token))
    path.chmod(0o600)


def _load_token() -> dict[str, Any] | None:
    path = _token_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def has_auth() -> bool:
    """Whether an OAuth token exists (i.e. `auth` has been run on this machine)."""
    return _token_path().exists()


# ── OAuth flow ────────────────────────────────────────────────────────────────


def _exchange_code(code: str) -> dict[str, Any]:
    client_id, client_secret = _load_credentials()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _refresh_token(token: dict[str, Any]) -> dict[str, Any]:
    """Refresh the access token. WHOOP rotates refresh tokens, so the caller
    must persist the returned token immediately (the old one is now invalid)."""
    client_id, client_secret = _load_credentials()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": token["refresh_token"],
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "offline",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def authorize(open_browser: bool = True) -> None:
    """Run the interactive OAuth2 authorization-code flow.

    Starts a one-shot HTTP server on the redirect port, opens the browser to
    the WHOOP consent page, exchanges the returned code, and persists tokens.
    """
    client_id, _ = _load_credentials()
    state = secrets.token_urlsafe(16)
    port = urlparse(REDIRECT_URI).port or 80

    url = (
        AUTH_URL
        + "?"
        + urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": REDIRECT_URI,
                "scope": SCOPES,
                "state": state,
            }
        )
    )

    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlparse(self.path).query)
            if "code" in params and params.get("state", [None])[0] == state:
                result["code"] = params["code"][0]
                body = b"Authorized! You can close this tab."
            else:
                result["error"] = params.get("error", ["unknown"])[0]
                body = b"Authorization failed, check terminal."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence request logging
            pass

    print(f"Waiting for authorization on {REDIRECT_URI} ...")
    print(f"If no browser opens, visit:\n\n  {url}\n")
    if open_browser:
        webbrowser.open(url)

    with HTTPServer(("localhost", port), Handler) as server:
        while not result:
            server.handle_request()

    if "error" in result:
        raise RuntimeError(f"Authorization failed: {result['error']}")

    _save_token(_exchange_code(result["code"]))
    print(f"Authorized! Token saved to {_token_path()}")


def _access_token() -> str:
    """Return a valid access token, refreshing (and persisting) if expired."""
    token = _load_token()
    if token is None:
        raise RuntimeError(
            "No Whoop API token. Run: python -m quantifiedme.load.whoop_api auth"
        )
    if token.get("expires_at", 0) < time.time() + 60:
        token = _refresh_token(token)
        _save_token(token)
    return token["access_token"]


# ── API client ────────────────────────────────────────────────────────────────


def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    for attempt in range(5):
        resp = requests.get(
            API_BASE + path,
            params=params,
            headers={"Authorization": f"Bearer {_access_token()}"},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            logger.info("Rate limited, sleeping %ds", wait)
            time.sleep(wait)
            continue
        if resp.status_code == 401 and attempt == 0:
            # Access token invalidated server-side; force a refresh and retry
            token = _load_token()
            if token:
                _save_token(_refresh_token(token))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Giving up on {path} after repeated rate limits")


def _paginate(path: str, start: datetime | None = None) -> list[dict[str, Any]]:
    """Fetch all records from a paginated collection endpoint."""
    records: list[dict[str, Any]] = []
    params: dict[str, Any] = {"limit": 25}
    if start is not None:
        params["start"] = start.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
    while True:
        page = _api_get(path, params)
        records += page.get("records", [])
        # response uses snake_case `next_token`; the query param is camelCase
        next_token = page.get("next_token")
        if not next_token:
            break
        params["nextToken"] = next_token
    return records


# ── Cached collection fetching ────────────────────────────────────────────────

COLLECTIONS = {
    "cycles": ("/cycle", "id", "start"),
    "recoveries": ("/recovery", "cycle_id", "created_at"),
    "sleeps": ("/activity/sleep", "id", "start"),
    "workouts": ("/activity/workout", "id", "start"),
}


def _cache_path(name: str) -> Path:
    return Path(platformdirs.user_cache_dir("quantifiedme")) / "whoop" / f"{name}.json"


def fetch_collection(name: str, use_cache: bool = True) -> list[dict[str, Any]]:
    """Fetch a collection, incrementally updating the on-disk cache.

    Only records newer than (newest cached − FETCH_OVERLAP) are fetched; the
    overlap window picks up late score revisions. Cached records are upserted
    by id, so re-fetches replace rather than duplicate.
    """
    path_api, id_key, ts_key = COLLECTIONS[name]
    cache_file = _cache_path(name)

    cached: dict[str, dict[str, Any]] = {}
    start: datetime | None = None
    if use_cache and cache_file.exists():
        cached = {str(r[id_key]): r for r in json.loads(cache_file.read_text())}
        if cached:
            newest = max(r[ts_key] for r in cached.values())
            start = (
                datetime.fromisoformat(newest.replace("Z", "+00:00")) - FETCH_OVERLAP
            )

    fresh = _paginate(path_api, start=start)
    logger.info("Fetched %d %s records (cache had %d)", len(fresh), name, len(cached))
    for r in fresh:
        cached[str(r[id_key])] = r

    records = sorted(cached.values(), key=lambda r: r[ts_key])
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(records))
    return records


# ── Record → DataFrame mapping ────────────────────────────────────────────────
#
# Output schemas match the standard-CSV loaders in .whoop, so the derived
# modules (sleep.py, all_df.py) work unchanged regardless of source.


def _local_date(iso_utc: str, tz_offset: str) -> pd.Timestamp:
    """UTC timestamp + WHOOP tz offset ("+02:00"/"Z") → local calendar date (UTC-indexed)."""
    ts = pd.Timestamp(iso_utc)
    offset = (
        pd.Timedelta(0)
        if tz_offset in ("Z", "")
        else pd.Timedelta(
            hours=int(tz_offset[:3]), minutes=int(tz_offset[0] + tz_offset[4:6])
        )
    )
    return pd.Timestamp((ts + offset).date(), tz="UTC")


def _sleeps_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Sleep records → daily sleep df (same schema as whoop._load_sleep_standard).

    Index: wake date (local date of sleep end). Naps and unscored excluded.
    """
    rows = []
    for r in records:
        if r.get("nap") or r.get("score_state") != "SCORED":
            continue
        score = r["score"]
        stages = score["stage_summary"]
        ms = pd.to_timedelta
        asleep = (
            stages["total_light_sleep_time_milli"]
            + stages["total_slow_wave_sleep_time_milli"]
            + stages["total_rem_sleep_time_milli"]
        )
        rows.append(
            {
                "timestamp": _local_date(r["end"], r.get("timezone_offset", "Z")),
                "score": score.get("sleep_performance_percentage"),
                "duration": ms(asleep, unit="ms"),
                "time_in_bed": ms(stages["total_in_bed_time_milli"], unit="ms"),
                "efficiency": score.get("sleep_efficiency_percentage"),
                "consistency": score.get("sleep_consistency_percentage"),
                "debt": (score.get("sleep_needed") or {}).get(
                    "need_from_sleep_debt_milli", 0
                )
                / 60_000,
                "respiratory_rate": score.get("respiratory_rate"),
                "rem": ms(stages["total_rem_sleep_time_milli"], unit="ms"),
                "deep": ms(stages["total_slow_wave_sleep_time_milli"], unit="ms"),
                "light": ms(stages["total_light_sleep_time_milli"], unit="ms"),
                "awake": ms(stages["total_awake_time_milli"], unit="ms"),
            }
        )
    return _to_daily_df(rows)


def _cycles_to_df(
    cycles: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
    sleeps: list[dict[str, Any]],
) -> pd.DataFrame:
    """Cycle+recovery records → daily cycles df (same schema as whoop._load_cycles_standard).

    Recoveries are the spine (one per scored day); strain/energy joined from
    the cycle, wake date from the associated sleep (falls back to the
    recovery's created_at, which is stamped at wake).
    """
    cycle_by_id = {c["id"]: c for c in cycles}
    sleep_by_id = {s["id"]: s for s in sleeps}

    rows = []
    for r in recoveries:
        if r.get("score_state") != "SCORED":
            continue
        score = r["score"]

        sleep = sleep_by_id.get(r.get("sleep_id"))
        if sleep is not None:
            date = _local_date(sleep["end"], sleep.get("timezone_offset", "Z"))
        else:
            date = _local_date(r["created_at"], "Z")

        cycle_score = (cycle_by_id.get(r["cycle_id"]) or {}).get("score") or {}
        kilojoule = cycle_score.get("kilojoule")
        rows.append(
            {
                "timestamp": date,
                "recovery": score.get("recovery_score"),
                "resting_hr": score.get("resting_heart_rate"),
                "hrv": score.get("hrv_rmssd_milli"),
                "skin_temp": score.get("skin_temp_celsius"),
                "spo2": score.get("spo2_percentage"),
                "strain": cycle_score.get("strain"),
                "energy_kcal": kilojoule * KCAL_PER_KILOJOULE if kilojoule else None,
            }
        )
    return _to_daily_df(rows)


def _workouts_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Workout records → event df (same schema as whoop._load_workouts_standard)."""
    rows = []
    for r in records:
        score = r.get("score") or {}
        start = pd.Timestamp(r["start"])
        end = pd.Timestamp(r["end"])
        kilojoule = score.get("kilojoule")
        rows.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "activity": r.get("sport_name"),
                "strain": score.get("strain"),
                "energy_kcal": kilojoule * KCAL_PER_KILOJOULE if kilojoule else None,
                "max_hr": score.get("max_heart_rate"),
                "avg_hr": score.get("average_heart_rate"),
            }
        )
    columns = [
        "start",
        "end",
        "duration",
        "activity",
        "strain",
        "energy_kcal",
        "max_hr",
        "avg_hr",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("start").reset_index(drop=True)


def _to_daily_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Multiple records can map to the same wake date (e.g. split sleeps);
    # keep the last (most recently scored)
    df = df.groupby("timestamp").last()
    df.index.name = "timestamp"
    return df.sort_index()


# ── Public API ────────────────────────────────────────────────────────────────


def load_sleep_df() -> pd.DataFrame:
    """Load daily sleep summary from the WHOOP API."""
    return _sleeps_to_df(fetch_collection("sleeps"))


def load_cycles_df() -> pd.DataFrame:
    """Load daily physiological summary (recovery, HRV, RHR, strain) from the WHOOP API."""
    return _cycles_to_df(
        fetch_collection("cycles"),
        fetch_collection("recoveries"),
        fetch_collection("sleeps"),
    )


def load_workouts_df() -> pd.DataFrame:
    """Load workout events from the WHOOP API."""
    return _workouts_to_df(fetch_collection("workouts"))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="WHOOP API v2 client")
    sub = parser.add_subparsers(dest="cmd", required=True)
    auth_cmd = sub.add_parser("auth", help="run interactive OAuth authorization")
    auth_cmd.add_argument(
        "--no-browser", action="store_true", help="print URL instead of opening browser"
    )
    sub.add_parser("status", help="show auth status and profile")
    sub.add_parser("fetch", help="fetch all collections into the local cache")
    args = parser.parse_args()

    if args.cmd == "auth":
        authorize(open_browser=not args.no_browser)
    elif args.cmd == "status":
        if not has_auth():
            print("Not authorized. Run: python -m quantifiedme.load.whoop_api auth")
            return
        profile = _api_get("/user/profile/basic")
        print(
            f"Authorized as {profile.get('first_name')} {profile.get('last_name')}"
            f" ({profile.get('email')})"
        )
        print(f"Token file: {_token_path()}")
    elif args.cmd == "fetch":
        for name in COLLECTIONS:
            records = fetch_collection(name)
            _, _, ts_key = COLLECTIONS[name]
            newest = max((r[ts_key] for r in records), default="n/a")
            print(f"{name}: {len(records)} records, newest {newest}")
        df = load_cycles_df()
        print(
            f"\ncycles df: {len(df)} days, {df.index.min().date()} → {df.index.max().date()}"
        )


if __name__ == "__main__":
    main()
