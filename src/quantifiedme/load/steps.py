"""
Loads daily step counts from Home Assistant long-term statistics.

The Home Assistant companion app exposes step sensors (e.g. Android's
``sensor.<phone>_daily_steps`` from Health Connect, or
``sensor.<device>_steps_sensor`` from the hardware step counter). These are
``total_increasing`` sensors, so HA keeps *long-term statistics* for them
indefinitely (unlike the ``states`` table, which is purged after
``purge_keep_days``). The daily ``change`` of such a statistic is the number of
steps taken that day, even across counter resets.

Statistics are only exposed over the websocket API
(``recorder/statistics_during_period``), so this loader talks websocket
directly (via the ``websockets`` package).

Configure in ``config.toml``::

    [data.steps]
    ha_url = "https://homeassistant.local:8123"
    # long-lived access token: read from $HA_TOKEN, else from this file
    ha_token_file = "~/.config/quantifiedme/ha_token"
    # allow_insecure = true  # permit a plain http:// URL (token sent unencrypted)
    entities = ["sensor.phone_daily_steps", "sensor.watch_steps_sensor"]

When several entities are configured (e.g. phone + watch), the daily value is
the **maximum** across them: they count the same steps, so summing would
double-count, and the max favors whichever device was carried/worn more.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_config

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 5 * 365


def _ws_url(base_url: str, allow_insecure: bool = False) -> str:
    """HA base URL → websocket URL. Plain http is refused unless explicitly
    allowed, since the access token is sent over the connection."""
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/api/websocket"
    if base.startswith("http://"):
        if not allow_insecure:
            raise ValueError(
                "Refusing to send the Home Assistant token over plain http; use an "
                "https URL or set data.steps.allow_insecure = true"
            )
        return "ws://" + base[len("http://") :] + "/api/websocket"
    raise ValueError(f"Unsupported Home Assistant URL scheme: {base_url}")


def _load_token(cfg: dict[str, Any]) -> str:
    token = os.environ.get("HA_TOKEN")
    if token:
        return token.strip()
    token_file = cfg.get("ha_token_file")
    if token_file:
        path = Path(token_file).expanduser()
        if path.exists():
            return path.read_text().strip()
        raise FileNotFoundError(f"Home Assistant token file not found at {path}")
    raise KeyError("No Home Assistant token: set $HA_TOKEN or data.steps.ha_token_file")


def fetch_statistics(
    url: str,
    token: str,
    statistic_ids: list[str],
    start: datetime,
    period: str = "day",
    types: tuple[str, ...] = ("change",),
    allow_insecure: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Fetch long-term statistics over the HA websocket API.

    Returns ``(statistics, time_zone)`` where ``statistics`` maps statistic id →
    list of rows (``start`` in epoch ms plus the requested ``types``), and
    ``time_zone`` is HA's configured time zone (day periods are aligned to its
    local midnight).
    """
    from websockets.sync.client import connect

    with connect(_ws_url(url, allow_insecure), max_size=None, open_timeout=30) as ws:

        def recv() -> dict[str, Any]:
            return json.loads(ws.recv(timeout=120))

        msg = recv()
        if msg.get("type") != "auth_required":
            raise RuntimeError(f"Unexpected HA websocket greeting: {msg.get('type')}")
        ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = recv()
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"HA websocket auth failed: {msg.get('type')}")

        ws.send(json.dumps({"id": 1, "type": "get_config"}))
        ws.send(
            json.dumps(
                {
                    "id": 2,
                    "type": "recorder/statistics_during_period",
                    "start_time": start.astimezone(timezone.utc).isoformat(),
                    "statistic_ids": statistic_ids,
                    "period": period,
                    "types": list(types),
                }
            )
        )
        results: dict[int, Any] = {}
        while len(results) < 2:
            msg = recv()
            if msg.get("type") != "result" or msg.get("id") not in (1, 2):
                continue
            if not msg.get("success"):
                raise RuntimeError(f"HA websocket command failed: {msg.get('error')}")
            results[msg["id"]] = msg["result"]

    return results[2], results[1].get("time_zone", "UTC")


def statistics_to_daily_df(
    stats: dict[str, list[dict[str, Any]]], time_zone: str
) -> pd.DataFrame:
    """Daily ``change`` statistics → DataFrame with one ``steps`` column.

    Index: local calendar date (naive ``DatetimeIndex``). Value: max across
    entities of the day's step count. Zero changes (the sensor didn't report,
    e.g. a watch that wasn't worn) and negative changes (sensor glitches) are
    treated as missing, so such days are dropped rather than recorded as 0.
    """
    frames = []
    for sid, rows in stats.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "change" not in df.columns:
            continue
        start = df["start"]
        ts = (
            pd.to_datetime(start, unit="ms", utc=True)
            if pd.api.types.is_numeric_dtype(start)
            else pd.to_datetime(start, utc=True)
        )
        dates = ts.dt.tz_convert(time_zone).dt.tz_localize(None).dt.normalize()
        frames.append(pd.Series(df["change"].to_numpy(), index=dates, name=sid))

    if not frames:
        out = pd.DataFrame(columns=["steps"], dtype=float)
        out.index = pd.DatetimeIndex([], name="date")
        return out

    wide = pd.concat(frames, axis=1)
    wide = wide.where(wide > 0)
    daily = wide.max(axis=1).dropna().to_frame("steps")
    daily.index.name = "date"
    return daily.sort_index()


def load_daily_df(days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Load daily step counts from Home Assistant (see module docstring)."""
    config = load_config()
    cfg = config["data"]["steps"]
    entities = list(cfg["entities"])
    token = _load_token(cfg)
    start = datetime.now(tz=timezone.utc) - timedelta(days=days)
    stats, time_zone = fetch_statistics(
        cfg["ha_url"],
        token,
        entities,
        start,
        allow_insecure=bool(cfg.get("allow_insecure", False)),
    )
    missing = [e for e in entities if not stats.get(e)]
    if missing:
        logger.warning(f"No step statistics for: {missing}")
    return statistics_to_daily_df(stats, time_zone)
