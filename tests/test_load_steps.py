"""Tests for the Home Assistant steps loader. No network: the websocket is faked."""

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pytest

from quantifiedme.load import steps
from quantifiedme.load.steps import _ws_url, fetch_statistics, statistics_to_daily_df

# 2026-03-02 23:00Z / 2026-03-03 23:00Z == local midnight in Europe/Stockholm (CET)
MAR3 = int(datetime(2026, 3, 2, 23, tzinfo=timezone.utc).timestamp() * 1000)
MAR4 = int(datetime(2026, 3, 3, 23, tzinfo=timezone.utc).timestamp() * 1000)

STATS: dict[str, list[dict[str, Any]]] = {
    "sensor.phone_daily_steps": [
        {"start": MAR3, "end": MAR4, "change": 8000.0},
        {"start": MAR4, "end": MAR4 + 86_400_000, "change": 3000.0},
        # 2026-03-05: no device reported
        {"start": MAR4 + 86_400_000, "end": MAR4 + 2 * 86_400_000, "change": 0.0},
    ],
    "sensor.watch_steps_sensor": [
        {"start": MAR3, "end": MAR4, "change": 9500.0},
        {"start": MAR4, "end": MAR4 + 86_400_000, "change": -12.0},  # glitch
    ],
}


def test_ws_url() -> None:
    assert _ws_url("https://ha.example.com/") == "wss://ha.example.com/api/websocket"
    assert (
        _ws_url("http://10.0.0.2:8123", allow_insecure=True)
        == "ws://10.0.0.2:8123/api/websocket"
    )


def test_ws_url_refuses_plain_http_by_default() -> None:
    with pytest.raises(ValueError, match="plain http"):
        _ws_url("http://10.0.0.2:8123")
    with pytest.raises(ValueError, match="scheme"):
        _ws_url("ftp://ha.example.com")


def test_statistics_to_daily_df_max_across_entities() -> None:
    daily = statistics_to_daily_df(STATS, "Europe/Stockholm")
    assert list(daily.columns) == ["steps"]
    assert daily.index.name == "date"
    # local-midnight starts map to the local calendar date, not the UTC date
    assert list(daily.index) == [pd.Timestamp("2026-03-03"), pd.Timestamp("2026-03-04")]
    assert daily.loc["2026-03-03", "steps"] == 9500  # max, not sum
    assert daily.loc["2026-03-04", "steps"] == 3000  # negative glitch ignored
    assert pd.Timestamp("2026-03-05") not in daily.index  # 0 = not reporting


def test_statistics_to_daily_df_iso_start() -> None:
    stats = {"sensor.x": [{"start": "2026-03-02T23:00:00+00:00", "change": 42.0}]}
    daily = statistics_to_daily_df(stats, "Europe/Stockholm")
    assert daily.loc["2026-03-03", "steps"] == 42


def test_statistics_to_daily_df_empty() -> None:
    daily = statistics_to_daily_df({"sensor.x": []}, "UTC")
    assert daily.empty
    assert "steps" in daily.columns


class FakeWebSocket:
    """Minimal stand-in for websockets.sync.client.ClientConnection."""

    def __init__(self, auth_ok: bool = True) -> None:
        self.auth_ok = auth_ok
        self.sent: list[dict[str, Any]] = []
        self.inbox: list[dict[str, Any]] = [{"type": "auth_required"}]

    def __enter__(self) -> "FakeWebSocket":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def send(self, raw: str) -> None:
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg["type"] == "auth":
            self.inbox.append({"type": "auth_ok" if self.auth_ok else "auth_invalid"})
        elif msg["type"] == "get_config":
            self.inbox.append(
                {
                    "id": msg["id"],
                    "type": "result",
                    "success": True,
                    "result": {"time_zone": "Europe/Stockholm"},
                }
            )
        elif msg["type"] == "recorder/statistics_during_period":
            # interleave an unrelated event to check it's skipped
            self.inbox.append({"id": 99, "type": "event"})
            self.inbox.append(
                {"id": msg["id"], "type": "result", "success": True, "result": STATS}
            )

    def recv(self, timeout: float | None = None) -> str:
        return json.dumps(self.inbox.pop(0))


def _patch_connect(monkeypatch: pytest.MonkeyPatch, ws: FakeWebSocket) -> None:
    import websockets.sync.client

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: ws)


def test_fetch_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket()
    _patch_connect(monkeypatch, ws)
    stats, tz = fetch_statistics(
        "https://ha.example.com",
        "token",
        ["sensor.phone_daily_steps"],
        datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    assert tz == "Europe/Stockholm"
    assert stats == STATS
    assert ws.sent[0] == {"type": "auth", "access_token": "token"}
    query = ws.sent[2]
    assert query["period"] == "day"
    assert query["types"] == ["change"]
    assert query["statistic_ids"] == ["sensor.phone_daily_steps"]


def test_fetch_statistics_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch, FakeWebSocket(auth_ok=False))
    with pytest.raises(RuntimeError, match="auth failed"):
        fetch_statistics("https://ha.example.com", "bad", ["sensor.x"], datetime.now())


def test_load_daily_df(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    token_file = tmp_path / "ha_token"
    token_file.write_text("secret\n")
    config = {
        "data": {
            "steps": {
                "ha_url": "https://ha.example.com",
                "ha_token_file": str(token_file),
                "entities": list(STATS),
            }
        }
    }
    monkeypatch.setattr(steps, "load_config", lambda: config)
    monkeypatch.delenv("HA_TOKEN", raising=False)
    ws = FakeWebSocket()
    _patch_connect(monkeypatch, ws)

    daily = steps.load_daily_df()
    assert ws.sent[0]["access_token"] == "secret"
    assert daily.loc["2026-03-03", "steps"] == 9500


def test_load_daily_df_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(steps, "load_config", lambda: {"data": {}})
    with pytest.raises(KeyError):
        steps.load_daily_df()
