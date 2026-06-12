"""Shared low-level reader for "SMS Backup & Restore" XML backups.

Erik's Android phone runs the "SMS Backup and Restore" app, which writes daily
XML backups (``calls-*.xml``, ``sms-*.xml``) into a Syncthing folder. Two agents
consume the same raw backups for different purposes:

- **Alice** (``alice/scripts/comms_summary.py``) wants privacy-safe per-day
  *aggregates* (counts, call minutes, distinct-contact counts) for the QS
  connection-dimension signal.
- **erbot** (``erbot/scripts/extract_erik_sms.py``) wants Erik's *sent* SMS
  *text* (type==2), run through erbot's own private blocklist, to build a
  "What Would Erik Do" corpus.

Both need the same low-level parsing: stream the (large) XML with ``iterparse``
so message bodies never accumulate in memory, normalize phone numbers the same
way, and pick the newest backup file. This module single-sources exactly that
shared layer and nothing more.

PRIVACY CONTRACT
----------------
This module *reads* raw attributes (which include bodies and numbers) but
**never persists, hashes, logs, or returns derived-but-still-identifying
values on its own initiative**. It hands raw attribute dicts to the caller and
stops there. The two privacy-sensitive decisions stay with each consumer,
because their requirements genuinely differ:

- **Hashing**: Alice needs a *salted, per-process, non-reversible, never-stored*
  contact key (distinct-count only). erbot needs a *stable, unsalted* contact
  hash so the same contact dedups across runs. Those are incompatible, so this
  module only offers :func:`normalize_number` (the shared pre-hash step) — the
  actual hashing belongs to each consumer.
- **Body filtering**: Alice drops bodies entirely; erbot runs them through its
  private blocklist. Neither belongs here.

So: no raw values leave this module except inside the attribute dicts the caller
explicitly asked for. This file contains only parsing logic — no data.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import iterparse

if TYPE_CHECKING:
    from collections.abc import Iterator
    from xml.etree.ElementTree import Element

# Call "type" codes used by the app (Android CallLog.Calls.TYPE).
CALL_TYPES = {
    "1": "incoming",
    "2": "outgoing",
    "3": "missed",
    "4": "voicemail",
    "5": "rejected",
    "6": "blocked",
}
# SMS "type" codes (Android Telephony.Sms.MESSAGE_TYPE).
SMS_RECEIVED = "1"
SMS_SENT = "2"


def iter_records(path: str | os.PathLike, tag: str) -> Iterator[dict[str, str]]:
    """Stream ``<tag>`` elements' attributes, clearing each so memory stays bounded.

    ``iterparse`` yields elements as they close; we copy the attributes,
    ``clear()`` the element, and purge it from the root so the (potentially
    40MB+) document never materializes in memory. Yields a plain ``dict`` per
    record so callers can't accidentally retain live XML nodes.
    """
    root: Element | None = None
    for event, elem in iterparse(str(path), events=("start", "end")):
        if event == "start" and root is None:
            root = elem
        elif event == "end" and elem.tag == tag:
            yield dict(elem.attrib)
            elem.clear()
            if root is not None:
                del root[:]  # purge cleared child from parent; keeps root list bounded


def iter_sent_sms(path: str | os.PathLike) -> Iterator[dict[str, str]]:
    """Yield attribute dicts for *sent* SMS (type==2) from an ``sms-*.xml`` file.

    Convenience wrapper for the erbot corpus path, which only cares about Erik's
    own outgoing texts. Received messages and MMS are skipped.
    """
    for attrs in iter_records(path, "sms"):
        if attrs.get("type") == SMS_SENT:
            yield attrs


def normalize_number(raw: str | None) -> str:
    """Normalize a phone number / address to digits and a leading ``+`` only.

    Shared pre-hash step so Alice's salted distinct-count key and erbot's stable
    contact hash operate on identical input. Returns ``""`` for a missing value;
    callers decide how to represent "unknown".
    """
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    prefix = "+" if raw.lstrip().startswith("+") else ""
    return prefix + digits


def epoch_ms_to_day(date_ms: str | None, tz: timezone | None = None) -> str | None:
    """Convert an epoch-millis string to ``YYYY-MM-DD``.

    ``tz=None`` uses system local time — on erb-m2 that is Erik's timezone, which
    is what "which day of Erik's life did this happen" wants. Pass an explicit
    ``tz`` for deterministic tests. Returns ``None`` for missing/garbage input.
    """
    dt = _epoch_ms_to_dt(date_ms, tz)
    return dt.strftime("%Y-%m-%d") if dt else None


def epoch_ms_to_iso(date_ms: str | None, tz: timezone | None = None) -> str | None:
    """Convert an epoch-millis string to an ISO-8601 timestamp.

    Used by the erbot corpus path (``iso_timestamp``) to stamp each message.
    Returns ``None`` for missing/garbage input.
    """
    dt = _epoch_ms_to_dt(date_ms, tz)
    return dt.isoformat() if dt else None


def _epoch_ms_to_dt(date_ms: str | None, tz: timezone | None) -> datetime | None:
    if not date_ms:
        return None
    try:
        seconds = int(date_ms) / 1000.0
    except (ValueError, TypeError):
        return None
    # tz=None is intentional: it selects system local time, which on erb-m2 is
    # Erik's timezone ("which day of Erik's life"). Callers pass an explicit tz
    # for deterministic behavior; tests always do.
    return datetime.fromtimestamp(seconds, tz=tz)


def latest_file(comms_dir: str | os.PathLike, prefix: str) -> Path | None:
    """Newest file matching ``<prefix>*.xml`` in ``comms_dir``.

    The app stamps each backup's filename with a timestamp, so lexical sort by
    filename matches chronological order. Returns ``None`` if no file matches.
    ``~`` in ``comms_dir`` is expanded.
    """
    base = Path(os.path.expanduser(str(comms_dir)))
    matches = sorted(glob.glob(str(base / f"{prefix}*.xml")))
    return Path(matches[-1]) if matches else None
