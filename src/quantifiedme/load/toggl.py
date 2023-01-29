import logging
from typing import List
from datetime import datetime, timezone, timedelta

try:
    from toggl import api
except ModuleNotFoundError:
    pass

from aw_core import Event

logging.getLogger("toggl.utils").setLevel(logging.WARNING)


def load_toggl(start: datetime, stop: datetime) -> list[Event]:
    # The Toggl API has a query limit of 1 year
    if stop - start < timedelta(days=360):
        return _load_toggl(start, stop)
    else:
        split = stop - timedelta(days=360)
        return load_toggl(start, split) + _load_toggl(split, stop)


def _load_toggl(start: datetime, stop: datetime) -> list[Event]:
    # [x] TODO: For some reason this doesn't get all history, consider just switching back to loading from export (at least for older events)
    # The maintainer of togglcli fixed it quickly, huge thanks! https://github.com/AuHau/toggl-cli/issues/87

    def entries_from_all_workspaces() -> list[dict]:
        # FIXME: Returns entries for all users
        # FIXME: togglcli returns the same workspace for all entries
        workspaces = list(api.Workspace.objects.all())
        print(workspaces)
        print([w.id for w in workspaces])
        print(f"Found {len(workspaces)} workspaces: {list(w.name for w in workspaces)}")
        entries: list[dict] = [
            e.to_dict()
            for workspace in workspaces
            for e in api.TimeEntry.objects.all_from_reports(
                start=start, stop=stop, workspace=workspace.id
            )
        ]
        for e in entries[-10:]:
            print(e)
            # print(e["workspace"], e["project"])
        return entries

    def entries_from_main_workspace() -> list[dict]:
        entries = list(api.TimeEntry.objects.all_from_reports(start=start, stop=stop))
        return [e.to_dict() for e in entries]

    entries = entries_from_all_workspaces()
    print(f"Found {len(entries)} time entries in Toggl")
    events_toggl = []
    for e in entries:
        if e["start"] < start.astimezone(timezone.utc):
            continue
        project = e["project"].name if e["project"] else "no project"
        workspace = e["workspace"].name
        try:
            client = e["project"].client.name
        except AttributeError:
            client = "no client"
        description = e["description"]
        events_toggl.append(
            Event(
                timestamp=e["start"].isoformat(),
                duration=e["duration"],
                data={
                    "app": project,
                    "title": f"{client or 'no client'} -> {project or 'no project'} -> {description or 'no description'}",
                    "workspace": workspace,
                    "$source": "toggl",
                },
            )
        )

    return sorted(events_toggl, key=lambda e: e.timestamp)


if __name__ == "__main__":
    now = datetime.now(tz=timezone.utc)
    events = load_toggl(now - timedelta(days=10), now)
    for e in events[:10]:
        print(e["workspace"])
