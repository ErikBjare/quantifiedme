import logging
from typing import List
from datetime import datetime, timezone, timedelta

try:
    from toggl import api
except ModuleNotFoundError:
    pass

from aw_core import Event

logging.getLogger("toggl.utils").setLevel(logging.WARNING)


def load_toggl(start: datetime, stop: datetime) -> List[Event]:
    # The Toggl API has a query limit of 1 year
    if stop - start < timedelta(days=360):
        return _load_toggl(start, stop)
    else:
        split = stop - timedelta(days=360)
        return load_toggl(start, split) + _load_toggl(split, stop)


def _load_toggl(start: datetime, stop: datetime) -> List[Event]:
    # [x] TODO: For some reason this doesn't get all history, consider just switching back to loading from export (at least for older events)
    # The maintainer of togglcli fixed it quickly, huge thanks! https://github.com/AuHau/toggl-cli/issues/87

    def entries_from_all_workspaces():
        # [ ] TODO: Several issues, such as not setting the user of each TimeEntry and setting the same workspace on every TimeEntry
        workspaces = list(api.Workspace.objects.all())
        print(f"Found {len(workspaces)} workspaces: {list(w.name for w in workspaces)}")
        entries = __builtins__.sum(
            [
                list(
                    api.TimeEntry.objects.all_from_reports(
                        start=start, stop=stop, workspace=workspace
                    )
                )
                for workspace in workspaces
            ],
            [],
        )
        for e in entries[-10:]:
            print(e["workspace"], e["project"])
        return [e.to_dict() for e in entries]

    def entries_from_main_workspace():
        entries = list(api.TimeEntry.objects.all_from_reports(start=start, stop=stop))
        return [e.to_dict() for e in entries]

    entries = entries_from_main_workspace()
    print(f"Found {len(entries)} time entries in Toggl")
    events_toggl = []
    for e in entries:
        if e["start"] < start.astimezone(timezone.utc):
            continue
        project = e["project"].name if e["project"] else "no project"
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
                    "$source": "toggl",
                },
            )
        )

    return sorted(events_toggl, key=lambda e: e.timestamp)
