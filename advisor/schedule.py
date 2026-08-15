"""What to paste to make the advisor keep itself current.

This prints unit files rather than installing them. Editing someone's crontab
or enabling a systemd unit behind their back is a poor trade for the two
seconds it saves, and the printed version is also the documentation — you can
read exactly what would run before anything runs.

Both forms call ``advisor update``, which is the whole nightly job: harvest,
embed within a budget, then build a fresh feed.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_TIME = "06:00"


def executable() -> Path:
    """The ``advisor`` entry point beside the running interpreter.

    Resolved rather than assumed: this typically lives in a virtualenv, and
    both cron and systemd run with a minimal PATH that will not find it.
    """
    candidate = Path(sys.executable).parent / "advisor"
    if candidate.exists():
        return candidate
    return Path(sys.argv[0]).resolve()


def systemd_units(command: str, at: str = DEFAULT_TIME) -> tuple[str, str]:
    """The service and timer, as (service_text, timer_text)."""
    service = f"""\
[Unit]
Description=Research advisor: harvest, embed and refresh recommendations

[Service]
Type=oneshot
ExecStart={command} update --quiet
# The embed pass is CPU-hungry; keep the machine usable while it runs.
Nice=10
IOSchedulingClass=idle
"""

    timer = f"""\
[Unit]
Description=Nightly research advisor update

[Timer]
OnCalendar=*-*-* {at}:00
# Catch up after the machine was asleep at the scheduled time.
Persistent=true
RandomizedDelaySec=15m

[Install]
WantedBy=timers.target
"""
    return service, timer


def crontab_line(command: str, at: str = DEFAULT_TIME) -> str:
    hour, minute = at.split(":")
    return f"{int(minute)} {int(hour)} * * *  {command} update --quiet"


def instructions(at: str = DEFAULT_TIME) -> str:
    command = executable()
    service, timer = systemd_units(str(command), at)
    line = crontab_line(str(command), at)

    return f"""\
Nothing here has been installed — this is what to paste.

# systemd (recommended: survives reboots, logs to journalctl)

  Write ~/.config/systemd/user/advisor.service

{_indent(service)}
  Write ~/.config/systemd/user/advisor.timer

{_indent(timer)}
  Then enable it:

    systemctl --user daemon-reload
    systemctl --user enable --now advisor.timer
    systemctl --user list-timers advisor.timer     # confirm the next run
    journalctl --user -u advisor.service -f        # watch it work

# cron (if you would rather not use systemd)

    crontab -e

  and add:

    {line}

Either way the first run after a long gap does the most work. `advisor update`
bounds the embed pass so it cannot run past its window; whatever is left is
picked up the following night.
"""


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines()) + "\n"
