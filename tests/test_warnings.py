"""Regression tests for spurious 'Config ... is not recognized' warnings.

Guards two fixes that together silence every ``Config ... is not
recognized`` line on a default ``ams.load``:

1. ``ams.core.model.Model.__init__`` no longer injects unused
   ``allow_adjust`` / ``adjust_lower`` / ``adjust_upper`` keys into
   every model's Config — those keys were a port-artifact from ANDES
   and nothing in AMS read them.
2. ``ams.system.System.__init__`` purges known-deprecated keys from
   the rc-loaded config before ``Config.check()`` runs, so a user's
   ``~/.ams/ams.rc`` written against an older AMS release doesn't
   flood the log.
"""
import io
import logging
import textwrap

import ams


def _capture_recognized_warnings(run):
    """Run ``run()``, return any 'is not recognized' WARNING lines it emits."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        run()
    finally:
        root.removeHandler(handler)
    return [line for line in buf.getvalue().splitlines() if "is not recognized" in line]


def test_no_spurious_config_warnings_with_default_config():
    """``default_config=True`` path emits no 'not recognized' warnings."""

    def _load():
        ams.load(
            ams.get_case("5bus/pjm5bus_demo.xlsx"),
            setup=True, no_output=True, default_config=True,
        )

    leftover = _capture_recognized_warnings(_load)
    assert leftover == [], (
        f"got {len(leftover)} spurious config warnings:\n"
        + "\n".join(leftover[:10])
    )


def test_deprecated_rc_keys_are_purged_silently(tmp_path):
    """A user rc carrying retired System or Model keys must be dropped quietly.

    Simulates a rc written against an older AMS: [System] carries
    ``save_stats``; every model section carries the three
    ``allow_adjust`` / ``adjust_lower`` / ``adjust_upper`` keys that
    earlier AMS used to inject and ``save_config`` then persisted.
    """
    rc = tmp_path / "ams.rc"
    rc.write_text(textwrap.dedent("""
        [System]
        freq = 60
        save_stats = 0

        [Bus]
        allow_adjust = 1
        adjust_lower = 0
        adjust_upper = 1

        [PQ]
        allow_adjust = 1
        adjust_lower = 0
        adjust_upper = 1
    """).strip())

    def _load():
        ams.load(
            ams.get_case("5bus/pjm5bus_demo.xlsx"),
            setup=True, no_output=True, config_path=str(rc),
        )

    leftover = _capture_recognized_warnings(_load)
    assert leftover == [], (
        f"got {len(leftover)} warnings from stale rc:\n"
        + "\n".join(leftover[:10])
    )
