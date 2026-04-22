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

Also locks the contract that deprecated keys do not round-trip through
``save_config()`` — running ``ams misc --save-config`` on a stale rc
must produce a clean rc with no retired keys.
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


def test_deprecated_keys_are_not_re_persisted_by_save_config(tmp_path):
    """``save_config()`` after loading a stale rc must not re-write the retired keys.

    Catches the scenario where a user regenerates their rc via
    ``ams misc --save-config`` — the output file must be free of
    deprecated keys, otherwise they'd round-trip indefinitely.
    Specifically guards against a regression where ``Config._dict``
    (the backing store used by ``as_dict()``) retains retired keys
    even after ``__dict__`` is purged.
    """
    stale_rc = tmp_path / "stale.rc"
    stale_rc.write_text(textwrap.dedent("""
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

    ss = ams.load(
        ams.get_case("5bus/pjm5bus_demo.xlsx"),
        setup=True, no_output=True, config_path=str(stale_rc),
    )

    out_rc = tmp_path / "regenerated.rc"
    ss.save_config(file_path=str(out_rc), overwrite=True)
    written = out_rc.read_text()

    for deprecated in ("save_stats", "allow_adjust", "adjust_lower", "adjust_upper"):
        assert deprecated not in written, (
            f"deprecated key {deprecated!r} was re-persisted by save_config:\n"
            + written
        )
