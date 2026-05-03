"""
Tests for ``tools/migrate_timeslot.py`` — the v1.3.0 case-file
migrator that splits legacy ``EDTSlot`` / ``UCTSlot`` CSV-list
cells into the new per-axis tables and renames the top-level keys.
"""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import pytest

# tools/ isn't on the install path; add it for the test session.
_TOOLS = Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(_TOOLS))

from migrate_timeslot import migrate_data, _rename_legacy_keys  # noqa: E402


def _legacy_case():
    """Minimal legacy case dict with both EDTSlot and UCTSlot."""
    return {
        'Area': [{'idx': 1, 'u': 1.0, 'name': '1'},
                 {'idx': 2, 'u': 1.0, 'name': '2'}],
        'Slack': [{'idx': 'S1', 'u': 1.0, 'name': 'S1', 'bus': 0}],
        'PV': [{'idx': 'G2', 'u': 1.0, 'name': 'G2', 'bus': 1}],
        'EDTSlot': [
            {'idx': 'EDT1', 'u': 1.0, 'name': 'EDT1',
             'sd': '0.5,0.7', 'ug': '1,1'},
            {'idx': 'EDT2', 'u': 1.0, 'name': 'EDT2',
             'sd': '0.6,0.8', 'ug': '1,0'},
        ],
        'UCTSlot': [
            {'idx': 'UCT1', 'u': 1.0, 'name': 'UCT1', 'sd': '0.5,0.7'},
        ],
    }


class TestMigrate(unittest.TestCase):

    def test_legacy_csv_split(self):
        """Legacy CSV-list cells split into the right per-axis tables."""
        out, summary = migrate_data(_legacy_case())
        self.assertEqual(summary['ed_load'], 4)   # 2 areas * 2 slots
        self.assertEqual(summary['ed_gen'], 4)    # 2 gens  * 2 slots
        self.assertEqual(summary['uc_load'], 2)   # 2 areas * 1 slot
        self.assertEqual(len(out['EDSlotLoad']), 4)
        # Row content: (area, slot, sd) preserved with file ordering.
        first = out['EDSlotLoad'][0]
        self.assertEqual((first['area'], first['slot'], first['sd']),
                         (1, 'EDT1', 0.5))

    def test_top_level_key_rename(self):
        """``EDTSlot`` / ``UCTSlot`` keys rename to ``EDSlot`` / ``UCSlot``."""
        out, summary = migrate_data(_legacy_case())
        self.assertIn('EDSlot', out)
        self.assertIn('UCSlot', out)
        self.assertNotIn('EDTSlot', out)
        self.assertNotIn('UCTSlot', out)
        self.assertTrue(summary['renamed'])

    def test_idempotent_on_migrated_input(self):
        """Re-running on already-migrated data is a no-op."""
        out1, _ = migrate_data(_legacy_case())
        out2, summary2 = migrate_data(deepcopy(out1))
        self.assertTrue(summary2['no_op'])
        self.assertEqual(out1, out2)

    def test_partial_state_raises(self):
        """Mixed legacy cells + pre-existing per-axis table raises
        rather than silently dropping rows."""
        data = _legacy_case()
        # Manually pre-populate EDSlotLoad with one row to simulate
        # a partly-migrated case file.
        data['EDSlotLoad'] = [{'area': 1, 'slot': 'EDT1', 'sd': 9.99}]
        # Rename the legacy EDTSlot key to its new form too — only the
        # cells, not the table, were "migrated".
        with pytest.raises(ValueError, match="partly-migrated"):
            migrate_data(data)

    def test_csv_length_mismatch_raises(self):
        """``sd`` cell length not matching ``len(Area)`` raises with
        the slot idx + observed/expected lengths."""
        data = _legacy_case()
        # Length 3, areas=2 — and still comma-bearing so it's
        # detected as a legacy cell (a non-comma string would skip).
        data['EDTSlot'][0]['sd'] = '0.5,0.7,0.9'
        with pytest.raises(ValueError, match="EDT1.*length 3.*expected 2"):
            migrate_data(data)

    def test_no_area_drops_legacy_sd(self):
        """Cases without an Area block (npcc/wecc-style) drop their
        legacy sd cells with a warning instead of erroring."""
        data = {
            'Slack': [{'idx': 'S1', 'u': 1.0, 'name': 'S1', 'bus': 0}],
            'EDTSlot': [
                {'idx': 'EDT1', 'u': 1.0, 'name': 'EDT1',
                 'sd': '0.5,0.7', 'ug': '1'},
            ],
        }
        out, summary = migrate_data(data)
        self.assertEqual(summary['ed_load'], 0)
        self.assertNotIn('EDSlotLoad', out)
        # EDSlot rows themselves are still slimmed.
        self.assertNotIn('sd', out['EDSlot'][0])

    def test_rename_only_pass(self):
        """A case already in v1.3.0 shape with only the legacy key
        names (no CSV cells) gets renamed, no per-axis emission."""
        data = {
            'Area': [{'idx': 1, 'u': 1.0, 'name': '1'}],
            'EDTSlot': [{'idx': 'EDT1', 'u': 1.0, 'name': 'EDT1'}],
            'EDSlotLoad': [{'area': 1, 'slot': 'EDT1', 'sd': 0.5}],
        }
        out, summary = migrate_data(data)
        self.assertTrue(summary['renamed'])
        self.assertFalse(summary['no_op'])
        self.assertEqual(summary['ed_load'], 0)
        self.assertIn('EDSlot', out)
        self.assertEqual(len(out['EDSlotLoad']), 1)


class TestRenameLegacyKeys(unittest.TestCase):

    def test_keys_renamed_in_order(self):
        data = {'A': 1, 'EDTSlot': [], 'B': 2, 'UCTSlot': []}
        out, renamed = _rename_legacy_keys(data)
        self.assertTrue(renamed)
        self.assertEqual(list(out.keys()), ['A', 'EDSlot', 'B', 'UCSlot'])

    def test_no_rename_when_absent(self):
        data = {'A': 1, 'EDSlot': []}
        out, renamed = _rename_legacy_keys(data)
        self.assertFalse(renamed)
        self.assertEqual(out, data)
