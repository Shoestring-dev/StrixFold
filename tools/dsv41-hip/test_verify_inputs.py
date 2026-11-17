#!/usr/bin/env python3
"""The RoPE references verify_inputs.py checks a bundle's tables against: which file it picks, and the shipped ones.

    python3 -m unittest tools/dsv41-hip/test_verify_inputs.py
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_inputs  # noqa: E402

TABLES = verify_inputs.ROPE_TABLES


class RopeReference(unittest.TestCase):
    def test_a_given_reference_wins(self):
        given = Path("/elsewhere/my.rope-reference.json")
        self.assertEqual(verify_inputs.rope_reference_path(given, Path("/b/inputs"), 65536), given)

    def test_a_shipped_row_count_uses_the_repository_reference(self):
        for rows in (65536, 131072, 1048576):
            got = verify_inputs.rope_reference_path(None, Path("/b/inputs"), rows)
            self.assertEqual(got, HERE / "assets" / f"rope_reference_{rows}.json")

    def test_another_row_count_uses_the_one_prepare_inputs_wrote_beside_the_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "inputs-tp2"
            want = Path(f"{bundle.resolve()}.rope-reference.json")
            self.assertEqual(verify_inputs.rope_reference_path(None, bundle, 200000), want)
            self.assertEqual(verify_inputs.rope_reference_path(None, Path(f"{bundle}/"), 200000), want)

    def test_the_shipped_references_are_prefixes_of_the_creator_kit_tables(self):
        creator = json.loads((HERE / "assets" / "creator_kit_v0.6.1_rope.json").read_text())
        shipped = sorted(HERE.glob("assets/rope_reference_*.json"))
        self.assertGreaterEqual(len(shipped), 6)
        for path in shipped:
            ref = json.loads(path.read_text())
            self.assertEqual(path.name, f"rope_reference_{ref['rows']}.json")
            self.assertEqual(ref["half"], creator["half"])
            self.assertTrue(ref["derived_from"]["creator_hashes_matched"])
            self.assertLessEqual(ref["rows"], creator["rows"])
            self.assertEqual(len({ref[n] for n in TABLES}), 4)
        full = json.loads((HERE / "assets" / f"rope_reference_{creator['rows']}.json").read_text())
        self.assertEqual({n: full[n] for n in TABLES}, {n: creator[n] for n in TABLES})


if __name__ == "__main__":
    unittest.main()
