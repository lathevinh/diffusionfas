import unittest

from diffusionfas.coverage_protocol import (
    counts, coverage_matrix, domain_feasibility, family_candidates, validate_records,
)
from diffusionfas.io import object_digest


def fixture():
    rows = []
    for role in ("fitting", "validation", "calibration", "evaluation"):
        for label, family in (("bona_fide", None), ("attack", "print"), ("attack", "mask")):
            key = f"SiW-Mv2/{role}-{family}"
            rows.append({"key": key, "dataset": "SiW-Mv2", "role": role,
                         "official_split": "test" if role == "evaluation" else "train",
                         "label": label, "family": family, "type": family, "usable": True,
                         "subject_id": None, "component": key})
    return rows


class CoverageProtocolTests(unittest.TestCase):
    def test_matrix_retains_failures_unknown_subjects_and_unknown_taxonomy(self):
        rows = fixture()
        rows[1] = {**rows[1], "usable": False}
        rows[2] = {**rows[2], "family": None}
        validate_records(rows)
        matrix = coverage_matrix(rows)
        self.assertEqual(sum(r["originals"] for r in matrix), len(rows))
        self.assertEqual(sum(r["usable"] for r in matrix), len(rows)-1)
        self.assertEqual(sum(r["unknown_subject_originals"] for r in matrix), len(rows))
        self.assertEqual(sum(r["known_subjects"] for r in matrix), 0)
        self.assertTrue(any(r["family"] == "unknown_taxonomy" for r in matrix))
        self.assertEqual(counts([])["originals"], 0)

    def test_rejects_duplicate_identity_cross_role_component_and_test_reassignment(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            validate_records(rows+[rows[0]])
        for bad in ({**rows[3], "component": rows[0]["component"]},
                    {**rows[3], "official_split": "test"},
                    {**rows[3], "usable": 1},
                    {**rows[3], "family": "invented-bona-fide-family"}):
            with self.assertRaises(ValueError):
                validate_records(rows[:3]+[bad]+rows[4:])

    def test_family_absent_in_all_baseline_non_test_roles_full_target_unchanged(self):
        rows = fixture()
        summaries, inventories = family_candidates(rows)
        self.assertEqual(len(inventories), 2)
        indexed = {r["key"]: r for r in rows}
        target = sorted(r["key"] for r in rows if r["role"] == "evaluation")
        for summary, inventory in zip(summaries, inventories, strict=True):
            family = inventory["heldout_family"]
            for role in ("fitting", "validation", "calibration"):
                self.assertTrue(all(indexed[k]["family"] != family
                                    for k in inventory["baseline_roles"][role]["original"]))
            self.assertEqual(inventory["baseline_roles"]["evaluation"]["original"], target)
            self.assertEqual(summary["inventory_sha256"], object_digest(inventory))
            self.assertEqual(summary["heldout_fitting_attack_eligible"], 1)
            self.assertEqual(summary["target5pct_integer_FA_budget"], 0)
            self.assertFalse(summary["execution_ready"])
            self.assertFalse(summary["training_authorized"])
            self.assertFalse(summary["unseen_domain_claim_permitted"])
            self.assertFalse(summary["subject_disjoint_claim_permitted"])

    def test_component_closure_removes_other_family_and_bona_fide_explicitly_blocks(self):
        rows = fixture()
        rows[0]["component"] = rows[2]["component"]
        rows[1]["component"] = rows[2]["component"]
        validate_records(rows)
        summaries, inventories = family_candidates(rows)
        i = next(i for i in inventories if i["heldout_family"] == "mask")
        s = next(s for s in summaries if s["heldout_family"] == "mask")
        self.assertEqual(i["baseline_roles"]["fitting"]["original"], [])
        self.assertEqual(len(i["collateral_removed"]), 2)
        self.assertIn("heldout_component_contains_other_family_or_bona_fide", s["blockers"])
        self.assertIn("missing_usable_class_or_heldout_fitting_support", s["blockers"])

    def test_no_unknown_family_or_failure_is_used_as_heldout_comparator_support(self):
        rows = fixture()
        rows[1]["family"] = None
        rows[2]["usable"] = False
        summaries, inventories = family_candidates(rows)
        i = next(i for i in inventories if i["heldout_family"] == "mask")
        s = next(s for s in summaries if s["heldout_family"] == "mask")
        self.assertNotIn(rows[1]["key"], i["baseline_roles"]["fitting"]["original"])
        self.assertEqual(i["comparator_heldout_fitting_attack_eligible"], [])
        self.assertIn("unknown_training_taxonomy_cannot_certify_family_absence", s["blockers"])

    def test_domain_coverage_does_not_count_target_train_or_failed_source_as_coverage(self):
        rows = fixture()
        source = {**rows[1], "key": "other/a", "dataset": "other", "component": "other/a"}
        matrix = domain_feasibility(rows+[source])
        target = next(x for x in matrix if x["target"] == "SiW-Mv2")
        self.assertEqual(target["usable_fitting_families"], ["print"])
        status = {x["family"]: x["coverage"] for x in target["families"]}
        self.assertEqual(status, {"mask": "source_family_unseen", "print": "source_family_covered"})
        unknown = {**source, "family": None}
        target = next(x for x in domain_feasibility(rows+[unknown]) if x["target"] == "SiW-Mv2")
        self.assertTrue(all(x["coverage"] == "source_coverage_unknown" for x in target["families"]))
        failed = {**source, "usable": False}
        target = next(x for x in domain_feasibility(rows+[failed]) if x["target"] == "SiW-Mv2")
        self.assertEqual(target["usable_fitting_families"], [])


if __name__ == "__main__":
    unittest.main()
