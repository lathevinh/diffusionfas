from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diffusionfas.io import read_json, write_json
from diffusionfas.siw_attack_diagnosis import (
    BUDGET, ALPHAS, attack_slice, check_evaluation_groups, completed_inputs, coverage_status, identity, joined_scores,
    taxonomy,
)


def row(token, label, good=True):
    return {"dataset": "SiW-Mv2", "video_id": token, "binary_label": label,
            "detector_status": "success" if good else "failed", "subject_id": None,
            "official_split": "test"}


class SiWAttackDiagnosisTests(unittest.TestCase):
    def test_caps_and_coverage_do_not_invent_exact_pai_or_missing_taxonomy(self):
        self.assertEqual(BUDGET, {"max_seconds": 1800, "max_rss_bytes": 2*1024**3,
                                 "max_output_bytes": 64*1024**2})
        self.assertEqual(ALPHAS, (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10)))
        self.assertEqual(coverage_status("print", {"print", "replay"}, 0), "source_family_covered")
        self.assertEqual(coverage_status("mask", {"print", "replay"}, 0), "source_family_unseen")
        self.assertEqual(coverage_status(None, {"print"}, 0), "unknown_taxonomy")
        self.assertEqual(coverage_status("mask", {"print"}, 1), "source_coverage_unknown")

    def test_ordered_score_join_rejects_reorder_length_nonfinite_failure_and_prediction_drift(self):
        rows = [row("a", "attack"), row("b", "attack", False), row("r", "bona_fide")]
        indexed = {identity(r): r for r in rows}
        context = {"target": "SiW-Mv2", "evaluation": list(indexed)}
        expected = [identity(rows[0]), identity(rows[2])]
        saved = {"target": "SiW-Mv2", "keys": {"evaluation": expected},
                 "models": {"legacy_content": {"probabilities": {"evaluation": [.2, .1]}}},
                 "prediction": {"legacy_content": [.2, .1]}}
        self.assertEqual(joined_scores(context, saved, indexed, "legacy_content", "evaluation"),
                         dict(zip(expected, [.2, .1])))
        for keys, values, prediction in (
            (expected[::-1], [.2, .1], [.2, .1]),
            (list(indexed), [.2, .3, .1], [.2, .3, .1]),
            (expected, [.2], [.2]),
            (expected, [np.nan, .1], [np.nan, .1]),
            (expected, [1.1, .1], [1.1, .1]),
            (expected, [.2, .1], [.3, .1]),
        ):
            with self.subTest(keys=keys, values=values):
                bad = {**saved, "keys": {"evaluation": keys},
                       "models": {"legacy_content": {"probabilities": {"evaluation": values}}},
                       "prediction": {"legacy_content": prediction}}
                with self.assertRaises(ValueError):
                    joined_scores(context, bad, indexed, "legacy_content", "evaluation")

    def test_bootstrap_group_commitment_covers_targets_not_all_fitting_rows(self):
        components = {"source": "source-group", "target1": "group1", "target2": "group2"}
        plan = {"contexts": [{"evaluation": ["target1"]}, {"evaluation": ["target2"]}],
                "groups": {"target1": "group1", "target2": "group2"}}
        check_evaluation_groups(plan, components)
        for bad in ({**plan["groups"], "source": "source-group"},
                    {"target1": "wrong", "target2": "group2"},
                    {"target1": "group1"}):
            with self.assertRaises(ValueError):
                check_evaluation_groups({**plan, "groups": bad}, components)

    def test_original_common_denominators_ties_and_small_group_oracle(self):
        attacks = [row("a", "attack"), row("b", "attack"), row("x", "attack", False)]
        reals = [row("r1", "bona_fide"), row("r2", "bona_fide"), row("rx", "bona_fide", False)]
        scores = {identity(r): s for r, s in zip(attacks[:2]+reals[:2], [.1, .2, .1, .05])}
        point = attack_slice(attacks, reals, scores, .1, Fraction(1, 20))
        self.assertEqual((point["original_attack"], point["usable_attack"], point["excluded_attack"]), (3, 2, 1))
        self.assertEqual(point["accepted_attack"], 0)
        self.assertEqual(point["FA_original"], 0)
        self.assertEqual(point["original_oracle"]["max_BFA"], 1/3)
        self.assertEqual(point["original_oracle"]["accepted_attack"], 0)
        self.assertTrue(point["zero_error_budget"])
        self.assertTrue(point["fewer_than20_original_attacks"])
        point = attack_slice(attacks, reals, scores, .2, Fraction(1, 20))
        self.assertEqual(point["accepted_attack"], 1)
        self.assertEqual(point["FA_original"], 1/3)
        self.assertEqual(point["FA_common"], 1/2)
        with self.assertRaises(ValueError):
            attack_slice(attacks, reals, {**scores, identity(attacks[-1]): .0}, .1, Fraction(1, 20))
        with self.assertRaises(ValueError):
            attack_slice(attacks+[attacks[0]], reals, scores, .1, Fraction(1, 20))

    def test_all_failed_attacks_explicitly_have_no_discrimination_evidence(self):
        a, r = row("a", "attack", False), row("r", "bona_fide")
        point = attack_slice([a], [r], {identity(r): .1}, .2, Fraction(1, 20))
        self.assertIsNone(point["FA_common"])
        self.assertFalse(point["common_FA_defined"])
        self.assertFalse(point["discrimination_evidence_available"])
        self.assertEqual(point["original_oracle"]["max_BFA"], 1)

    def test_disjoint_attack_slices_recount_global_acceptance_without_reassigning_reals(self):
        attacks = [row(str(i), "attack", i != 3) for i in range(4)]
        reals = [row("r", "bona_fide"), row("rx", "bona_fide", False)]
        scores = {identity(r): s for r, s in zip(attacks[:3]+reals[:1], [.1, .5, .05, .1])}
        groups = [attack_slice(pool, reals, scores, .5, Fraction(1, 20))
                  for pool in (attacks[:2], attacks[2:])]
        self.assertEqual(sum(g["original_attack"] for g in groups), 4)
        self.assertEqual(sum(g["usable_attack"] for g in groups), 3)
        self.assertEqual(sum(g["accepted_attack"] for g in groups), 2)
        self.assertEqual({g["oracle_original_bona_fide"] for g in groups}, {2})
        for group in groups:
            self.assertTrue(group["original_oracle"]["uses_target_labels"])
            self.assertFalse(group["original_oracle"]["deployable"])

    def test_pinned_real_taxonomy_join_and_no_fabricated_subject(self):
        if (not Path("/mnt/e/FAS/src/fas/siwmv2.py").is_file()
                or not Path("/mnt/e/FAS-private/artifacts/phase1/siwmv2_metadata_inventory_v1.json").is_file()
                or not Path("/mnt/e/FAS-private/artifacts/phase1/msu_metadata_inventory_v1.json").is_file()):
            self.skipTest("bound native metadata integration requires existing private inventories")
        from diffusionfas.regional_study import source_services
        source_services(Path("/mnt/e/FAS"))
        base = Path("/mnt/e/FAS-private/artifacts/phase1")
        siw = read_json(base/"siwmv2_metadata_inventory_v1.json")["videos"]
        selected = [next(r for r in siw if r["reference_attack_type"] == kind and r["official_split"] == "test")
                    for kind in ("Paper", "Replay", "Mask_Silicone", "Makeup_Co", "Partial_Eye")]
        original = [{k: r[k] for k in ("dataset", "video_id", "binary_label", "subject_id", "official_split")}
                    for r in selected]
        mapping, bindings = taxonomy(original, Path("/mnt/e/FAS"))
        self.assertEqual({v["family"] for v in mapping.values()}, {"print", "replay", "mask", "makeup", "partial"})
        self.assertIn("siwmv2_protocol_i_intersection_v1.json", bindings)
        with self.assertRaises(ValueError):
            taxonomy([{**original[0], "subject_id": "invented"}], Path("/mnt/e/FAS"))
        with self.assertRaises(ValueError):
            taxonomy([{**original[0], "official_split": "train"}], Path("/mnt/e/FAS"))

    def test_completed_inputs_rejects_changed_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"plan.json", {"study": "old20"})
            public = {"completed": True, "receipt_sha256": {"plan.json": "0"*64}}
            with patch("diffusionfas.siw_attack_diagnosis.read_json", side_effect=lambda path, sha=None:
                       public if path.name == "spatial-crop-completed-v1.json" else read_json(path, sha)):
                with self.assertRaises(ValueError):
                    completed_inputs(root)


if __name__ == "__main__":
    unittest.main()
