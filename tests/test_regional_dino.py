from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from diffusionfas.io import digest, write_json
from diffusionfas.regional_dino import (
    ARMS, DIMENSIONS, HEAD_SETTINGS, REGIONS, crop_landmarks, dro_weights, fit_group_dro,
    group_matrix, pool_tokens, region_weights,
)
from diffusionfas.regional_study import (
    BUDGET, POLICY, SCOPE, SOURCE_REVISION, checked_plan, implementation, require_approval,
    paired_intervals, regional_population, runtime, source_services, summarize,
)
from diffusionfas.regional_verify import independent_intervals, reconstruct_scores, verify_completed


class RegionalTests(unittest.TestCase):
    def points(self):
        return np.array([[180., 180.], [340., 180.], [260., 260.], [200., 340.], [320., 340.]])

    def transaction(self):
        return {"landmarks": self.points().tolist(), "crop_shape": [518, 518, 3],
                "geometry": {"alignment_matrix": [[1., 0., 0.], [0., 1., 0.]],
                             "crop_xyxy_half_open": [0, 0, 518, 518], "context_scale": 1.30,
                             "interpolation": "bicubic", "border": "constant_zero"}}

    def test_exact_pixel_center_landmark_transfer_rectangular_crop(self):
        transaction = self.transaction()
        np.testing.assert_array_equal(crop_landmarks(transaction), self.points())
        transaction["crop_shape"] = [800, 400, 3]
        transaction["geometry"]["crop_xyxy_half_open"] = [20, 10, 420, 810]
        transaction["geometry"]["alignment_matrix"] = [[0., -1., 600.], [1., 0., 10.]]
        rotated = np.column_stack((600-self.points()[:, 1], self.points()[:, 0]+10))
        expected = (rotated-[20, 10]+.5)*518/[400, 800]-.5
        np.testing.assert_allclose(crop_landmarks(transaction), expected, rtol=0, atol=1e-12)
        transaction["crop_shape"] = [400, 800, 3]
        with self.assertRaises(ValueError):
            crop_landmarks(transaction)

    def test_anatomical_controls_match_count_and_dimensions_without_clipping(self):
        weights = region_weights(self.points())
        self.assertEqual(len(REGIONS), 4)
        self.assertEqual(set(weights), {"anatomical", "spatial_control"})
        for value in weights.values():
            self.assertEqual(value.shape, (4, 1369))
            self.assertEqual(value.dtype, np.float32)
            np.testing.assert_allclose(value.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(weights["anatomical"].astype(bool).sum(axis=1),
                                      weights["spatial_control"].astype(bool).sum(axis=1))
        self.assertFalse(np.array_equal(weights["anatomical"], weights["spatial_control"]))
        with self.assertRaisesRegex(ValueError, "outside"):
            region_weights(self.points()+[400, 0])
        broken = self.points()
        broken[1] = broken[0]
        with self.assertRaisesRegex(ValueError, "degenerate"):
            region_weights(broken)
        tiny = np.tile([7., 7.], (5, 1))
        tiny[1, 0] += .01
        with self.assertRaisesRegex(ValueError, "patch-center"):
            region_weights(tiny)

    def test_pooling_registers_cannot_enter_and_all_modes_share_global_prefix(self):
        cls = np.arange(768, dtype=np.float32)[None]
        patches = np.arange(1369, dtype=np.float32)[None, :, None]*np.ones((1, 1, 768), np.float32)
        weights = region_weights(self.points())
        pooled = pool_tokens(cls, patches, weights)
        for name, value in pooled.items():
            self.assertEqual(value.shape, (1, DIMENSIONS[name]))
            self.assertEqual(value.dtype, np.float32)
            np.testing.assert_array_equal(value[:, :1536], pooled["global"])
        expected = weights["anatomical"] @ np.arange(1369, dtype=np.float32)
        np.testing.assert_allclose(pooled["anatomical"][0, 1536:].reshape(4, 768)[:, 0], expected, atol=.001)
        with self.assertRaises(ValueError):
            pool_tokens(cls, np.zeros((1, 1373, 768), np.float32), weights)
        invalid = copy.deepcopy(weights)
        invalid["anatomical"][0] = 0
        with self.assertRaises(ValueError):
            pool_tokens(cls, patches, invalid)

    def test_group_reduction_is_equal_source_equal_class_not_sample_mean(self):
        names, weights = group_matrix(np.array([0, 0, 1, 0, 1]), ["a", "a", "a", "b", "b"])
        self.assertEqual(names, ["a", "b"])
        np.testing.assert_array_equal(weights, [[.25, .25, .5, 0, 0], [0, 0, 0, .5, .5]])
        np.testing.assert_array_equal(weights @ np.array([0., 4., 4., 2., 4.]), [3, 3])
        with self.assertRaisesRegex(ValueError, "both classes"):
            group_matrix(np.array([0, 1]), ["a", "b"])
        with self.assertRaises(ValueError):
            group_matrix(np.array([0, np.nan]), ["a", "a"])

    def test_dro_focuses_hard_source_stably_and_zero_step_keeps_equal_weights(self):
        logq, q = dro_weights(np.full(2, -np.log(2)), np.array([1., 3.]), .1)
        np.testing.assert_allclose(q[1]/q[0], np.exp(.2))
        self.assertAlmostEqual(q.sum(), 1)
        _, uniform = dro_weights(np.full(2, -np.log(2)), np.array([1., 9999.]), 0)
        np.testing.assert_allclose(uniform, [.5, .5])
        for _ in range(100):
            logq, q = dro_weights(logq, np.array([1., 1000.]), .1)
        self.assertTrue(np.isfinite(logq).all())
        self.assertAlmostEqual(q.sum(), 1)
        for losses in (np.array([-1., 1.]), np.array([np.nan, 1.])):
            with self.assertRaises(ValueError):
                dro_weights(np.zeros(2), losses, .1)

    @unittest.skipUnless(Path("/mnt/e/FAS/src/fas/source_fitting.py").is_file(), "optional original FAS integration")
    def test_zero_step_matches_original_erm_optimizer_and_selection(self):
        sys.dont_write_bytecode = True
        fitting = source_services(Path("/mnt/e/FAS"))["fas.source_fitting"]
        rng = np.random.default_rng(42)
        x, vx = rng.normal(size=(12, 4)), rng.normal(size=(8, 4))
        y, vy = np.array([0, 1]*6), np.array([0, 1]*4)
        domains, validation_domains = ["a"]*6+["b"]*6, ["a"]*4+["b"]*4
        settings = {**HEAD_SETTINGS, "dro_step_size": 0.}
        arguments = (x, y, domains, vx, vy, validation_domains)
        erm = fitting.fit_head(*arguments, seed=7, settings=settings, guard=lambda: None)
        dro = fit_group_dro(*arguments, seed=7, settings=settings, fitting=fitting, guard=lambda: None)
        np.testing.assert_allclose(dro["coefficients"], erm["coefficients"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(dro["validation_history"], erm["validation_history"], rtol=0, atol=1e-14)
        self.assertEqual(dro["selected_epoch"], erm["selected_epoch"])
        self.assertEqual(len(dro["adversary_history"]), 200)
        guarded = []
        fitted = fit_group_dro(*arguments, seed=7, fitting=fitting, guard=lambda: guarded.append(1))
        self.assertEqual(len(guarded), 200)
        self.assertTrue(np.isfinite(fitted["coefficients"]).all())
        with self.assertRaises(ValueError):
            fit_group_dro(x, y, domains, vx, vy, ["c"]*4+["b"]*4, seed=7,
                          fitting=fitting, guard=lambda: None)

    def plan(self):
        from diffusionfas.dino_comparison import DINO_REVISION, SEEDS
        return {"study": "regional_dino_v1", "source_revision": SOURCE_REVISION,
                "scientific_execution_authorized": False, "scope": SCOPE, "budget": BUDGET,
                "policy": POLICY, "head_settings": HEAD_SETTINGS, "arms": list(ARMS),
                "seeds": list(SEEDS), "dimensions": DIMENSIONS, "runtime": runtime(),
                "implementation_sha256": implementation(), "upstream_revision": DINO_REVISION}

    def test_exact_protocol_and_new_owner_approval_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            write_json(path, self.plan())
            checked_plan(path, digest(path))
            approval = Path(directory)/"approval.json"
            write_json(approval, {"approved": True, "plan_sha256": digest(path),
                                  "scope": SCOPE, "budget": BUDGET, "no_automatic_retry": True})
            require_approval(path, digest(path), approval)
            write_json(Path(directory)/"started.json", {})
            with self.assertRaises(FileExistsError):
                require_approval(path, digest(path), approval)
        for key, value in (("scope", {**SCOPE, "new_download": True}),
                           ("budget", {**BUDGET, "max_seconds": BUDGET["max_seconds"]+1}),
                           ("arms", [ARMS[0]]),
                           ("policy", {**POLICY, "target_model_selection": True})):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"plan.json"
                write_json(path, {**self.plan(), key: value})
                with self.assertRaises(PermissionError):
                    checked_plan(path, digest(path))

    def test_failed_execution_cannot_be_reported_as_scientific_success(self):
        self.assertFalse(summarize(Path("/unused"), {"completed": False})["completed"])

    def test_bad_region_is_shared_failure_not_dropped_or_a_feature_fallback(self):
        rows = [{"dataset": "a", "video_id": name, "detector_status": "success",
                 "model_input_eligible": True, "score_fit_eligible": True} for name in ("good", "bad")]
        def transaction(row):
            result = self.transaction()
            if row["video_id"] == "bad":
                result["landmarks"] = (self.points()+400).tolist()
            return result
        with patch("diffusionfas.regional_study.transaction_for", side_effect=transaction):
            result, failures = regional_population(rows)
        self.assertEqual(len(result), len(rows))
        self.assertEqual(set(failures), {"a/bad"})
        self.assertEqual(result[0]["detector_status"], "success")
        self.assertEqual(result[1]["detector_status"], "failed")
        self.assertFalse(result[1]["model_input_eligible"])
        self.assertEqual(result[1]["final_k1_action"], "non_accept")
        self.assertEqual(rows[1]["detector_status"], "success")
        with patch("diffusionfas.regional_study.transaction_for", side_effect=ValueError("drift")):
            with self.assertRaisesRegex(ValueError, "drift"):
                regional_population(rows)

    def test_independent_scoring_shape_and_orientation(self):
        features = np.array([[1., 0.], [-1., 0.]], np.float32)
        model = {"head": {"coefficients": [2., 0., 0.]},
                 "calibrator": {"slope": 1., "intercept": 0.}}
        np.testing.assert_allclose(reconstruct_scores(features, model), 1/(1+np.exp(-np.array([2., -2.]))))
        model["calibrator"]["slope"] = -1
        with self.assertRaises(ValueError):
            reconstruct_scores(features, model)

    @unittest.skipUnless(Path("/mnt/e/FAS/src/fas/source_fitting.py").is_file(), "optional original FAS integration")
    def test_independent_bootstrap_matches_producer_with_ties_and_failures(self):
        services = source_services(Path("/mnt/e/FAS"))
        originals = [{"dataset": "toy", "video_id": f"v{i}", "group_id": f"g{i//2}",
                      "subject_id": f"g{i//2}", "selected_frame": {"rgb_sha256": f"h{i}"},
                      "binary_label": "attack" if i % 2 else "bona_fide",
                      "detector_status": "failed" if i == 7 else "success"} for i in range(8)]
        predictions = {arm: np.array([.5, .5, .1, .9, .8, .2, .4])+.001*index
                       for index, arm in enumerate(ARMS)}
        policies = {arm: {"1/20": {"threshold": .6}} for arm in ARMS}
        with patch.dict(POLICY, {"bootstrap_repetitions": 80}):
            actual = paired_intervals(predictions, originals, policies, services, lambda: None)
            expected = independent_intervals(predictions, originals, policies, services)
        self.assertEqual(len(actual), 45)
        for left, right in zip(actual, expected, strict=True):
            for key in ("arm", "metric", "comparison", "valid_repetitions", "repetitions", "declared_groups"):
                self.assertEqual(left[key], right[key])
            np.testing.assert_allclose(left["percentile95"], right["percentile95"], rtol=0, atol=1e-12)

    @unittest.skipUnless(Path("/mnt/e/FAS/src/fas/source_fitting.py").is_file(), "optional original FAS integration")
    def test_complete_synthetic_72_head_output_is_independently_reconstructed(self):
        services = source_services(Path("/mnt/e/FAS"))
        fitting, study = services["fas.source_fitting"], services["study"]
        rows = []
        for domain in ("a", "b", "c", "d"):
            for role in ("fitting", "validation", "calibration", "evaluation"):
                for label in ("bona_fide", "attack"):
                    key = f"{domain}-{role}-{label}"
                    rows.append({"dataset": domain, "video_id": key, "group_id": f"{domain}-{role}",
                                 "subject_id": f"{domain}-{role}", "binary_label": label,
                                 "selected_frame": {"rgb_sha256": key}, "detector_status": "success",
                                 "role": role})
        ids = sorted(fitting.identity(row) for row in rows)
        indexed = {fitting.identity(row): row for row in rows}
        position = {key: i for i, key in enumerate(ids)}
        arrays = {name: np.zeros((len(ids), dimension), np.float32) for name, dimension in DIMENSIONS.items()}
        for array in arrays.values():
            array[:, 0] = [1 if indexed[key]["binary_label"] == "attack" else -1 for key in ids]
        contexts = [{"target": domain, "seed": seed, **{
            role: sorted(fitting.identity(row) for row in rows if row["role"] == role and
                         (row["dataset"] == domain if role == "evaluation" else row["dataset"] != domain))
            for role in ("fitting", "validation", "calibration", "evaluation")}}
            for domain in ("a", "b", "c", "d") for seed in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as directory, patch.dict(POLICY, {"bootstrap_repetitions": 20}):
            root = Path(directory)
            plan = root/"plan.json"
            write_json(plan, {"synthetic_fixture": True})
            sha = digest(plan)
            write_json(root/"started.json", {"plan_sha256": sha, "started_monotonic": time.monotonic(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
            feature_files = {}
            for name, array in arrays.items():
                with (root/f"{name}.npy").open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                feature_files[f"{name}.npy"] = digest(root/f"{name}.npy")
            write_json(root/"features.json", {"ids": ids, "files_sha256": feature_files,
                                             "exact_synthetic_token_replay": True})
            cases, intervals, files = [], [], {}
            for context in contexts:
                originals = [indexed[key] for key in context["evaluation"]]
                calibration_rows = [indexed[key] for key in context["calibration"]]
                y = np.array([row["binary_label"] == "attack" for row in originals], dtype=int)
                models, predictions, policies = {}, {}, {}
                for arm in ARMS:
                    representation, objective = arm.split("__")
                    head = {"coefficients": [1.]+[0.]*DIMENSIONS[representation],
                            "validation_history": [1.]*200, "validation_bce": 1.,
                            "selected_epoch": 1, "seed": context["seed"]}
                    if objective == "group_dro":
                        head["adversary_history"] = [[1/3]*3 for _ in range(200)]
                    model = {"head": head, "calibrator": {"slope": 1., "intercept": 0.}}
                    model["calibration_scores"] = reconstruct_scores(arrays[representation][
                        [position[key] for key in context["calibration"]]], model).tolist()
                    models[arm] = model
                    predictions[arm] = reconstruct_scores(arrays[representation][
                        [position[key] for key in context["evaluation"]]], model)
                    policies[arm] = {str(alpha): study.threshold(np.asarray(model["calibration_scores"]),
                        calibration_rows, alpha) for alpha in study.ALPHAS}
                    for alpha in study.ALPHAS:
                        point = study.metrics(predictions[arm], originals, policies[arm][str(alpha)]["threshold"])
                        point["FA_safe"] = point["accepted_attack"] <= alpha.numerator//alpha.denominator
                        point["joint_utility"] = point["FA_safe"] and point["BFA"] >= .5
                        cases.append({"target": context["target"], "seed": context["seed"], "arm": arm,
                            "alpha": str(alpha), "policy": point,
                            "oracle": services["oracle316"].oracle(predictions[arm], y, 1, 1, alpha=alpha)})
                filename = f"context-{context['target']}-{context['seed']}.json"
                write_json(root/filename, {"models": models, "predictions": {
                    arm: value.tolist() for arm, value in predictions.items()},
                    "policies": policies, "evaluation_ids": context["evaluation"]})
                files[filename] = digest(root/filename)
                intervals.extend({"target": context["target"], "seed": context["seed"], **record}
                                 for record in paired_intervals(predictions, originals, policies, services, lambda: None))
            write_json(root/"result.json", {"plan_sha256": sha, "cases": cases,
                "conditional_intervals": intervals, "context_files_sha256": files,
                "feature_receipt_sha256": digest(root/"features.json")})
            with patch("diffusionfas.regional_verify.checked_plan", return_value={
                "fas_root": "/mnt/e/FAS", "shared_geometry_failures": {}}), patch(
                    "diffusionfas.regional_verify.inputs", return_value=(services, rows, {"contexts": contexts})), patch(
                    "diffusionfas.regional_verify.regional_population", return_value=(rows, {})):
                verify_completed(plan, sha)
            import json
            verification = json.loads((root/"verification.json").read_bytes())
            self.assertTrue(verification["completed"])
            self.assertEqual(verification["operating_cases"], 216)
            self.assertEqual(verification["interval_records"], 540)


if __name__ == "__main__":
    unittest.main()
