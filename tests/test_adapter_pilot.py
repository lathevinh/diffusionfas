from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from diffusionfas.adapter_pilot import ARMS, build_plan, checked_inputs, public_validation, summarize
from diffusionfas.cli import manifest_payload
from diffusionfas.io import digest, read_json, write_json
from test_training import ROOT, synthetic_population


class Fixture:
    def __init__(self, root: Path, completed: bool = True) -> None:
        self.root, self.output = root, root / "pilot"
        base = synthetic_population(root)
        videos = []
        for domain in sorted({v.dataset for v in base}):
            source = next(v for v in base if v.dataset == domain and v.partition == "fitting")
            videos.extend(replace(source, key=f"{domain}/fitting/{i}", group=f"fitting/{i}",
                                  subject=f"fitting/{i}") for i in range(100))
            videos.extend(v for v in base if v.dataset == domain and v.partition == "validation")
        videos.append(replace(base[0], key="OULU-NPU/target", dataset="OULU-NPU",
                              partition="evaluation", group="target", subject="target",
                              status="pending", cache_path=None, cache_sha256=None))
        videos.append(replace(base[0], key="CASIA-FASD/attack", label="attack", group="attack",
                              subject="attack", status="pending", cache_path=None, cache_sha256=None))
        self.preparation = root / "preparation/plan.json"
        write_json(self.preparation, {"synthetic": True})
        self.manifest = root / "population.json"
        write_json(self.manifest, manifest_payload(
            videos, {"preparation_plan_sha256": digest(self.preparation)}))
        write_json(self.preparation.parent / "completion.json", {
            "completed": completed, "plan_sha256": digest(self.preparation),
            "prepared_manifest_sha256": digest(self.manifest),
        })
        write_json(self.preparation.parent / "supervision.json", {
            "completed": completed, "plan_sha256": digest(self.preparation),
        })
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        for name in ("config.json", "scheduler_config.json", "model.safetensors"):
            (self.bundle / name).write_bytes(b"synthetic-not-loaded")
        write_json(self.bundle / "manifest.json", {
            "rights_reviewed": True, "upstream_revision": "synthetic",
            "files": {p.name: digest(p) for p in self.bundle.iterdir()},
        })

    def plan(self) -> Path:
        return build_plan(self.manifest, digest(self.manifest), self.preparation, self.bundle,
                          digest(self.bundle / "manifest.json"), ROOT / "configs/offline-v1.json",
                          self.output)


class AdapterPilotTests(unittest.TestCase):
    def test_plan_locks_source_only_balanced_matched_two_epoch_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            path = fixture.plan()
            plan, sources = checked_inputs(path, digest(path))
            self.assertEqual(plan["fitting_originals"], 300)
            self.assertEqual(plan["validation_originals"], 6)
            self.assertEqual(plan["steps_per_arm"], 300)
            self.assertEqual(len(sources), 306)
            self.assertTrue(all(v.dataset != "OULU-NPU" and v.label == "bona_fide" for v in sources))
            self.assertEqual({k for k, v in plan["config"]["execution"].items() if v},
                             {"source_training", "source_scoring"})
            self.assertEqual(plan["config"]["training"]["max_epochs"], 2)
            self.assertEqual(plan["budget"]["max_gpu_bytes"], 8 * 1024**3)
            self.assertEqual(plan["budget"]["max_output_bytes"], 2 * 1024**3)
            for epoch in plan["schedule"]:
                self.assertEqual(set(epoch["draws_by_domain"].values()), {100})

    def test_incomplete_preparation_is_rejected_before_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), completed=False)
            with self.assertRaisesRegex(ValueError, "completed source-preparation"):
                fixture.plan()
            self.assertFalse(fixture.output.exists())

    def test_changed_population_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            path = fixture.plan()
            with fixture.manifest.open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "population changed"):
                checked_inputs(path, digest(path))

    def test_target_permission_cannot_be_enabled_in_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            path = fixture.plan()
            plan = read_json(path)
            plan["config"]["execution"]["target_evaluation"] = True
            altered = fixture.output / "altered.json"
            write_json(altered, plan)
            with self.assertRaises(PermissionError):
                checked_inputs(altered, digest(altered))

    def test_public_validation_excludes_private_scores_and_keys(self) -> None:
        metrics = {"macro_loss": 0.1, "by_domain": {"source": {"originals": 2, "mean_loss": 0.1}},
                   "records": [{"key": "private", "profile": [0.1]}]}
        self.assertEqual(set(public_validation(metrics)), {"macro_loss", "by_domain"})

    def test_partial_pilot_reports_started_training_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            path = fixture.plan()
            write_json(path.parent / f"started-{ARMS[0]}.json",
                       {"plan_sha256": digest(path), "variant": ARMS[0]})
            write_json(path.parent / f"result-{ARMS[0]}.json", {"variant": ARMS[0], "steps": 300})
            report = summarize(path, {"completed": False})
            self.assertFalse(report["completed"])
            self.assertEqual(report["scientific_source_training_runs"], 1)
            self.assertEqual(len(report["completed_arms"]), 1)
            self.assertIsNone(report["training"])
