import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from diffusionfas.io import digest
from diffusionfas.spatial_crop_preparation import encode_square, footprint_check, inventory
from diffusionfas.spatial_crop_preview import load_crop_pair, render_pair
from diffusionfas.spatial_diagnosis import crop_candidates
import test_spatial_diagnosis as fixtures


class SpatialCropPreparationTests(unittest.TestCase):
    def test_native_asset_schema_variants_keep_canonical_pixel_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"transactions").mkdir()
            record = root/"transactions/0000.json"
            record.write_text("{}")
            transaction = fixtures.SpatialDiagnosisTests().transaction()
            native = np.random.default_rng(6).integers(0, 256, (480, 640, 3), dtype=np.uint8)
            transaction["selected_frame"]["rgb_sha256"] = hashlib.sha256(native.tobytes()).hexdigest()
            old, square, _ = render_pair(native, transaction)
            native_path, old_path = root/"native.png", root/"legacy.png"
            Image.fromarray(native).save(native_path)
            Image.fromarray(old).save(old_path)
            transaction["selected_RGB_asset"] = {"path": "native.png", "sha256": digest(native_path)}
            row = {"record_path": str(record), "record_sha256": digest(record),
                   "crops": {"224": {"path": str(old_path), "sha256": digest(old_path)}}}
            first, new, _, bindings = load_crop_pair(row, transaction)
            self.assertTrue(np.array_equal(first, native))
            self.assertTrue(np.array_equal(new, square))
            self.assertEqual(bindings[str(native_path)], digest(native_path))
            asset = transaction["selected_RGB_asset"]
            asset.update({"RGB_sha256": transaction["selected_frame"]["rgb_sha256"],
                          "shape": [480, 640, 3]})
            load_crop_pair(row, transaction)
            asset["RGB_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "optional native RGB"):
                load_crop_pair(row, transaction)
            del asset["RGB_sha256"]
            asset["shape"] = [10, 10, 3]
            with self.assertRaisesRegex(ValueError, "optional native shape"):
                load_crop_pair(row, transaction)
            del asset["shape"]
            transaction["selected_frame"]["rgb_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "exact historical"):
                load_crop_pair(row, transaction)

    def test_square_png_is_lossless_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"crop.png"
            pixels = np.random.default_rng(4).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            asset = encode_square(path, pixels)
            self.assertEqual(asset["sha256"], digest(path))
            self.assertEqual(asset["shape"], [224, 224, 3])
            with self.assertRaises(FileExistsError):
                encode_square(path, pixels)
            with self.assertRaises(ValueError):
                encode_square(Path(directory)/"invalid.png", pixels[:100])

    def test_stricter_quantization_support_is_checked_without_padding(self):
        t = fixtures.SpatialDiagnosisTests().transaction()
        candidate = crop_candidates(t)["candidates"]["expanded_legacy_square"]
        footprint_check(t, candidate)
        t["selected_frame"]["width"] = 420
        with self.assertRaisesRegex(ValueError, "footprint"):
            footprint_check(t, candidate)

    def cohort(self):
        indexed, entries, contexts = {}, [], []
        for domain in ("a", "b", "c", "d"):
            for role in ("fitting", "validation", "calibration", "evaluation"):
                for label in ("attack", "bona_fide"):
                    key = f"{domain}/{role}/{label}"
                    indexed[key] = {"dataset": domain, "binary_label": label, "role": role}
                    entries.append({"key": key, "status": "usable"})
            contexts.append({"target": domain, **{role: [
                key for key, row in indexed.items() if row["role"] == role and
                (row["dataset"] == domain if role == "evaluation" else row["dataset"] != domain)]
                for role in ("fitting", "validation", "calibration", "evaluation")}})
        for context in contexts:
            for role in ("fitting", "validation", "calibration", "evaluation"):
                context[role] = [key for key, row in indexed.items() if row["role"] == role and
                    (row["dataset"] == context["target"] if role == "evaluation" else
                     row["dataset"] != context["target"])]
        return contexts, indexed, entries

    def test_exact_matched_jobs_and_shared_original_counts(self):
        contexts, indexed, entries = self.cohort()
        isolation = Mock()
        jobs, roles = inventory(contexts, indexed, entries, {"study": Mock(isolation=isolation)})
        self.assertEqual(len(jobs), 24)
        self.assertEqual(isolation.call_count, 4)
        self.assertEqual(len(roles), 4)
        for target in ("a", "b", "c", "d"):
            pairs = [job for job in jobs if job["target"] == target]
            self.assertEqual(len(pairs), 6)
            self.assertTrue(all(job["usable"] == pairs[0]["usable"] for job in pairs))
        with self.assertRaisesRegex(ValueError, "identity"):
            inventory(contexts, indexed, entries+[entries[0]], {"study": Mock()})
        entries[0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "stratum"):
            inventory(contexts, indexed, entries, {"study": Mock(isolation=isolation)})

    def test_target_never_enters_source_fitting(self):
        contexts, indexed, entries = self.cohort()
        contexts[0]["fitting"].append("a/fitting/attack")
        with self.assertRaisesRegex(ValueError, "exclude held-out"):
            inventory(contexts, indexed, entries, {"study": Mock()})


if __name__ == "__main__":
    unittest.main()
