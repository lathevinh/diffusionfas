"""Run offline synthetic tests and emit an aggregate validation receipt."""

from __future__ import annotations

import argparse
import compileall
import contextlib
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from diffusionfas.io import digest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("run validation through conda run -n fas")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.monotonic()
    for folder in ("src", "tests", "scripts"):
        if not compileall.compile_dir(ROOT / folder, quiet=1):
            raise SyntaxError(f"compilation failed: {folder}")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with contextlib.redirect_stdout(sys.stderr):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    files = sorted(path for folder in ("src", "tests", "scripts") for path in (ROOT / folder).rglob("*.py"))
    write_json(arguments.out, {
        "schema_version": 1, "environment": "conda:fas",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "wall_seconds": time.monotonic() - started,
        "scope": "offline_synthetic_only_not_PAD_performance_or_GPU_throughput",
        "PAD_video_decodes": 0, "PAD_training_runs": 0, "PAD_target_evaluations": 0,
        "pretrained_checkpoint_downloads": 0,
        "synthetic_video_decode_and_optimizer_steps_executed": True,
        "implementation_sha256": {str(path.relative_to(ROOT)): digest(path) for path in files},
        "config_sha256": digest(ROOT / "configs/offline-v1.json"),
    })


if __name__ == "__main__":
    main()
