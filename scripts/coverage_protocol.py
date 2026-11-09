"""Prepare bounded metadata-only family-coverage inventories; never train."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-root", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise RuntimeError("existing fas conda environment required")
    from diffusionfas.coverage_protocol import BUDGET, run
    from diffusionfas.io import digest, require_private_output, write_json
    if args.worker:
        import torch
        torch.set_num_threads(2)
        print(json.dumps({"completed": run(args.diagnosis_root, args.completed_root, args.private_root)["completed"]}))
        return
    require_private_output(args.private_root/"summary.json")
    if args.private_root.exists():
        raise FileExistsError("fresh immutable coverage preparation root required")
    from diffusionfas.supervision import run_bounded
    result = run_bounded(
        [sys.executable, str(Path(__file__).resolve()), "--diagnosis-root", str(args.diagnosis_root),
         "--completed-root", str(args.completed_root), "--private-root", str(args.private_root/"data"), "--worker"],
        cwd=ROOT, owned_root=args.private_root, log=args.private_root/"preparation.log",
        max_seconds=BUDGET["max_seconds"], max_rss_bytes=BUDGET["max_rss_bytes"],
        max_disk_bytes=BUDGET["max_output_bytes"], rss_root_pid=os.getpid(),
    )
    write_json(args.private_root/"supervision.json", result)
    if not result["completed"]:
        raise RuntimeError("coverage preparation failed; evidence retained, no automatic retry")
    print(json.dumps({"completed": True, "summary_sha256": digest(args.private_root/"data/summary.json"),
                      "supervision": result}))


if __name__ == "__main__":
    main()
