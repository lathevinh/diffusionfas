"""Run the owner-selected sixteen-frame real-pixel crop preview on CPU."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True
for name, value in {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                    "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}.items():
    os.environ[name] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-root", type=Path, required=True)
    parser.add_argument("--diagnosis-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise RuntimeError("existing fas conda environment required")
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    from diffusionfas.io import digest, require_private_output, write_json
    from diffusionfas.spatial_crop_preview import BUDGET, run
    if args.worker:
        result = run(args.completed_root, args.diagnosis_root, args.private_root)
        print(json.dumps({"completed": result["completed"], "frames": result["frames"]}))
        return
    require_private_output(args.private_root/"summary.json")
    if args.private_root.exists():
        raise FileExistsError("new immutable crop preview root required")
    from diffusionfas.supervision import run_bounded
    result = run_bounded(
        [sys.executable, str(Path(__file__).resolve()), "--completed-root", str(args.completed_root),
         "--diagnosis-root", str(args.diagnosis_root), "--private-root", str(args.private_root/"data"), "--worker"],
        cwd=ROOT, owned_root=args.private_root, log=args.private_root/"preview.log",
        max_seconds=BUDGET["max_seconds"], max_rss_bytes=BUDGET["max_rss_bytes"],
        max_disk_bytes=BUDGET["max_output_bytes"], rss_root_pid=os.getpid(),
    )
    write_json(args.private_root/"supervision.json", result)
    if not result["completed"]:
        raise RuntimeError("crop preview failed; retained evidence, no automatic retry")
    print(json.dumps({"completed": True, "summary_sha256": digest(args.private_root/"data/summary.json"),
                      "supervision": result}))


if __name__ == "__main__":
    main()
