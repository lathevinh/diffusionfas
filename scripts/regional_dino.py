"""Prepare or execute the explicitly approved regional DINO factorial study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["XFORMERS_DISABLED"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("--fas-root", type=Path, required=True)
    prepare.add_argument("--private-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise RuntimeError("regional DINO requires the existing fas conda environment")
    from diffusionfas.io import digest, write_json
    from diffusionfas.regional_study import BUDGET, build_plan, require_approval, run_study, summarize
    if args.command == "plan":
        path = build_plan(args.fas_root, args.private_root)
        print({"plan_sha256": digest(path), "scientific_execution_authorized": False}, flush=True)
        return
    if args.command == "verify":
        from diffusionfas.regional_verify import verify_completed
        verify_completed(args.plan, args.plan_sha256)
        print({"verified": True}, flush=True)
        return
    if args.worker:
        if args.plan_sha256 is None:
            raise PermissionError("parent plan commitment required")
        run_study(args.plan, args.plan_sha256, args.approval)
        return
    from diffusionfas.adapter_pilot import device_memory_bytes
    from diffusionfas.supervision import run_bounded
    sha = digest(args.plan)
    require_approval(args.plan, sha, args.approval)
    root = args.plan.parent
    if any((root/name).exists() for name in ("batch.log", "started.json", "supervision.json")):
        raise FileExistsError("regional study already attempted; no retry")
    if device_memory_bytes() >= BUDGET["max_gpu_bytes"]:
        raise MemoryError("shared device already exceeds the approved conservative GPU cap")
    supervision = run_bounded(
        [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
         "--approval", str(args.approval), "--worker", "--plan-sha256", sha],
        cwd=ROOT, owned_root=root, log=root/"batch.log",
        max_seconds=BUDGET["max_seconds"], max_rss_bytes=BUDGET["max_rss_bytes"],
        max_disk_bytes=BUDGET["max_output_bytes"], max_gpu_bytes=BUDGET["max_gpu_bytes"],
        gpu_memory_bytes=device_memory_bytes)
    supervision["plan_sha256"] = sha
    write_json(root/"supervision.json", supervision)
    report = summarize(args.plan, supervision)
    write_json(root/"aggregate.json", report)
    print({"completed": report["completed"], "aggregate_sha256": digest(root/"aggregate.json")}, flush=True)
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
