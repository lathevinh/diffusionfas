"""Create a private CPU-only crop review and stop for explicit human approval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionfas.io import digest, read_json, write_json
from diffusionfas.supervision import run_bounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--diagnostic-root", type=Path, required=True)
    plan.add_argument("--private-root", type=Path, required=True)
    plan.add_argument("--real-pixel-audit-root", type=Path,
                      help="select six MSU validation originals from the pinned no-padding audit")
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("crop preview requires conda env fas")
    from diffusionfas.crop_preview import BUDGET, build_plan, checked_inputs, run_preview, summarize_preview
    if args.command == "plan":
        path = build_plan(args.diagnostic_root, args.private_root, args.real_pixel_audit_root)
        print({"plan_sha256": digest(path)}, flush=True)
    elif args.worker:
        if args.plan_sha256 is None:
            raise ValueError("preview worker needs parent plan commitment")
        run_preview(args.plan, args.plan_sha256)
    else:
        sha = digest(args.plan)
        checked_inputs(args.plan, sha)
        root = args.plan.parent
        if any((root / name).exists() for name in ("batch.log", "supervision.json", "started.json", "review.html")):
            raise ValueError("preview already attempted; no automatic retry")
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
             "--worker", "--plan-sha256", sha],
            cwd=ROOT, owned_root=root, log=root / "batch.log",
            max_seconds=BUDGET["max_seconds"], max_rss_bytes=BUDGET["max_rss_bytes"],
            max_disk_bytes=BUDGET["max_output_bytes"])
        result["plan_sha256"] = sha
        write_json(root / "supervision.json", result)
        if digest(args.plan) != sha:
            raise ValueError("preview plan changed during execution")
        report = summarize_preview(args.plan, result)
        write_json(root / "aggregate.json", report)
        print({"completed": report["completed"], "human_review": "pending",
               "html": str(root / "review.html"), "counts": report["counts"]}, flush=True)
        if not report["completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
