"""Seal or supervise one exactly approved four-arm short-clip DINO probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True
for key, value in {"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "2",
                   "OMP_NUM_THREADS": "2", "XFORMERS_DISABLED": "1",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
    os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--fas-root", type=Path, required=True)
    plan.add_argument("--private-root", type=Path, required=True)
    plan.add_argument("--adapted-root", type=Path, required=True)
    plan.add_argument("--oulu-root", type=Path, required=True)
    plan.add_argument("--detector", type=Path, required=True)
    plan.add_argument("--data-root", type=Path, required=True)
    for name in ("approve", "run", "_worker", "_verify", "check"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--plan-sha256", required=True)
        if name == "approve":
            command.add_argument("--answer", required=True)
        if name in ("run", "_worker", "_verify"):
            command.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise RuntimeError("use the existing fas conda environment")
    import cv2
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    from diffusionfas.clip_study import (
        ANSWER, BUDGET, POLICY, SCOPE, build_plan, checked_plan, preparation_summary,
        completed_summary, require_approval, run_worker, verify_inputs,
    )
    from diffusionfas.io import digest, read_json, write_json
    if args.command == "plan":
        path = build_plan(args.fas_root, args.private_root, adapted_root=args.adapted_root,
                          oulu_root=args.oulu_root, detector=args.detector, data_root=args.data_root)
        write_json(path.parent/"preparation.json", preparation_summary(read_json(path), digest(path)))
        print({"plan_sha256": digest(path), "scientific_execution_authorized": False}, flush=True)
        return
    if args.command == "check":
        value = checked_plan(args.plan, args.plan_sha256)
        begun = time.monotonic()

        def metadata_guard() -> None:
            if time.monotonic()-begun > 1800:
                raise TimeoutError("metadata check 30-minute cap")

        verify_inputs(value, metadata_guard)
        print({"sealed_inputs_verified": True, "new_PAD_decodes": 0}, flush=True)
        return
    if args.command == "approve":
        checked_plan(args.plan, args.plan_sha256)
        if args.answer != ANSWER or (args.plan.parent/"attempt.json").exists():
            raise PermissionError("exact owner answer and unattempted plan required")
        receipt = {"answer": args.answer, "plan_sha256": args.plan_sha256,
                   "scope": SCOPE, "policy": POLICY, "budget": BUDGET,
                   "automatic_retry": False, "approved_monotonic": time.monotonic(),
                   "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        write_json(args.plan.parent/"approval.json", receipt)
        print({"approval_sha256": digest(args.plan.parent/"approval.json")}, flush=True)
        return
    if args.command == "_worker":
        run_worker(args.plan, args.plan_sha256, args.approval)
        return
    if args.command == "_verify":
        from diffusionfas.clip_verify import verify_completed
        verify_completed(args.plan, args.plan_sha256, args.approval)
        return
    sealed, approval = require_approval(args.plan, args.plan_sha256, args.approval)
    root = args.plan.parent
    if any((root/name).exists() for name in ("attempt.json", "started.json", "batch.log", "supervision.json")):
        raise FileExistsError("one clip attempt only; no retry or clock reset")
    from diffusionfas.adapter_pilot import device_memory_bytes
    from diffusionfas.supervision import run_bounded
    if device_memory_bytes() >= BUDGET["max_gpu_bytes"]:
        raise MemoryError("shared device already exceeds the conservative total-device GPU cap")
    write_json(root/"attempt.json", {"plan_sha256": args.plan_sha256, "supervisor_pid": os.getpid(),
                                    "approval_sha256": digest(args.approval)})
    supervision = run_bounded(
        [sys.executable, str(Path(__file__).resolve()), "_worker", "--plan", str(args.plan),
         "--plan-sha256", args.plan_sha256, "--approval", str(args.approval)],
        cwd=ROOT, owned_root=root, log=root/"batch.log",
        max_seconds=approval["approved_monotonic"]+BUDGET["max_seconds"]-time.monotonic(),
        max_rss_bytes=BUDGET["max_rss_bytes"], max_disk_bytes=BUDGET["max_output_bytes"],
        max_gpu_bytes=BUDGET["max_gpu_bytes"], gpu_memory_bytes=device_memory_bytes,
        rss_root_pid=os.getpid())
    write_json(root/"supervision.json", {"plan_sha256": args.plan_sha256, **supervision})
    if not supervision["completed"]:
        print({"completed": False, "supervision": supervision}, flush=True)
        raise SystemExit(1)
    verification = read_json(root/"verification.json")
    if verification["completed"] is not True or verification["plan_sha256"] != args.plan_sha256:
        raise ValueError("separate successful reconstruction required for completion")
    write_json(root/"aggregate.json", completed_summary(sealed, root, args.plan_sha256))
    from diffusionfas.supervision import tree_bytes
    if (tree_bytes(root)+4096 > BUDGET["max_output_bytes"]
            or time.monotonic() > approval["approved_monotonic"]+BUDGET["max_seconds"]):
        raise RuntimeError("clip closure exceeded shared wall/output cap")
    write_json(root/"completion.json", {"completed": True, "plan_sha256": args.plan_sha256,
                                      "verification_sha256": digest(root/"verification.json"),
                                      "aggregate_sha256": digest(root/"aggregate.json"),
                                      "supervision_sha256": digest(root/"supervision.json")})
    print({"completed": True, "verification": verification}, flush=True)


if __name__ == "__main__":
    main()
