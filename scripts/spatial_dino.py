"""Prepare, seal and supervise one exactly approved spatial DINO study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True
for name, value in {"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "2",
                    "OMP_NUM_THREADS": "2", "XFORMERS_DISABLED": "1", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
    os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--private-root", type=Path, required=True)
    crop_smoke = commands.add_parser("crop-smoke")
    crop_smoke.add_argument("--private-root", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--private-root", type=Path, required=True)
    prepare.add_argument("--probe", type=Path, required=True)
    worker = commands.add_parser("_smoke")
    worker.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("_replay")
    replay.add_argument("--checkpoint", type=Path, required=True)
    replay.add_argument("--checkpoint-sha256", required=True)
    for command in (smoke, crop_smoke, prepare, worker, replay):
        command.add_argument("--fas-root", type=Path, required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--preparation", type=Path, required=True)
    seal.add_argument("--preparation-sha256", required=True)
    seal.add_argument("--fas-root", type=Path, required=True)
    seal.add_argument("--private-root", type=Path, required=True)
    recovery = commands.add_parser("recover-seal")
    recovery.add_argument("--origin-plan", type=Path, required=True)
    recovery.add_argument("--private-root", type=Path, required=True)
    crop_seal = commands.add_parser("crop-seal")
    crop_seal.add_argument("--preparation", type=Path, required=True)
    crop_seal.add_argument("--verification", type=Path, required=True)
    crop_seal.add_argument("--probe", type=Path, required=True)
    crop_seal.add_argument("--private-root", type=Path, required=True)
    for name in ("check", "approve", "run", "_worker", "_verify"):
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
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    from diffusionfas.io import digest, read_json, require_private_output, write_json
    from diffusionfas.spatial_preparation import PROBE_CAPS, prepare, synthetic_probe, synthetic_replay
    if args.command == "crop-seal":
        from diffusionfas.spatial_crop_study import seal as seal_crops
        path = seal_crops(args.preparation, args.verification, args.probe, args.private_root)
        print(json.dumps({"crop_plan_sha256": digest(path), "scientific_execution_authorized": False}))
    elif args.command == "recover-seal":
        from diffusionfas.spatial_recovery import seal as seal_recovery
        path = seal_recovery(args.origin_plan, args.private_root)
        print(json.dumps({"recovery_plan_sha256": digest(path), "scientific_execution_authorized": False}))
    elif args.command in ("seal", "check", "approve", "run", "_worker", "_verify"):
        from diffusionfas.spatial_study import (
            approve, checked_plan, seal, supervise, verify_inputs, worker,
        )
        if args.command == "seal":
            path = seal(args.preparation, args.preparation_sha256, args.fas_root, args.private_root)
            print(json.dumps({"execution_plan_sha256": digest(path), "scientific_execution_authorized": False}))
        elif args.command == "check":
            import time
            begun = time.monotonic()

            def guard() -> None:
                if time.monotonic()-begun > 1800:
                    raise TimeoutError("spatial metadata check ceiling")

            verify_inputs(checked_plan(args.plan, args.plan_sha256), guard)
            print(json.dumps({"sealed_inputs_verified": True, "PAD_crop_decodes": 0}))
        elif args.command == "approve":
            path = approve(args.plan, args.plan_sha256, args.answer)
            print(json.dumps({"approval_sha256": digest(path)}))
        elif args.command == "_worker":
            worker(args.plan, args.plan_sha256, args.approval)
        elif args.command == "_verify":
            from diffusionfas.spatial_verify import verify_completed
            verify_completed(args.plan, args.plan_sha256, args.approval)
        else:
            summary = supervise(args.plan, args.plan_sha256, args.approval)
            print(json.dumps({"completed": summary["completed"], "plan_sha256": args.plan_sha256}))
    elif args.command == "_replay":
        print(json.dumps(synthetic_replay(args.checkpoint, args.checkpoint_sha256, args.fas_root)))
    elif args.command == "_smoke":
        synthetic_probe(args.output, args.fas_root)
    elif args.command == "prepare":
        path = prepare(args.fas_root, args.private_root, args.probe)
        print(json.dumps({"preparation_sha256": digest(path), "scientific_execution_authorized": False}))
    else:
        from diffusionfas.adapter_pilot import device_memory_bytes
        from diffusionfas.supervision import run_bounded
        caps = PROBE_CAPS
        if args.command == "crop-smoke":
            from diffusionfas.spatial_crop_study import BUDGET as crop_budget
            caps = {**PROBE_CAPS, "max_gpu_bytes": crop_budget["max_gpu_bytes"]}
        require_private_output(args.private_root/"probe.json")
        if args.private_root.exists():
            raise FileExistsError("new immutable synthetic probe directory required")
        if device_memory_bytes() >= caps["max_gpu_bytes"]:
            raise MemoryError("shared-device occupancy exceeds synthetic probe budget")
        args.private_root.mkdir(parents=True)
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "_smoke", "--fas-root", str(args.fas_root),
             "--output", str(args.private_root/"probe.json")],
            cwd=ROOT, owned_root=args.private_root, log=args.private_root/"probe.log",
            max_seconds=caps["max_seconds"], max_rss_bytes=caps["max_rss_bytes"],
            max_disk_bytes=caps["max_output_bytes"], max_gpu_bytes=caps["max_gpu_bytes"],
            gpu_memory_bytes=device_memory_bytes, rss_root_pid=os.getpid())
        write_json(args.private_root/"supervision.json",
                   {**result, **({"caps": caps} if args.command == "crop-smoke" else {})})
        if not result["completed"]:
            raise RuntimeError("synthetic probe failed; inspect retained probe.log, no automatic retry")
        value = read_json(args.private_root/"probe.json")
        print(json.dumps({"status": value["status"], "supervision": result,
                          "scientific_execution_authorized": False}))


if __name__ == "__main__":
    main()
