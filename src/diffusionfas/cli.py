"""Offline-first commands; scientific operations require explicit configuration gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

from .data import DOMAINS, load_manifest, select_source
from .fas_import import import_metadata
from .io import digest, object_digest, read_json, require_private_output, scientific_config, write_json
from .preflight import inventory, resources


def require_permission(config: dict[str, Any], operation: str) -> None:
    if config["execution"].get(operation) is not True:
        raise PermissionError(f"{operation} is not authorized; budget/approval required before enabling")


def manifest_payload(videos: list[Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "provenance": provenance, "videos": [v.to_dict() for v in videos]}


def run_import(arguments: argparse.Namespace) -> None:
    require_private_output(arguments.out)
    videos, provenance = import_metadata(arguments.source_manifest, arguments.partitions, arguments.oulu_media_root)
    write_json(arguments.out, manifest_payload(videos, provenance))
    print(json.dumps({"imported_originals": len(videos), "new_media_decodes": 0,
                      "new_training_runs": 0, "manifest_sha256": digest(arguments.out)}))


def run_preflight(arguments: argparse.Namespace) -> None:
    videos, provenance = load_manifest(arguments.manifest)
    config = read_json(arguments.config)
    report = inventory(videos)
    report["runtime"] = resources(arguments.storage_root)
    report["provenance"] = provenance
    report["manifest_sha256"] = digest(arguments.manifest)
    report["config_sha256"] = digest(arguments.config)
    report["blockers"] = [
        "No pretrained checkpoint acquired; no local safe bundle bound.",
        "Source videos remain archived unless local media paths are explicitly bound.",
        "New temporal tracking/crop acceptance is not established by historical single-frame success.",
        "Source training and target evaluation remain disabled until resource budget approval.",
        "Pretrained candidate native 256/linear versus scratch feasibility 128/cosine is NOT a matched architecture comparison.",
        "SiW subject identities unknown; no participant-disjointness certification.",
    ]
    report["configured_permissions"] = config["execution"]
    write_json(arguments.out, report)
    print(json.dumps({"originals": len(videos), "new_media_decodes": 0,
                      "new_training_runs": 0, "report_sha256": digest(arguments.out)}))


def run_bind(arguments: argparse.Namespace) -> None:
    require_private_output(arguments.out)
    videos, provenance = load_manifest(arguments.manifest)
    mapping = read_json(arguments.mapping)
    lookup = {v.key: v for v in videos}
    if not isinstance(mapping, dict) or not set(mapping) <= set(lookup):
        raise ValueError("mapping must contain only known original keys")
    for key, path in mapping.items():
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("absolute local media paths required")
        if digest(Path(path)) != lookup[key].media_sha256:
            raise ValueError("mapped media does not match bound original hash")
        lookup[key] = replace(lookup[key], media_path=str(Path(path).resolve()))
    provenance = {**provenance, "parent_manifest_sha256": digest(arguments.manifest),
                  "media_mapping_sha256": digest(arguments.mapping)}
    write_json(arguments.out, manifest_payload([lookup[v.key] for v in videos], provenance))


def run_prepare(arguments: argparse.Namespace) -> None:
    from .native_replay import FasNativeReplay, decoder_policy
    from .preprocessing import FasFaceProvider, MediaFailure, prepare_video

    config = read_json(arguments.config)
    require_permission(config, "target_decode" if arguments.partition == "evaluation" else "source_decode")
    require_private_output(arguments.out)
    require_private_output(arguments.cache_root / "placeholder")
    videos, provenance = load_manifest(arguments.manifest)
    if arguments.include_attacks and arguments.partition not in {"calibration", "evaluation"}:
        raise ValueError("attack preparation restricted to source calibration")
    selected = (
        [v for v in videos if v.dataset == config["target"] and v.partition == "evaluation"]
        if arguments.partition == "evaluation" else
        select_source(videos, config["target"], arguments.partition)
    )
    selected = [v for v in selected if v.label == "bona_fide" or arguments.include_attacks
                or arguments.partition == "evaluation"]
    if not selected:
        raise ValueError("no declared originals selected")
    budget = config["budget"]
    if not budget["max_seconds"] or not budget["max_output_bytes"]:
        raise ValueError("explicit preparation wall/output budget required")
    if arguments.cache_root.exists():
        raise ValueError("preparation requires a new cache directory")
    import time

    deadline = time.monotonic() + budget["max_seconds"]
    decoders, helper_sha = decoder_policy(config)
    native_videos = [v for v in selected if v.status == "eligible_pending_tracking"
                     and decoders[v.dataset] == "fas_strict_ffmpeg"]
    replays = {}
    native_source_sha = None
    if native_videos:
        source_manifest = getattr(arguments, "source_manifest", None)
        native_source_sha = provenance.get("source_manifest_sha256")
        if source_manifest is None or not native_source_sha or helper_sha is None:
            raise ValueError("native decoder requires --source-manifest bound by imported provenance")
        native = FasNativeReplay(arguments.fas_root, helper_sha, source_manifest, native_source_sha)
        replays = {v.key: native.bind(v, arguments.cache_root / "native-replay" / object_digest(v.key))
                   for v in native_videos}
    provider = FasFaceProvider(arguments.fas_root, arguments.preprocessing_sha256,
                               arguments.detector, arguments.detector_sha256)
    lookup = {v.key: v for v in videos}
    for video in selected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("source preparation wall budget exhausted")
        if video.status != "eligible_pending_tracking":
            continue
        used = sum(p.stat().st_size for p in arguments.cache_root.rglob("*") if p.is_file())
        remaining_bytes = budget["max_output_bytes"] - used
        if remaining_bytes <= 0:
            raise OSError("source preparation output budget exhausted")
        output = arguments.cache_root / f"{object_digest(video.key)}.npz"
        try:
            lookup[video.key] = prepare_video(video, provider, output, fps=config["clip"]["fps"],
                                              max_seconds=min(remaining, 180),
                                              max_output_bytes=remaining_bytes,
                                              native_replay=replays.get(video.key))
        except MediaFailure as error:
            lookup[video.key] = replace(video, status=f"preparation_failure:{error}")
        if sum(p.stat().st_size for p in arguments.cache_root.rglob("*") if p.is_file()) > budget["max_output_bytes"]:
            raise OSError("preparation output budget exhausted including native diagnostics")
    provenance = {**provenance, "parent_manifest_sha256": digest(arguments.manifest),
                  "preprocessing_sha256": arguments.preprocessing_sha256,
                  "detector_sha256": arguments.detector_sha256,
                  "clip_config": config["clip"],
                  "decoder_policy": {"decoders": decoders, "native_replay_helper_sha256": helper_sha,
                                     "source_manifest_sha256": native_source_sha}}
    write_json(arguments.out, manifest_payload([lookup[v.key] for v in videos], provenance))


def run_train(arguments: argparse.Namespace) -> None:
    import torch

    from .models import build_model
    from .training import train

    config = read_json(arguments.config)
    require_permission(config, "source_training")
    require_private_output(arguments.out / "placeholder")
    videos, _ = load_manifest(arguments.manifest)
    torch.manual_seed(config["seed"])
    model = build_model(config)
    result = train(model, videos, config, arguments.out, device=arguments.device, resume=arguments.resume)
    write_json(arguments.out / "completion.json", {**result, "config_sha256": digest(arguments.config),
                                                  "manifest_sha256": digest(arguments.manifest)})
    print(json.dumps(result))


def run_profiles(arguments: argparse.Namespace) -> None:
    import torch

    from .dataset import ClipCache
    from .models import build_model
    from .scoring import residual_profile
    from .training import check_source_pools, checkpoint_binding, install_trainable

    config = read_json(arguments.config)
    require_permission(config, "target_evaluation" if arguments.partition == "evaluation" else "source_scoring")
    require_private_output(arguments.out)
    videos, _ = load_manifest(arguments.manifest)
    selected = (
        [v for v in videos if v.dataset == config["target"] and v.partition == "evaluation"]
        if arguments.partition == "evaluation"
        else select_source(videos, config["target"], arguments.partition)
    )
    if arguments.include_attacks and arguments.partition == "validation":
        raise ValueError("validation profile selection is bonafide-only")
    if not arguments.include_attacks and arguments.partition != "evaluation":
        selected = [v for v in selected if v.label == "bona_fide"]
    if not selected:
        raise ValueError("empty scoring population")
    if arguments.middle_frame and config["model"]["variant"].endswith("temporal"):
        raise ValueError("middle-frame comparison uses spatial checkpoint, not a different temporal input")
    torch.manual_seed(config["seed"])
    model = build_model(config).to(arguments.device)
    if arguments.checkpoint_root is not None:
        fitting, validation, _ = check_source_pools(videos, config["target"])
        binding = checkpoint_binding(config, fitting + validation)
        last = torch.load(arguments.checkpoint_root / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(arguments.checkpoint_root / "best.pt", map_location="cpu", weights_only=True)
        if last["binding"] != binding or best["binding"] != binding or last["complete"] is not True:
            raise ValueError("completed matching training and best checkpoint required")
        if best["best_epoch"] != last["best_epoch"]:
            raise ValueError("best/last checkpoint selection mismatch")
        install_trainable(model, best["ema"])
        checkpoint_sha = digest(arguments.checkpoint_root / "best.pt")
    else:
        if config["model"]["variant"] != "pretrained_spatial":
            raise ValueError("trainable model scoring requires completed source checkpoints")
        checkpoint_sha = config["model"]["bundle_manifest_sha256"]
    cache = ClipCache(model.resolution, config["clip"]["duration"], config["clip"]["fps"])
    results = []
    for video in selected:
        row = {"key": video.key, "dataset": video.dataset, "group": video.group,
               "partition": video.partition, "label": video.label, "attack_family": video.attack_family}
        if video.status != "prepared":
            row.update(profile=None, failure=video.status)
        else:
            clip = cache.get(video)[None].to(arguments.device)
            addresses = None
            if arguments.middle_frame:
                middle = (clip.shape[1] - 1) // 2
                clip = clip[:, middle:middle + 1]
                addresses = [middle]
            profile = residual_profile(model, clip, [video.key], seed=config["seed"],
                                       frame_addresses=addresses, **config["scoring"])
            row.update(profile=profile[0].cpu().tolist(), failure=None)
        results.append(row)
    write_json(arguments.out, {
        "schema_version": 1, "target": config["target"], "config": config,
        "scoring_binding": object_digest({"config": scientific_config(config), "checkpoint": checkpoint_sha,
                                          "middle_frame": arguments.middle_frame}),
        "checkpoint_sha256": checkpoint_sha, "records": results,
    })


def run_calibrate(arguments: argparse.Namespace) -> None:
    import numpy as np

    from .metrics import Prediction, attack_threshold, strict_threshold
    from .scoring import RobustNormalizer

    require_private_output(arguments.out)
    validation, calibration = read_json(arguments.validation), read_json(arguments.calibration)
    if validation["scoring_binding"] != calibration["scoring_binding"]:
        raise ValueError("validation/calibration model and scoring bindings differ")
    target = validation["target"]
    usable = [r for r in validation["records"] if r["profile"] is not None]
    if any(r["dataset"] == target or r["partition"] != "validation" or r["label"] != "bona_fide"
           for r in validation["records"]):
        raise ValueError("all normalization records must be source validation bonafide")
    if not usable:
        raise ValueError("no usable source validation profiles")
    normalizer = RobustNormalizer.fit(np.array([r["profile"] for r in usable]), usable, target)
    rows = []
    for record in calibration["records"]:
        score = None if record["profile"] is None else float(normalizer.transform(np.array([record["profile"]]))[0])
        rows.append(Prediction(**{k: record[k] for k in ("key", "dataset", "group", "partition", "label")},
                               score=score, failure=record["failure"], attack_family=record.get("attack_family")))
    threshold = (
        strict_threshold(rows, target, arguments.rate) if arguments.regime == "strict"
        else attack_threshold(rows, target, arguments.rate)
    )
    write_json(arguments.out, {
        "target": target, "scoring_binding": validation["scoring_binding"],
        "normalizer": asdict(normalizer), "regime": arguments.regime, "rate": arguments.rate,
        "threshold": threshold, "validation_sha256": digest(arguments.validation),
        "calibration_sha256": digest(arguments.calibration),
        "calibration_originals": len(rows),
    })


def run_evaluate(arguments: argparse.Namespace) -> None:
    import numpy as np

    from .metrics import Prediction, operating_metrics, oracle_bfa, paired_bootstrap
    from .scoring import RobustNormalizer

    profiles, calibration = read_json(arguments.profiles), read_json(arguments.calibration)
    require_permission(profiles["config"], "target_evaluation")
    if profiles["scoring_binding"] != calibration["scoring_binding"] or profiles["target"] != calibration["target"]:
        raise ValueError("target profiles do not match frozen source calibration")
    if not profiles["records"] or any(
        r["dataset"] != profiles["target"] or r["partition"] != "evaluation" for r in profiles["records"]
    ):
        raise ValueError("only declared outer target evaluation originals allowed")
    normalizer = RobustNormalizer(**calibration["normalizer"])
    predictions = []
    for record in profiles["records"]:
        score = None if record["profile"] is None else float(normalizer.transform(np.array([record["profile"]]))[0])
        predictions.append(Prediction(
            **{k: record[k] for k in ("key", "dataset", "group", "partition", "label")},
            score=score, failure=record["failure"], attack_family=record.get("attack_family"),
        ))
    report = operating_metrics(predictions, calibration["threshold"])
    report["oracle_at_5pct_FA"] = oracle_bfa(predictions)
    report["conditional_bootstrap"] = paired_bootstrap(
        {"model": predictions}, {"model": calibration["threshold"]}, repetitions=arguments.bootstrap,
    )
    report.update(target=profiles["target"], regime=calibration["regime"],
                  profiles_sha256=digest(arguments.profiles),
                  source_calibration_sha256=digest(arguments.calibration))
    write_json(arguments.out, report)


def run_feature_baseline(arguments: argparse.Namespace) -> None:
    import numpy as np

    from .baselines import FeatureGaussian, FrozenDino, clip_feature
    from .dataset import ClipCache
    from .training import check_source_pools

    config = read_json(arguments.config)
    require_permission(config, "source_scoring")
    require_permission(config, "source_training")
    if arguments.target_profiles:
        require_permission(config, "target_evaluation")
    require_private_output(arguments.out / "placeholder")
    if arguments.out.exists():
        raise ValueError("feature baseline requires a new output directory")
    videos, _ = load_manifest(arguments.manifest)
    fitting, _, _ = check_source_pools(videos, config["target"])
    encoder = FrozenDino(arguments.source_checkout, arguments.source_revision, arguments.checkpoint,
                         arguments.checkpoint_sha256, arguments.device)
    cache = ClipCache(224, config["clip"]["duration"], config["clip"]["fps"])
    features = np.stack([clip_feature(encoder.encode(cache.get(v)), temporal=arguments.temporal) for v in fitting])
    gaussian = FeatureGaussian.fit(features, fitting, config["target"])
    binding = object_digest({
        "encoder": arguments.checkpoint_sha256, "source_revision": arguments.source_revision,
        "fitting": [v.to_dict() for v in fitting], "clip": config["clip"],
        "temporal": arguments.temporal, "gaussian": "equal-source-LedoitWolf-v1",
    })
    for partition in ("validation", "calibration", "evaluation"):
        if partition == "evaluation" and not arguments.target_profiles:
            continue
        selected = (
            [v for v in videos if v.dataset == config["target"] and v.partition == partition]
            if partition == "evaluation" else select_source(videos, config["target"], partition)
        )
        if partition == "validation" or (partition == "calibration" and not arguments.include_attacks):
            selected = [v for v in selected if v.label == "bona_fide"]
        records = []
        for video in selected:
            row = {"key": video.key, "dataset": video.dataset, "group": video.group,
                   "partition": video.partition, "label": video.label, "attack_family": video.attack_family}
            if video.status != "prepared":
                row.update(profile=None, failure=video.status)
            else:
                feature = clip_feature(encoder.encode(cache.get(video)), temporal=arguments.temporal)
                row.update(profile=[float(gaussian.score(feature[None])[0])], failure=None)
            records.append(row)
        write_json(arguments.out / f"{partition}.json", {
            "schema_version": 1, "target": config["target"], "config": config,
            "scoring_binding": binding, "external_pretraining": True, "records": records,
        })
    with (arguments.out / "gaussian.npz").open("xb") as stream:
        np.savez(stream, location=gaussian.location, precision=gaussian.precision)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("import-fas", help="metadata only; no media decoding")
    command.add_argument("--source-manifest", type=Path, required=True)
    command.add_argument("--partitions", type=Path, required=True)
    command.add_argument("--oulu-media-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_import)

    command = commands.add_parser("preflight", help="aggregate saved metadata and runtime only")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--storage-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_preflight)

    command = commands.add_parser("bind-media", help="bind explicitly staged local originals by hash")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_bind)

    command = commands.add_parser("prepare", help="permission-gated source crop/cache preparation")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--partition", choices=("fitting", "validation", "calibration", "evaluation"), required=True)
    command.add_argument("--include-attacks", action="store_true")
    command.add_argument("--cache-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--fas-root", type=Path, required=True)
    command.add_argument("--preprocessing-sha256", required=True)
    command.add_argument("--detector", type=Path, required=True)
    command.add_argument("--detector-sha256", required=True)
    command.add_argument("--source-manifest", type=Path,
                         help="accepted FAS source manifest required by strict native decoder policy")
    command.set_defaults(run=run_prepare)

    command = commands.add_parser("train", help="disabled until explicit budget/source-training approval")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--device", default="cpu")
    command.add_argument("--resume", type=Path)
    command.set_defaults(run=run_train)

    command = commands.add_parser("profiles", help="source/target scoring with separate execution gates")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--partition", choices=("validation", "calibration", "evaluation"), required=True)
    command.add_argument("--checkpoint-root", type=Path)
    command.add_argument("--device", default="cpu")
    command.add_argument("--include-attacks", action="store_true")
    command.add_argument("--middle-frame", action="store_true")
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_profiles)

    command = commands.add_parser("calibrate", help="source-only validation normalization and thresholds")
    command.add_argument("--validation", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--regime", choices=("strict", "attack-aware"), required=True)
    command.add_argument("--rate", type=float, default=0.05)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_calibrate)

    command = commands.add_parser("evaluate", help="apply frozen source policy; never choose a target threshold")
    command.add_argument("--profiles", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--bootstrap", type=int, default=2000)
    command.set_defaults(run=run_evaluate)

    command = commands.add_parser("feature-baseline", help="local-only frozen DINO plus one-class Gaussian")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--source-checkout", type=Path, required=True)
    command.add_argument("--source-revision", required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--checkpoint-sha256", required=True)
    command.add_argument("--device", default="cpu")
    command.add_argument("--temporal", action="store_true")
    command.add_argument("--include-attacks", action="store_true")
    command.add_argument("--target-profiles", action="store_true")
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_feature_baseline)
    return root


def main() -> None:
    arguments = parser().parse_args()
    try:
        arguments.run(arguments)
    except (ValueError, PermissionError, OSError, RuntimeError) as error:
        print(f"diffusionfas {arguments.command} failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
