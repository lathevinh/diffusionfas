"""Explicit research-only MAT mirror acquisition and private visual pilot."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
import hashlib
import html
import importlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from .acquisition import download
from .adapter_pilot import configure_gpu
from .crop_ablation import approved_preview, PREVIEW_SHA
from .crop_preview import read_native_frames, save_png
from .data import addressed_seed
from .fas_import import MANIFEST_SHA
from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .mat_input import POLICY, anatomical_masks, crop_native512, masked_error, model_input
from .native_replay import FasNativeReplay, decoder_policy
from .preprocessing import MediaFailure
from .source_preparation import item_video
from .source_probe import stage_member
from .supervision import tree_bytes

BUDGET = {"max_seconds": 1800, "max_rss_bytes": 8 * 1024**3,
          "max_gpu_bytes": 8 * 1024**3, "max_output_bytes": 4 * 1024**3,
          "max_public_download_bytes": 2 * 1024**3}
UPSTREAM_REVISION = "d273d891ecdad2e1df106516423a75bc45b2d800"
MODEL_ID = "not-lain/MAT-CelebA-HQ_512"
MODEL_REVISION = "b9dd007aea6d7c9e53818f2d11b071b1be5bf48c"
WEIGHT_SHA = "b47eb8ff60bc25131b531eab96cf8f8d4e85a86da688c7595ea50003df93d8bb"
WEIGHT_BYTES = 253303072
MODEL_CONFIG = {"c_dim": 0, "img_channels": 3, "img_resolution": 512, "w_dim": 512, "z_dim": 512}
AUTHORITY = {"approved_third_party_mirror_research_only": True, "official_weight_equivalence_verified": False,
             "preview_only": True, "source_validation_MSU_only": True, "AUROC": False,
             "training": False, "fitting_pixels": False, "calibration_pixels": False,
             "target_pixels": False, "automatic_retry": False}


def source_files(root: Path) -> dict[str, str]:
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                           capture_output=True, text=True, timeout=10).stdout.strip()
    if revision != UPSTREAM_REVISION or dirty:
        raise ValueError("clean pinned official MAT source required")
    paths = sorted(p for p in root.rglob("*.py") if ".git" not in p.parts)
    return {str(p.relative_to(root)): digest(p) for p in [*paths, root / "LICENSE", root / "README.md"]}


def acquired_weights(root: Path) -> dict[str, Any]:
    receipt = read_json(root / "acquisition.json")
    if (receipt["model_id"] != MODEL_ID or receipt["revision"] != MODEL_REVISION
            or receipt["format"] != "safetensors" or receipt["unsafe_pickle_used"] is not False
            or read_json(root / "weights/config.json") != MODEL_CONFIG
            or digest(root / "weights/model.safetensors") != WEIGHT_SHA
            or (root / "weights/model.safetensors").stat().st_size != WEIGHT_BYTES):
        raise ValueError("pinned already acquired MAT safe weights required")
    return receipt


def build_plan(preview_root: Path, upstream: Path, output: Path, reuse_root: Path | None = None) -> Path:
    require_private_output(output / "plan.json")
    if output.exists():
        raise ValueError("new MAT pilot root required")
    preview, _ = approved_preview(preview_root)
    if len(preview["selected"]) != 6:
        raise ValueError("fixed previously reviewed six-video preview required")
    files = source_files(upstream)
    if tree_bytes(upstream) + WEIGHT_BYTES + 4096 > BUDGET["max_public_download_bytes"]:
        raise ValueError("MAT source plus fixed checkpoint exceed acquisition byte cap")
    project = Path(__file__).resolve().parents[2]
    plan = {"scope": "private_MAT_512_masked_reconstruction_visual_pilot",
            "authority": AUTHORITY, "budget": BUDGET, "policy": POLICY,
            "upstream_root": str(upstream.resolve()), "upstream_revision": UPSTREAM_REVISION,
            "upstream_files_sha256": files, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "weight_sha256": WEIGHT_SHA, "weight_bytes": WEIGHT_BYTES, "model_config": MODEL_CONFIG,
            "preview_root": str(preview_root.resolve()), "preview_plan_sha256": PREVIEW_SHA,
            "crop_acceptance_sha256": digest(preview_root / "visual-approval.json"),
            "entries": preview["selected"], "fas_root": preview["fas_root"],
            "source_manifest": preview["source_manifest"], "decoder_config": preview["decoder_config"],
            "msu_reader_sha256": preview["msu_reader_sha256"],
            "seed": 20261103, "implementation_sha256": {
                str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                for p in sorted((project / directory).rglob("*.py"))}}
    if reuse_root is not None:
        acquired_weights(reuse_root)
        plan["reuse_acquisition_root"] = str(reuse_root.resolve())
        plan["reuse_acquisition_sha256"] = digest(reuse_root / "acquisition.json")
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["authority"] != AUTHORITY or plan["budget"] != BUDGET or plan["policy"] != POLICY
            or plan["scope"] != "private_MAT_512_masked_reconstruction_visual_pilot"
            or plan["model_id"] != MODEL_ID or plan["model_revision"] != MODEL_REVISION
            or plan["weight_sha256"] != WEIGHT_SHA or plan["weight_bytes"] != WEIGHT_BYTES
            or plan["model_config"] != MODEL_CONFIG or plan["seed"] != 20261103):
        raise PermissionError("fixed explicitly approved mirror, preview authority and caps required")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("MAT pilot implementation drift")
    if source_files(Path(plan["upstream_root"])) != plan["upstream_files_sha256"]:
        raise ValueError("MAT upstream code drift")
    preview, _ = approved_preview(Path(plan["preview_root"]))
    read_json(Path(plan["preview_root"]) / "visual-approval.json", plan["crop_acceptance_sha256"])
    for name in ("fas_root", "source_manifest", "decoder_config", "msu_reader_sha256"):
        if plan[name] != preview[name]:
            raise ValueError("MAT native decoder lineage drift")
    if plan["entries"] != preview["selected"]:
        raise ValueError("MAT must retain the exact approved six videos and frame slots")
    if "reuse_acquisition_root" in plan:
        acquired_weights(Path(plan["reuse_acquisition_root"]))
        read_json(Path(plan["reuse_acquisition_root"]) / "acquisition.json", plan["reuse_acquisition_sha256"])
    return plan


def reference_operator(function):
    @wraps(function)
    def reference(*args, **kwargs):
        kwargs["impl"] = "ref"
        return function(*args, **kwargs)
    return reference


def configure_mat_gpu(plan: dict[str, Any]) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    report = configure_gpu(plan)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    return {**report, "deterministic_algorithms": True, "cudnn_deterministic": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def load_mat(plan: dict[str, Any], weights: Path):
    if digest(weights) != WEIGHT_SHA:
        raise ValueError("MAT safe checkpoint hash mismatch")
    upstream = Path(plan["upstream_root"])
    sys.path.insert(0, str(upstream))
    module = importlib.import_module("networks.mat")
    if Path(module.__file__).resolve() != (upstream / "networks/mat.py").resolve():
        raise ValueError("MAT imported from unbound source")
    bias = importlib.import_module("torch_utils.ops.bias_act")
    fir = importlib.import_module("torch_utils.ops.upfirdn2d")
    for imported in (bias, fir):
        if not Path(imported.__file__).resolve().is_relative_to(upstream.resolve()):
            raise ValueError("MAT operators imported from unbound source")
    # Explicit upstream reference operators, not a silent CUDA-build fallback.
    bias.bias_act = reference_operator(bias.bias_act)
    fir.upfirdn2d = reference_operator(fir.upfirdn2d)
    model = module.Generator(**MODEL_CONFIG).eval().requires_grad_(False)
    state = load_file(str(weights), device="cpu")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite MAT safe checkpoint")
    model.load_state_dict(state, strict=True)
    return model.cuda()


@torch.no_grad()
def reconstruct(model, rgb: np.ndarray, known: np.ndarray, seed: int,
                device: torch.device) -> np.ndarray:
    image, mask = model_input(rgb, known)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        z = np.random.default_rng(seed).standard_normal((1, 512)).astype(np.float32)
        image, mask = image.to(device), mask.to(device)
        label = torch.zeros((1, 0), device=device)
        output = model(image, mask, torch.from_numpy(z).to(device), label,
                       truncation_psi=1., noise_mode="const")
    if output.shape != (1, 3, 512, 512) or not torch.isfinite(output).all():
        raise FloatingPointError("invalid MAT reconstruction")
    torch.testing.assert_close(output * mask, image * mask, rtol=0, atol=0)
    return output[0].permute(1, 2, 0).cpu().numpy()


def smoke_test(model) -> dict[str, Any]:
    yy, xx = np.indices((512, 512))
    rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    known = np.ones((512, 512), dtype=np.float32)
    known[160:256, 200:312] = 0
    first = reconstruct(model, rgb, known, 123, torch.device("cuda"))
    second = reconstruct(model, rgb, known, 123, torch.device("cuda"))
    changed = rgb.copy()
    changed[known == 0] = 255 - changed[known == 0]
    hidden = reconstruct(model, changed, known, 123, torch.device("cuda"))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, hidden)
    return {"completed": True, "synthetic_forward_passes": 3,
            "same_seed_max_absolute_error": float(np.max(np.abs(first - second))),
            "hidden_ground_truth_change_max_absolute_error": float(np.max(np.abs(first - hidden))),
            "known_pixels_preserved_exactly": True, "operators": "upstream_ref_on_CUDA",
            "runtime_versions": {name: version(name) for name in ("torch", "timm", "safetensors")}}


def render_review(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    import base64

    def embed(image: dict[str, str], label: str) -> str:
        path = root / image["file"]
        if not path.resolve().is_relative_to((root / "images").resolve()) or digest(path) != image["sha256"]:
            raise ValueError("MAT preview PNG binding drift")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><figcaption>{label}</figcaption><img src="data:image/png;base64,{encoded}" alt="{label}"></figure>'

    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
             '<title>Private MAT face inpainting review</title><style>',
             'body{font:16px system-ui;background:#edf2f7;color:#162536;margin:24px}main{max-width:1500px;margin:auto}',
             'article{background:white;padding:18px;margin:20px 0;border-radius:12px}.row{display:flex;flex-wrap:wrap;gap:14px}',
             'figure{margin:8px 0;max-width:31%}img{width:100%;max-width:512px}figcaption{font-size:13px}',
             '@media(max-width:700px){figure{max-width:100%}}</style></head><body><main>',
             '<h1>MAT CelebA-HQ-512: anatomical inpainting</h1>',
             '<p><strong>PAUSED FOR HUMAN REVIEW. No AUROC, training or threshold fitting.</strong></p>',
             '<p>Approved third-party safetensors mirror; equivalence to author weights is UNVERIFIED.</p>',
             '<p>Native RGB to oriented square 512, no synthetic context or 256-cache upscale. '
             'Interpolation may enlarge limited native detail. Four independently applied masks; 0=hole, 1=known.</p>',
             '<p>Fixed latent and internal-dropout seeds per original/frame/region. Small anatomical masks may '
             'differ from training masks. Inspect position, face scale, head coverage and reconstruction artifacts.</p>']
    frames, predictions, failures = 0, 0, 0
    for index, entry in enumerate(plan["entries"]):
        row = read_json(root / "records" / f"result-{index:02d}.json")
        parts.append(f'<article><h2>Video {index + 1}: {html.escape(entry["video"]["label"])}</h2>')
        if row["failure"]:
            failures += 1
            parts.append(f'<p>Failure: {html.escape(row["failure"])}</p>')
        for frame in row["frames"]:
            frames += 1
            parts.append(f'<h3>Clip slot {frame["clip_slot"] + 1}/16</h3>' + embed(frame["original"], "Native-derived 512 input"))
            for name, result in frame["masks"].items():
                predictions += 1
                parts.append(f'<h4>{html.escape(name)}; hole {result["hole_fraction"]:.2%}; '
                             f'masked MSE {result["masked_mse"]:.6f}</h4><div class="row">')
                for kind, label in (("masked", "Masked input"), ("restored", "MAT reconstruction"),
                                    ("residual", "Squared error inside hole only; fixed color scale")):
                    parts.append(embed(result["images"][kind], label))
                parts.append('</div>')
        parts.append('</article>')
    parts.append('</main></body></html>')
    content = "".join(parts).encode()
    if tree_bytes(root) + len(content) + 1024**2 > BUDGET["max_output_bytes"]:
        raise OSError("MAT self-contained HTML exceeds output budget")
    with (root / "review.html").open("xb") as stream:
        stream.write(content)
    return {"rendered_frames": frames, "reconstructions": predictions,
            "technical_failure_originals": failures, "html_sha256": digest(root / "review.html")}


def run_preview(path: Path, sha: str) -> None:
    plan = checked_inputs(path, sha)
    root = path.parent
    if (root / "started.json").exists():
        raise ValueError("MAT pilot already attempted; no automatic retry")
    write_json(root / "started.json", {"plan_sha256": sha})
    deadline = time.monotonic() + BUDGET["max_seconds"] - 20
    for folder in ("weights", "images", "staging", "records"):
        (root / folder).mkdir()
    if "reuse_acquisition_root" in plan:
        previous = Path(plan["reuse_acquisition_root"])
        receipt = acquired_weights(previous)
        weights = previous / "weights/model.safetensors"
        write_json(root / "acquisition.json", {**receipt, "reused_acquisition_sha256": plan["reuse_acquisition_sha256"],
                                               "new_download_bytes": 0})
    else:
        base = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}"
        config_receipt = download(f"{base}/config.json", root / "weights/config.json",
                                  maximum_bytes=4096, deadline=deadline)
        if read_json(root / "weights/config.json") != MODEL_CONFIG:
            raise ValueError("mirror native resolution/config differs from approved model")
        weights = root / "weights/model.safetensors"
        weight_receipt = download(f"{base}/model.safetensors", weights,
                                  maximum_bytes=WEIGHT_BYTES, deadline=deadline, expected_sha256=WEIGHT_SHA)
        if weight_receipt["bytes"] != WEIGHT_BYTES:
            raise ValueError("MAT checkpoint byte count mismatch")
        write_json(root / "acquisition.json", {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                   "config": config_receipt, "weights": weight_receipt, "format": "safetensors",
                   "official_weight_equivalence_verified": False, "unsafe_pickle_used": False})
    write_json(root / "device.json", configure_mat_gpu(plan))
    model = load_mat(plan, weights)
    state_sha = tensor_digest(model.state_dict())
    write_json(root / "smoke.json", smoke_test(model))
    print({"phase": "synthetic_smoke", "completed": True}, flush=True)
    cv2.setNumThreads(1)
    fas_root = Path(plan["fas_root"])
    if digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]:
        raise ValueError("MSU reader drift")
    decoders, helper_sha = decoder_policy(plan["decoder_config"])
    if decoders["MSU-MFSD"] != "fas_strict_ffmpeg" or helper_sha is None:
        raise ValueError("strict MSU native replay required")
    native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]), MANIFEST_SHA)
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        context: dict[str, Any] = {}
        for index, entry in enumerate(plan["entries"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("MAT shared wall cap")
            video = item_video(entry)
            staged = root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}"
            rendered, failure = [], None
            try:
                stage_member(entry, fas_root, fas_root, staged, stack, context, deadline,
                             allow_validation_attacks=True)
                replay = native.bind(replace(video, media_path=str(staged)),
                                     root / "native-replay" / f"original-{index:02d}",
                                     expected_receipt_sha256=entry["receipt_sha256"])
                frames = read_native_frames(video, staged, {s["native_order"] for s in entry["samples"]},
                                            min(deadline, time.monotonic() + 180), replay)
                for sample in entry["samples"]:
                    rgb, geometry = crop_native512(frames[sample["native_order"]], np.asarray(sample["landmarks"]))
                    prefix = f"v{index:02d}-f{sample['clip_slot']:02d}"
                    original = save_png(root / "images" / f"{prefix}-original.png", rgb)
                    reconstructions = {}
                    for name, known in anatomical_masks(np.asarray(geometry["output_landmarks"])).items():
                        seed = addressed_seed(plan["seed"], video.key, sample["clip_slot"], name, "MAT-preview")
                        prediction = reconstruct(model, rgb, known, seed, torch.device("cuda"))
                        mse, error = masked_error(rgb, prediction, known)
                        masked = rgb.copy()
                        masked[known == 0] = 128
                        restored = np.rint((prediction + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        heat = cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(error / 4, 0, 1) * 255).astype(np.uint8),
                                                             cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                        heat[known == 1] = 0
                        images = {kind: save_png(root / "images" / f"{prefix}-{name}-{kind}.png", image)
                                  for kind, image in (("masked", masked), ("restored", restored), ("residual", heat))}
                        reconstructions[name] = {"seed": seed, "masked_mse": mse, "hole_fraction": float((known == 0).mean()),
                                                 "known_mask_sha256": hashlib.sha256(known.tobytes()).hexdigest(),
                                                 "images": images}
                    rendered.append({"clip_slot": sample["clip_slot"], "geometry": geometry, "original": original,
                                     "native_RGB_sha256": video.frame_hashes[sample["native_order"]],
                                     "masks": reconstructions})
                del frames
            except MediaFailure as error:
                failure = str(error)
            finally:
                staged.unlink(missing_ok=True)
            write_json(root / "records" / f"result-{index:02d}.json",
                       {"index": index, "failure": failure, "frames": rendered})
            if tree_bytes(root) > BUDGET["max_output_bytes"]:
                raise OSError("MAT output cap")
            print({"phase": "preview", "originals": index + 1, "selected": 6, "failure": failure}, flush=True)
    if state_sha != tensor_digest(model.state_dict()):
        raise ValueError("MAT frozen model state changed")
    review = render_review(root, plan)
    write_json(root / "completion.json", {
        "completed": True, "plan_sha256": sha, "review": review, "human_review": "pending",
        "training_runs": 0, "AUROC_runs": 0, "threshold_fitting_runs": 0,
        "fitting_pixel_access": 0, "calibration_pixel_access": 0, "target_pixel_access": 0,
        "frozen_model_sha256_before_and_after": state_sha,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "records_sha256": {p.name: digest(p) for p in sorted((root / "records").glob("*.json"))}})


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    root = path.parent
    plan = read_json(path)
    for index, entry in enumerate(plan["entries"]):
        (root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}").unlink(missing_ok=True)
    completion = read_json(root / "completion.json") if (root / "completion.json").exists() else None
    if completion is not None:
        if (completion["plan_sha256"] != digest(path)
                or completion["review"]["html_sha256"] != digest(root / "review.html")):
            raise ValueError("MAT completion/HTML binding drift")
        for name, expected in completion["records_sha256"].items():
            if digest(root / "records" / name) != expected:
                raise ValueError("MAT receipt drift")
    public = None if completion is None else {k: v for k, v in completion.items() if k != "records_sha256"}
    return {"completed": bool(supervision["completed"] and completion is not None),
            "supervision": supervision, "completion": public, "policy": POLICY, "authority": AUTHORITY,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA,
            "upstream_revision": UPSTREAM_REVISION, "plan_sha256": digest(path),
            "smoke": read_json(root / "smoke.json") if (root / "smoke.json").exists() else None,
            "human_review": "pending", "automatic_continuation": False}
