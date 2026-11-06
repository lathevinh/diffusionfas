"""Hash-bound, saved-score SiW attack-family diagnosis; no media or model execution."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
import time
from typing import Any

import numpy as np

from .io import digest, object_digest, read_json, require_private_output, write_json

BUDGET = {"max_seconds": 1800, "max_rss_bytes": 2*1024**3, "max_output_bytes": 64*1024**2}
ALPHAS = (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10))
TARGET = "SiW-Mv2"
ROLES = ("fitting", "validation", "calibration", "evaluation")


def identity(row: dict) -> str:
    return f"{row['dataset']}/{row['video_id']}"


def completed_inputs(root: Path) -> tuple[dict, dict, dict, dict]:
    public_path = Path(__file__).resolve().parents[2]/"results/spatial-crop-completed-v1.json"
    public = read_json(public_path)
    bindings = public["receipt_sha256"]
    records = {name: read_json(root/name, sha) for name, sha in bindings.items()}
    plan, result = records["plan.json"], records["result.json"]
    verification, completion = records["verification.json"], records["completion.json"]
    if (public["completed"] is not True or plan["study"] != "spatial_crop_execution_v1"
            or any(records[name]["completed"] is not True for name in (
                "aggregate.json", "verification.json", "supervision.json", "completion.json", "terminal.json"))
            or result["plan_sha256"] != bindings["plan.json"]
            or verification["plan_sha256"] != bindings["plan.json"]
            or verification["result_sha256"] != bindings["result.json"]
            or verification["checkpoint_replays"] != 24
            or verification["operating_cases"] != 144 or len(result["cases"]) != 144
            or verification["conditional_intervals"] != 360 or len(result["conditional_intervals"]) != 360
            or completion["aggregate_sha256"] != bindings["aggregate.json"]
            or completion["verification_sha256"] != bindings["verification.json"]
            or completion["supervision_sha256"] != bindings["supervision.json"]
            or records["terminal.json"]["completion_sha256"] != bindings["completion.json"]):
        raise ValueError("completed24-fit receipt chain required, not historical20-fit inputs")
    implementation_root = Path(__file__).resolve().parents[2]
    for relative, sha in plan["code_sha256"].items():
        if digest(implementation_root/relative) != sha:
            raise ValueError(f"saved scientific implementation changed: {relative}")
    saved = read_json(root/f"context-{TARGET}.json", result["artifacts"][TARGET])
    return plan, result, saved, {**bindings, f"context-{TARGET}.json": result["artifacts"][TARGET],
                                 "publication.json": digest(public_path)}


def taxonomy(rows: list[dict], fas_root: Path) -> tuple[dict, dict]:
    from fas import casia, msu, oulu, siwmv2

    base = fas_root.parent/"FAS-private/artifacts/phase1"
    inventories, bindings = {}, {}
    for dataset, stem, report, adapter in (
        ("MSU-MFSD", "msu_metadata_inventory_v1", "msu-adapter-inventory-v1", msu),
        (TARGET, "siwmv2_metadata_inventory_v1", "siwmv2-adapter-inventory-v1", siwmv2),
    ):
        public_path = fas_root/f"results/phase1/{report}.json"
        public = read_json(public_path)
        path = base/f"{stem}.json"
        value = read_json(path, public["inventory_sha256"])
        if (value["dataset"] != dataset or value["label_mapping"] != adapter.LABEL_MAPPING
                or value["label_mapping_hash"] != adapter.LABEL_MAPPING_HASH):
            raise ValueError("inventory taxonomy differs from pinned metadata adapter")
        indexed = {row["video_id"]: row for row in value["videos"]}
        if len(indexed) != len(value["videos"]):
            raise ValueError("unique original taxonomy identities required")
        inventories[dataset] = indexed
        bindings[path.name] = public["inventory_sha256"]
        bindings[public_path.name] = digest(public_path)
    population_path = base/"siwmv2_protocol_i_intersection_v1.json"
    membership = siwmv2.load_population(population_path.read_bytes())
    members = {row["video_id"]: row for row in membership["eligible_videos"]}
    if len(members) != len(membership["eligible_videos"]):
        raise ValueError("unique frozen SiW membership required")
    bindings[population_path.name] = siwmv2.MEMBERSHIP_SHA256
    result = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset == "OULU-NPU":
            metadata = oulu.parse_video_id(row["video_id"])
        elif dataset == "CASIA-FASD":
            metadata = casia.parse_media_path(row["video_id"]+".avi")
        elif dataset in inventories:
            metadata = inventories[dataset][row["video_id"]]
            if dataset == TARGET:
                parsed = siwmv2.parse_media_path(metadata["media_relpath"], members[row["video_id"]])
                if any(metadata[field] != value for field, value in parsed.items()):
                    raise ValueError("SiW inventory differs from hash-bound protocol membership")
        else:
            raise ValueError("unsupported source taxonomy dataset")
        fields = ("dataset", "video_id", "binary_label", "subject_id")
        if any(metadata[field] != row[field] for field in fields) or (
                dataset != "OULU-NPU" and metadata["official_split"] != row["official_split"]):
            raise ValueError("taxonomy/accepted manifest class, subject or split join changed")
        attack = row["binary_label"] == "attack"
        result[identity(row)] = {
            "family": metadata["attack_family"] if attack else None,
            "type": (metadata.get("reference_attack_type") or metadata["instrument"]) if attack else None,
        }
    if len(result) != len(rows):
        raise ValueError("complete unique taxonomy/manifest join required")
    return result, bindings


def joined_scores(context: dict, saved: dict, indexed: dict, system: str, role: str) -> dict[str, float]:
    expected = [key for key in context[role] if indexed[key]["detector_status"] == "success"]
    keys = saved["keys"][role]
    scores = np.asarray(saved["models"][system]["probabilities"][role], dtype=float)
    if (saved["target"] != context["target"] or keys != expected or len(set(keys)) != len(keys)
            or scores.shape != (len(keys),) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
            or (role == "evaluation" and not np.array_equal(scores, saved["prediction"][system]))):
        raise ValueError("exact ordered finite saved-score/usable identity join required")
    return dict(zip(keys, scores.tolist(), strict=True))


def coverage_status(family: str | None, fitting_families: set, unknown_fitting: int) -> str:
    if family is None:
        return "unknown_taxonomy"
    if family in fitting_families:
        return "source_family_covered"
    return "source_coverage_unknown" if unknown_fitting else "source_family_unseen"


def check_evaluation_groups(plan: dict, components: dict) -> None:
    expected = {key: components[key] for context in plan["contexts"] for key in context["evaluation"]}
    if expected != plan["groups"]:
        raise ValueError("saved target-bootstrap exact-frame/group components changed")


def attack_slice(attacks: list[dict], reals: list[dict], scores: dict,
                 threshold: float, alpha: Fraction) -> dict[str, Any]:
    from .spatial_diagnosis import grouped_oracle

    if (not attacks or not reals or any(r["binary_label"] != "attack" for r in attacks)
            or any(r["binary_label"] != "bona_fide" for r in reals)
            or not np.isfinite(threshold) or not isinstance(alpha, Fraction) or not 0 <= alpha < 1):
        raise ValueError("positive attack slice, shared bona fide population and valid policy required")
    usable = [r for r in attacks+reals if r["detector_status"] == "success"]
    keys = [identity(r) for r in usable]
    if len(set(identity(r) for r in attacks+reals)) != len(attacks)+len(reals):
        raise ValueError("unique slice originals required")
    values = np.asarray([scores[key] for key in keys], dtype=float)
    if (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
            or any(identity(r) in scores for r in attacks+reals if r["detector_status"] != "success")):
        raise ValueError("failures must have no scores; usable probabilities must be finite")
    labels = np.asarray([r["binary_label"] == "attack" for r in usable], dtype=int)
    n_usable = int(labels.sum())
    accepted = int(((values < threshold) & (labels == 1)).sum())
    oracle = grouped_oracle(values, labels, len(attacks), len(reals), alpha)
    return {
        "original_attack": len(attacks), "usable_attack": n_usable,
        "excluded_attack": len(attacks)-n_usable, "accepted_attack": accepted,
        "FA_original": accepted/len(attacks),
        "FA_common": accepted/n_usable if n_usable else None,
        "common_FA_defined": n_usable > 0,
        "empirical_error_budget": len(attacks)*alpha.numerator//alpha.denominator,
        "zero_error_budget": len(attacks)*alpha.numerator//alpha.denominator == 0,
        "fewer_than20_original_attacks": len(attacks) < 20,
        "discrimination_evidence_available": n_usable > 0 and len(usable) > n_usable,
        "original_oracle": oracle,
        "oracle_bona_fide_population": "all_target_bona_fide_shared_across_slices",
        "oracle_original_bona_fide": len(reals),
    }


def role_audit(rows: list[dict], roles: dict, guard) -> list[dict]:
    from .regional_study import transaction_for

    media = {}
    for row in rows:
        guard()
        media[identity(row)] = (transaction_for(row).get("media_sha256")
                               if row["detector_status"] == "success" else None)
    result = []
    for dataset in sorted({r["dataset"] for r in rows}):
        populations = {role: [r for r in rows if r["dataset"] == dataset and roles[identity(r)] == role]
                       for role in ROLES}
        fields = {
            "video": lambda r: r["video_id"], "subject": lambda r: r["subject_id"],
            "group": lambda r: r["group_id"],
            "selected_RGB": lambda r: r["selected_frame"]["rgb_sha256"] if r["selected_frame"] else None,
            "full_media": lambda r: media[identity(r)],
        }
        sets = {name: {role: {value for r in pool if (value := getter(r))}
                       for role, pool in populations.items()} for name, getter in fields.items()}
        crossings = {
            name: {f"{left}:{right}": len(pools[left] & pools[right])
                   for i, left in enumerate(ROLES) for right in ROLES[i+1:]}
            for name, pools in sets.items()
        }
        result.append({
            "dataset": dataset, "role_originals": {k: len(v) for k, v in populations.items()},
            "unknown_subject_originals": sum(not r["subject_id"] for pool in populations.values() for r in pool),
            "missing_full_media_hash_originals": sum(not media[identity(r)] for pool in populations.values() for r in pool),
            "missing_selected_RGB_hash_originals": sum(not r["selected_frame"] for pool in populations.values() for r in pool),
            "cross_role_shared_identifier_counts": crossings,
            "subject_isolation_checkable": all(r["subject_id"] for pool in populations.values() for r in pool),
            "known_identifier_isolation_passed": not any(count for pairs in crossings.values() for count in pairs.values()),
            "not_a_near_duplicate_or_cross_dataset_identity_audit": True,
        })
    return result


def run(root: Path, output: Path) -> dict:
    from .clip_verify import independent_points, independent_threshold
    from .regional_study import inputs
    from .spatial_crop_study import SYSTEMS, population
    from .spatial_diagnosis import check_operating_point, grouped_oracle

    require_private_output(output/"summary.json")
    if output.exists():
        raise FileExistsError("new immutable SiW diagnosis output root required")
    started = time.monotonic()

    def guard() -> None:
        if time.monotonic()-started > BUDGET["max_seconds"]-30:
            raise TimeoutError("SiW diagnosis closure reserve")

    plan, result, saved, bindings = completed_inputs(root)
    services, raw, partitions = inputs(Path(plan["fas_root"]))
    if [c for c in partitions["contexts"] if c["seed"] == plan["contexts"][0]["seed"]] != plan["contexts"]:
        raise ValueError("accepted original roles changed")
    indexed = population(plan, raw, services)
    mapped, metadata_bindings = taxonomy(raw, Path(plan["fas_root"]))
    preparation = services["prepare_paper319"]
    metadata_bindings[preparation.MANIFEST.name] = preparation.MANIFEST_SHA
    metadata_bindings["partitions.json"] = digest(preparation.OUTPUT/"partitions.json")
    roles, components = services["prepare_paper319"].assign_roles(raw)
    check_evaluation_groups(plan, components)
    context = next(c for c in plan["contexts"] if c["target"] == TARGET)
    if set(saved["models"]) != set(SYSTEMS) or set(saved["policies"]) != set(SYSTEMS):
        raise ValueError("exact six completed crop systems required")
    target = [indexed[key] for key in context["evaluation"]]
    attacks = [r for r in target if r["binary_label"] == "attack"]
    reals = [r for r in target if r["binary_label"] == "bona_fide"]
    if (len(target), len(attacks), len(reals), sum(r["detector_status"] == "success" for r in target)) != (623, 362, 261, 501):
        raise ValueError("unchanged full SiW target and common denominator required")
    fitting_attacks = [key for key in context["fitting"]
                       if indexed[key]["binary_label"] == "attack" and indexed[key]["detector_status"] == "success"]
    families = {mapped[key]["family"] for key in fitting_attacks if mapped[key]["family"] is not None}
    unknown_fitting = sum(mapped[key]["family"] is None for key in fitting_attacks)
    slices = {}
    for dimension in ("family", "type", "coverage"):
        grouped = {}
        for row in attacks:
            metadata = mapped[identity(row)]
            value = (coverage_status(metadata["family"], families, unknown_fitting)
                     if dimension == "coverage" else metadata[dimension] or "unknown_taxonomy")
            grouped.setdefault(value, []).append(row)
        slices[dimension] = grouped
    source_coverage = []
    for role in ROLES[:-1]:
        for dataset in sorted({indexed[key]["dataset"] for key in context[role]}):
            pool = [indexed[key] for key in context[role]
                    if indexed[key]["dataset"] == dataset and indexed[key]["binary_label"] == "attack"]
            for family in sorted({mapped[identity(r)]["family"] or "unknown_taxonomy" for r in pool}):
                subset = [r for r in pool if (mapped[identity(r)]["family"] or "unknown_taxonomy") == family]
                source_coverage.append({"role": role, "dataset": dataset, "family": family,
                                        "original_attack": len(subset),
                                        "usable_attack": sum(r["detector_status"] == "success" for r in subset)})
    cases = []
    previous = {(c["target"], c["arm"], c["alpha"], c["view"]): c for c in result["cases"]}
    if len(previous) != len(result["cases"]):
        raise ValueError("unique completed operating cases required")
    for system in SYSTEMS:
        scores = joined_scores(context, saved, indexed, system, "evaluation")
        source_scores = joined_scores(context, saved, indexed, system, "calibration")
        calibration = [indexed[key] for key in source_scores]
        values = np.asarray(list(scores.values()), dtype=float)
        labels = np.asarray([indexed[key]["binary_label"] == "attack" for key in scores], dtype=int)
        for alpha in ALPHAS:
            guard()
            threshold = independent_threshold(np.asarray(list(source_scores.values())), calibration, alpha)
            if threshold != saved["policies"][system][str(alpha)]["threshold"]:
                raise ValueError("existing source-selected threshold changed")
            policies = {}
            for view, pool in (("original", target),
                               ("common", [r for r in target if r["detector_status"] == "success"])):
                policy = independent_points(values, pool, threshold, alpha)
                check_operating_point(policy, previous[TARGET, system, str(alpha), view]["policy"])
                policies[view] = policy
            oracle = grouped_oracle(values, labels, len(attacks), len(reals), alpha)
            old_oracle = previous[TARGET, system, str(alpha), "original"]["oracle"]
            for fresh, old in (("accepted_attack", "oracle_accepted_attack"),
                               ("accepted_bona_fide", "oracle_accepted_bona_fide"),
                               ("FA", "oracle_FA"), ("max_BFA", "oracle_max_BFA")):
                if oracle[fresh] != old_oracle[old]:
                    raise ValueError("full-target oracle differs from completed study")
            groups = {dimension: {name: attack_slice(pool, reals, scores, threshold, alpha)
                                  for name, pool in sorted(grouped.items())}
                      for dimension, grouped in slices.items()}
            for grouped in groups.values():
                if (sum(g["original_attack"] for g in grouped.values()) != len(attacks)
                        or sum(g["usable_attack"] for g in grouped.values()) != int(labels.sum())
                        or sum(g["accepted_attack"] for g in grouped.values()) != policies["original"]["accepted_attack"]):
                    raise ValueError("attack slices do not reconstruct original/common/global acceptance")
            source_points = {}
            for dataset in sorted({r["dataset"] for r in calibration}):
                pool = [r for r in calibration if r["dataset"] == dataset]
                source_points[dataset] = independent_points(
                    np.asarray([source_scores[identity(r)] for r in pool]), pool, threshold, alpha)
            cases.append({"system": system, "alpha": str(alpha), "policy": policies,
                          "full_original_oracle": oracle, "groups": groups,
                          "source_calibration_policy": source_points})
    entries = {e["key"]: e for e in plan["entries"]}
    failure_counts = Counter(
        e["historical_status"] if e["historical_status"] != "usable" else "support_exclusion"
        for key, e in entries.items() if key in context["evaluation"] and e["status"] != "usable")
    joined = {key: {**mapped[key], "role": roles[key], "usable": indexed[key]["detector_status"] == "success"}
              for key in sorted(indexed)}
    audits = role_audit(raw, roles, guard)
    write_json(output/"joins.json", joined)
    summary = {
        "completed": True, "scope": "saved_scores_and_bound_metadata_only",
        "receipt_sha256": bindings, "metadata_sha256": metadata_bindings, "budget": BUDGET,
        "diagnostic_code_sha256": {
            relative: digest(Path(__file__).resolve().parents[2]/relative)
            for relative in ("src/diffusionfas/siw_attack_diagnosis.py", "scripts/siw_attack_diagnosis.py")
        },
        "target": TARGET, "target_originals": len(target), "target_common": len(values),
        "original_attack": len(attacks), "original_bona_fide": len(reals),
        "unknown_target_attack_taxonomy": sum(mapped[identity(r)]["family"] is None for r in attacks),
        "unknown_usable_fitting_attack_taxonomy": unknown_fitting,
        "usable_fitting_families": sorted(families), "source_coverage": source_coverage,
        "target_shared_failures": dict(sorted(failure_counts.items())),
        "attack_type_to_family": {mapped[identity(r)]["type"]: mapped[identity(r)]["family"] for r in attacks},
        "role_audit": audits, "cases": cases, "operating_cases_reconstructed": len(cases)*2,
        "joins_sha256": digest(output/"joins.json"), "joins_content_sha256": object_digest(joined),
        "new_model_loads": 0, "new_inference": 0, "new_fits": 0, "new_calibrator_fits": 0,
        "image_video_decodes": 0, "new_bootstraps": 0,
        "full_target_is_primary": True, "subsets_are_diagnostic_only": True,
        "source_family_coverage_is_not_exact_PAI_instrument_coverage": True,
        "project_family_mapping_is_not_provider_ontology": True,
        "group_oracles_share_all_target_bona_fide_and_use_target_labels": True,
        "small_groups_and_zero_error_budgets_not_certification_evidence": True,
        "target_exposed_single_training_seed_no_causal_claim": True,
    }
    if len(cases) != 18:
        raise ValueError("exact six-system/three-alpha diagnostic cases required")
    guard()
    write_json(output/"summary.json", summary)
    return summary
