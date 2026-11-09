"""Metadata-only coverage matrix and non-executable held-out-family inventories."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

from .io import digest, object_digest, read_json, require_private_output, write_json
from .siw_attack_diagnosis import BUDGET, ROLES, TARGET, check_evaluation_groups, identity


def validate_records(records: list[dict]) -> None:
    if not records or len({r["key"] for r in records}) != len(records):
        raise ValueError("nonempty unique original identities required")
    components = defaultdict(set)
    for row in records:
        if (row["role"] not in ROLES or row["label"] not in {"attack", "bona_fide"}
                or type(row["usable"]) is not bool or not row["component"]
                or (row["label"] == "bona_fide" and row["family"] is not None)
                or (row["family"] is not None and not isinstance(row["family"], str))
                or (row["role"] == "evaluation") != (row["official_split"] == "test")):
            raise ValueError("registered role/class/family/eligibility and original test boundary required")
        components[row["component"]].add((row["dataset"], row["role"]))
    if any(len(values) != 1 for values in components.values()):
        raise ValueError("component crosses an accepted role or dataset boundary")


def counts(pool: list[dict]) -> dict:
    usable = [r for r in pool if r["usable"]]
    return {
        "originals": len(pool), "usable": len(usable), "excluded": len(pool)-len(usable),
        "known_subjects": len({r["subject_id"] for r in pool if r["subject_id"]}),
        "usable_known_subjects": len({r["subject_id"] for r in usable if r["subject_id"]}),
        "unknown_subject_originals": sum(not r["subject_id"] for r in pool),
        "unknown_subject_usable": sum(not r["subject_id"] for r in usable),
        "components": len({r["component"] for r in pool}),
        "usable_components": len({r["component"] for r in usable}),
    }


def coverage_matrix(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        family = (row["family"] or "unknown_taxonomy") if row["label"] == "attack" else "not_applicable"
        grouped[row["dataset"], row["role"], row["label"], family].append(row)
    return [
        {"dataset": dataset, "role": role, "label": label, "family": family, **counts(pool)}
        for (dataset, role, label, family), pool in sorted(grouped.items())
    ]


def domain_feasibility(records: list[dict]) -> list[dict]:
    result = []
    for target in sorted({r["dataset"] for r in records}):
        fitting = [r for r in records if r["dataset"] != target and r["role"] == "fitting"
                   and r["label"] == "attack" and r["usable"]]
        known = {r["family"] for r in fitting if r["family"] is not None}
        unknown = sum(r["family"] is None for r in fitting)
        target_attacks = [r for r in records if r["dataset"] == target
                          and r["role"] == "evaluation" and r["label"] == "attack"]
        families = []
        for family in sorted({r["family"] or "unknown_taxonomy" for r in target_attacks}):
            pool = [r for r in target_attacks if (r["family"] or "unknown_taxonomy") == family]
            families.append({
                "family": family, **counts(pool),
                "usable_fitting_sources": sorted({r["dataset"] for r in fitting if r["family"] == family}),
                "coverage": ("unknown_taxonomy" if family == "unknown_taxonomy" else
                             "source_family_covered" if family in known else
                             "source_coverage_unknown" if unknown else "source_family_unseen"),
            })
        result.append({"target": target, "usable_fitting_families": sorted(known),
                       "unknown_fitting_attack_taxonomy": unknown, "families": families})
    return result


def family_candidates(records: list[dict]) -> tuple[list[dict], list[dict]]:
    selected = [r for r in records if r["dataset"] == TARGET]
    if not selected:
        raise ValueError("SiW metadata required for within-domain candidate preparation")
    indexed = {r["key"]: r for r in selected}
    families = sorted({r["family"] for r in selected if r["label"] == "attack" and r["family"] is not None})
    summaries, inventories = [], []
    for family in families:
        heldout_components = {r["component"] for r in selected if r["label"] == "attack" and r["family"] == family}
        pools = {}
        for role in ROLES:
            pool = [r for r in selected if r["role"] == role]
            kept = [r for r in pool if role == "evaluation" or (
                r["component"] not in heldout_components
                and (r["family"] is not None or r["label"] == "bona_fide"))]
            pools[role] = {
                "original": sorted(r["key"] for r in kept),
                "usable": sorted(r["key"] for r in kept if r["usable"]),
            }
        added = sorted(r["key"] for r in selected if r["role"] == "fitting"
                       and r["label"] == "attack" and r["family"] == family and r["usable"])
        collateral = [r for r in selected if r["role"] != "evaluation"
                      and r["component"] in heldout_components
                      and not (r["label"] == "attack" and r["family"] == family)]
        unknown = [r for r in selected if r["role"] != "evaluation" and r["label"] == "attack"
                   and r["family"] is None]
        baseline_attacks = [k for k in pools["fitting"]["usable"] if indexed[k]["label"] == "attack"]
        class_support = {
            role: {label: sum(indexed[k]["label"] == label for k in pools[role]["usable"])
                   for label in ("attack", "bona_fide")} for role in ROLES
        }
        blockers = [
            "SiW_subject_identity_unverified",
            "single_domain_sampler_not_implemented_pinned_sampler_requires_three_sources",
            "equal_cardinality_component_preserving_replacement_manifest_not_sealed",
            "fixed_paired_optimizer_step_schedule_and_training_budget_not_sealed",
            "engineering_probe_checkpoint_replay_and_new_owner_training_approval_required",
            "development_exposed_target_not_unbiased_final_test",
        ]
        if collateral:
            blockers.append("heldout_component_contains_other_family_or_bona_fide")
        if unknown:
            blockers.append("unknown_training_taxonomy_cannot_certify_family_absence")
        if not added or not baseline_attacks or any(not n for v in class_support.values() for n in v.values()):
            blockers.append("missing_usable_class_or_heldout_fitting_support")
        target_attacks = [r for r in selected if r["role"] == "evaluation"
                          and r["label"] == "attack" and r["family"] == family]
        inventory = {
            "heldout_family": family, "baseline_roles": pools,
            "comparator_heldout_fitting_attack_eligible": added,
            "baseline_fitting_attack_eligible": sorted(baseline_attacks),
            "component_closed_removed_non_evaluation": sorted(
                r["key"] for r in selected if r["role"] != "evaluation" and r["component"] in heldout_components),
            "collateral_removed": sorted(r["key"] for r in collateral),
            "unknown_taxonomy_excluded_non_evaluation": sorted(r["key"] for r in unknown),
            "comparator_is_eligible_pool_not_an_additive_training_arm": True,
            "executable": False, "training_authorized": False,
        }
        inventories.append(inventory)
        summaries.append({
            "heldout_family": family, "estimand": "within_SiW_family_coverage_development_diagnostic",
            "unseen_domain_claim_permitted": False, "subject_disjoint_claim_permitted": False,
            "role_support": {role: counts([indexed[k] for k in v["original"]]) for role, v in pools.items()},
            "usable_class_support": class_support,
            "heldout_fitting_attack_eligible": len(added),
            "baseline_fitting_attack_eligible": len(baseline_attacks),
            "maximum_possible_replacements_before_component_matching": min(len(added), len(baseline_attacks)),
            "target_heldout_attack": counts(target_attacks),
            "target5pct_integer_FA_budget": len(target_attacks)*5//100,
            "collateral_removed_originals": len(collateral),
            "unknown_taxonomy_excluded_non_evaluation": len(unknown),
            "inventory_sha256": object_digest(inventory),
            "all_target_originals_retained": len(pools["evaluation"]["original"]) == sum(
                r["role"] == "evaluation" for r in selected),
            "selection_and_calibration_heldout_family_absent": all(
                indexed[k]["family"] != family for role in ("validation", "calibration")
                for k in pools[role]["original"]),
            "blockers": blockers, "execution_ready": False, "training_authorized": False,
        })
    return summaries, inventories


def run(diagnosis_root: Path, completed_root: Path, output: Path) -> dict:
    from .regional_study import inputs

    require_private_output(output/"summary.json")
    if output.exists():
        raise FileExistsError("new immutable coverage-protocol root required")
    begun = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    public_path = repo/"results/siw-attack-diagnosis-v2.json"
    public = read_json(public_path)
    previous = read_json(diagnosis_root/"summary.json", public["verification"]["summary_sha256"])
    verification = read_json(diagnosis_root/"independent-verification.json",
                             public["independent_verification_sha256"])
    if (previous["completed"] is not True or verification["completed"] is not True
            or verification["summary_sha256"] != public["verification"]["summary_sha256"]
            or verification["taxonomy_and_role_joins"] != 7505):
        raise ValueError("completed independently verified metadata diagnosis required")
    joins = read_json(diagnosis_root/"joins.json", previous["joins_sha256"])
    for relative, sha in previous["receipt_sha256"].items():
        path = repo/"results/spatial-crop-completed-v1.json" if relative == "publication.json" else completed_root/relative
        if digest(path) != sha:
            raise ValueError("completed scientific receipt or saved-context binding changed")
    plan = read_json(completed_root/"plan.json", previous["receipt_sha256"]["plan.json"])
    services, raw, partitions = inputs(Path(plan["fas_root"]))
    roles, components = services["prepare_paper319"].assign_roles(raw)
    check_evaluation_groups(plan, components)
    if [c for c in partitions["contexts"] if c["seed"] == plan["contexts"][0]["seed"]] != plan["contexts"]:
        raise ValueError("original partitions changed")
    entries = {e["key"]: e for e in plan["entries"]}
    if set(joins) != set(entries) or set(roles) != set(joins):
        raise ValueError("complete common cohort lineage required")
    records = []
    for row in raw:
        key = identity(row)
        if (joins[key]["role"] != roles[key] or joins[key]["usable"] != (entries[key]["status"] == "usable")
                or row["binary_label"] != entries[key]["label"]):
            raise ValueError("taxonomy/role/common-mask join changed")
        records.append({
            "key": key, "dataset": row["dataset"], "label": row["binary_label"],
            "official_split": row["official_split"], "role": roles[key],
            "subject_id": row["subject_id"], "component": components[key],
            "family": joins[key]["family"], "type": joins[key]["type"],
            "usable": joins[key]["usable"],
        })
    validate_records(records)
    matrix = coverage_matrix(records)
    candidates, inventories = family_candidates(records)
    if (len(records), sum(r["usable"] for r in records), len(candidates)) != (7505, 6231, 5):
        raise ValueError("exact7505 originals,6231 usable and five family candidates required")
    if time.monotonic()-begun > BUDGET["max_seconds"]-30:
        raise TimeoutError("coverage preparation closure reserve")
    write_json(output/"records.json", records)
    write_json(output/"inventories.json", inventories)
    summary = {
        "completed": True, "scope": "metadata_only_non_executable_protocol_preparation",
        "budget": BUDGET, "originals": len(records), "usable": sum(r["usable"] for r in records),
        "excluded": sum(not r["usable"] for r in records), "matrix": matrix,
        "domain_feasibility": domain_feasibility(records), "candidates": candidates,
        "origin": {
            "diagnosis_publication_sha256": digest(public_path),
            "diagnosis_summary_sha256": digest(diagnosis_root/"summary.json"),
            "diagnosis_joins_sha256": digest(diagnosis_root/"joins.json"),
            "diagnosis_independent_verification_sha256": digest(diagnosis_root/"independent-verification.json"),
            "completed_plan_sha256": digest(completed_root/"plan.json"),
        },
        "diagnostic_code_sha256": {relative: digest(repo/relative) for relative in (
            "src/diffusionfas/coverage_protocol.py", "scripts/coverage_protocol.py")},
        "records_sha256": digest(output/"records.json"),
        "inventories_sha256": digest(output/"inventories.json"),
        "new_scores_or_predictions_read": 0, "new_model_loads": 0, "new_inference": 0,
        "image_video_decodes": 0, "new_fits": 0, "new_calibrator_fits": 0,
        "historical_role_assignment_changed": False, "training_authorized": False,
        "proposed_system": "legacy_content", "proposed_crop": "legacy",
        "architecture_objective_augmentation_recipe_changes_proposed": False,
        "equal_number_of_fitting_attack_originals_required_before_execution": True,
        "equal_bona_fide_validation_calibration_and_evaluation_pools_required": True,
        "equal_paired_optimizer_steps_and_initialization_required": True,
        "five_candidates_are_not_ten_authorized_fits": True,
        "independence_audit_inherited_not_repeated": public["role_audit"],
    }
    write_json(output/"summary.json", summary)
    return summary
