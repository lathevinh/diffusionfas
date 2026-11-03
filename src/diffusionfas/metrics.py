"""Integer operating points, source thresholds and grouped uncertainty."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Prediction:
    key: str
    dataset: str
    group: str
    partition: str
    label: str
    score: float | None
    failure: str | None = None
    attack_family: str | None = None

    def __post_init__(self) -> None:
        if self.label not in {"bona_fide", "attack"} or not self.key or not self.group:
            raise ValueError("prediction identity, label and group required")
        if (self.score is None) != bool(self.failure):
            raise ValueError("failed predictions require reason and no fabricated score")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("finite score required")


def _check_calibration(rows: list[Prediction], target: str, rate: float) -> list[str]:
    if not rows or not 0 < rate < 1:
        raise ValueError("nonempty calibration and rate inside (0,1) required")
    if any(r.dataset == target or r.partition != "calibration" for r in rows):
        raise ValueError("calibration is source-only and role-isolated")
    if len({r.key for r in rows}) != len(rows):
        raise ValueError("unique calibration originals required")
    return sorted({r.dataset for r in rows})


def strict_threshold(rows: list[Prediction], target: str, bpcer: float = 0.05) -> float:
    sources = _check_calibration(rows, target, bpcer)
    if any(r.label != "bona_fide" for r in rows):
        raise ValueError("strict one-class calibration cannot access attack scores")
    thresholds = []
    for source in sources:
        subset = [r for r in rows if r.dataset == source]
        scores = sorted(r.score for r in subset if r.score is not None)
        allowed_rejections = len(subset) * Fraction(str(bpcer))
        budget = allowed_rejections.numerator // allowed_rejections.denominator
        failures = len(subset) - len(scores)
        if len(subset) * bpcer < 1:
            raise ValueError("insufficient finite-sample support for requested BPCER")
        if failures > budget or not scores:
            raise ValueError("technical failures make requested source BPCER infeasible")
        accepted_count = len(subset) - budget
        threshold = float(np.nextafter(scores[accepted_count - 1], np.inf))
        if not math.isfinite(threshold):
            raise ValueError("no finite strict acceptance threshold for extreme scores")
        thresholds.append(threshold)
    return max(thresholds)


def attack_threshold(rows: list[Prediction], target: str, fa: float = 0.05) -> float:
    sources = _check_calibration(rows, target, fa)
    thresholds = []
    for source in sources:
        subset = [r for r in rows if r.dataset == source and r.label == "attack"]
        scores = sorted(r.score for r in subset if r.score is not None)
        if not subset or len(subset) * fa < 1:
            raise ValueError("insufficient attack support for requested FA")
        # Failures do not enlarge the successful-score acceptance budget.
        fraction = len(scores) * Fraction(str(fa))
        budget = fraction.numerator // fraction.denominator
        if len(scores) <= budget:
            raise ValueError("insufficient usable attack scores")
        thresholds.append(float(scores[budget]))
    return min(thresholds)


def auroc(rows: list[Prediction]) -> float | None:
    usable = [r for r in rows if r.score is not None]
    positives = [r.score for r in usable if r.label == "attack"]
    negatives = [r.score for r in usable if r.label == "bona_fide"]
    if not positives or not negatives:
        return None
    ordered = sorted(usable, key=lambda row: row.score)
    rank_sum, start = 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end].score == ordered[start].score:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(r.label == "attack" for r in ordered[start:end])
        start = end
    p, n = len(positives), len(negatives)
    return (rank_sum - p * (p + 1) / 2) / (p * n)


def auroc_summary(rows: list[Prediction]) -> dict[str, Any]:
    if not rows or len({r.key for r in rows}) != len(rows):
        raise ValueError("nonempty unique original predictions required")
    counts = {}
    for label in ("bona_fide", "attack"):
        subset = [r for r in rows if r.label == label]
        if not subset:
            raise ValueError("both original classes required")
        usable = sum(r.score is not None for r in subset)
        counts[label] = {"originals": len(subset), "usable": usable,
                         "technical_failures": len(subset) - usable}
    pairs = counts["attack"]["originals"] * counts["bona_fide"]["originals"]
    observed_pairs = counts["attack"]["usable"] * counts["bona_fide"]["usable"]
    value = auroc(rows)
    wins = value * observed_pairs if value is not None else 0
    return {"auroc_usable": value, "coverage": counts,
            "unavailable_reason": None if value is not None else "insufficient_usable_classes",
            "all_original_missing_score_bounds": [wins / pairs, (wins + pairs - observed_pairs) / pairs],
            "score_direction": "higher_is_attack"}


def paired_auroc_bootstrap(
    arms: dict[str, list[Prediction]], *, repetitions: int = 2000, seed: int = 20261103,
) -> dict[str, Any]:
    if not arms or repetitions < 1:
        raise ValueError("nonempty paired arms and positive repetitions required")
    reference = next(iter(arms.values()))
    auroc_summary(reference)
    identities = [(r.key, r.dataset, r.group, r.partition, r.label) for r in reference]
    if len({r.dataset for r in reference}) != 1:
        raise ValueError("bootstrap one domain at a time")
    for rows in arms.values():
        if [(r.key, r.dataset, r.group, r.partition, r.label) for r in rows] != identities:
            raise ValueError("paired AUROC bootstrap requires identical ordered original populations")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        groups[row.group].append(index)
    if len(groups) < 2:
        raise ValueError("at least two declared groups required")
    keys, rng = sorted(groups), np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in arms}
    invalid = 0
    for _ in range(repetitions):
        indexes = [i for key in rng.choice(keys, size=len(keys), replace=True) for i in groups[key]]
        sample = {name: auroc([rows[i] for i in indexes]) for name, rows in arms.items()}
        if any(value is None for value in sample.values()):
            invalid += 1
            continue
        for name, value in sample.items():
            if value is None:
                raise ValueError("usable bootstrap AUROC invariant violated")
            values[name].append(value)
    intervals: dict[str, Any] = {}
    if repetitions - invalid >= 0.95 * repetitions:
        arrays = {name: np.array(samples) for name, samples in values.items()}
        intervals.update({name: np.quantile(array, [0.025, 0.975]).tolist()
                          for name, array in arrays.items()})
        names = list(arrays)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                intervals[f"{right}-minus-{left}"] = np.quantile(
                    arrays[right] - arrays[left], [0.025, 0.975],
                ).tolist()
    return {"repetitions": repetitions, "valid_repetitions": repetitions - invalid,
            "invalid_usable_one_class": invalid, "declared_groups": len(groups),
            "intervals": intervals,
            "unavailable_reason": None if intervals else "insufficient_valid_bootstrap_replicates",
            "scope": "development_diagnostic_conditional_on_fixed_models"}


def operating_metrics(rows: list[Prediction], threshold: float) -> dict[str, Any]:
    if not rows or not math.isfinite(threshold):
        raise ValueError("nonempty predictions and finite threshold required")
    result: dict[str, Any] = {"auroc_usable": auroc(rows), "threshold": threshold}
    for label, name in (("attack", "attack"), ("bona_fide", "bonafide")):
        subset = [r for r in rows if r.label == label]
        if not subset:
            raise ValueError("both classes required for operating metrics")
        usable = [r for r in subset if r.score is not None]
        accepted = sum(r.score is not None and r.score < threshold for r in subset)
        result[name] = {
            "originals": len(subset), "usable": len(usable),
            "technical_failures": len(subset) - len(usable), "accepted": accepted,
            "acceptance_all_originals": accepted / len(subset),
            "acceptance_usable": accepted / len(usable) if usable else None,
        }
    attacks = [r for r in rows if r.label == "attack"]
    result["attack_families"] = {}
    for family in sorted({r.attack_family or "unknown" for r in attacks}):
        subset = [r for r in attacks if (r.attack_family or "unknown") == family]
        usable = [r for r in subset if r.score is not None]
        accepted = sum(r.score is not None and r.score < threshold for r in subset)
        result["attack_families"][family] = {
            "originals": len(subset), "usable": len(usable), "accepted": accepted,
            "FA_all_originals": accepted / len(subset),
            "APCER_usable": accepted / len(usable) if usable else None,
        }
    bfa = result["bonafide"]["acceptance_usable"]
    result["BPCER_usable"] = 1 - bfa if bfa is not None else None
    return result


def oracle_bfa(rows: list[Prediction], fa: float = 0.05) -> dict[str, Any]:
    if not 0 < fa < 1:
        raise ValueError("oracle FA must be inside (0,1)")
    attacks = sum(r.label == "attack" for r in rows)
    bonafide = sum(r.label == "bona_fide" for r in rows)
    if not attacks or not bonafide:
        raise ValueError("both original classes required for oracle diagnostic")
    fraction = attacks * Fraction(str(fa))
    budget = fraction.numerator // fraction.denominator
    groups: dict[float, list[Prediction]] = defaultdict(list)
    for row in rows:
        if row.score is not None:
            groups[row.score].append(row)
    accepted_attack, accepted_bonafide, best = 0, 0, 0
    for score in sorted(groups):
        candidate = groups[score]
        next_attack = accepted_attack + sum(r.label == "attack" for r in candidate)
        if next_attack > budget:
            break
        accepted_attack = next_attack
        accepted_bonafide += sum(r.label == "bona_fide" for r in candidate)
        best = max(best, accepted_bonafide)
    return {"BFA_all_originals": best / bonafide, "accepted_bonafide": best,
            "attack_acceptance_budget": budget, "FA_constraint": fa,
            "diagnostic_only": True, "deployable_policy": False}


def paired_bootstrap(
    arms: dict[str, list[Prediction]], thresholds: dict[str, float],
    *, repetitions: int = 2000, seed: int = 20261028,
) -> dict[str, Any]:
    if not arms or repetitions < 1 or set(arms) != set(thresholds):
        raise ValueError("aligned arms, thresholds and positive repetitions required")
    reference = next(iter(arms.values()))
    identities = [(r.key, r.dataset, r.group, r.partition, r.label) for r in reference]
    for rows in arms.values():
        if [(r.key, r.dataset, r.group, r.partition, r.label) for r in rows] != identities:
            raise ValueError("paired bootstrap requires identical ordered original populations")
    if len({r.dataset for r in reference}) != 1:
        raise ValueError("bootstrap one target at a time")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        groups[row.group].append(index)
    if len(groups) < 2:
        raise ValueError("at least two independent declared groups required")
    keys, rng = sorted(groups), np.random.default_rng(seed)
    values: dict[str, list[tuple[float, float]]] = {name: [] for name in arms}
    invalid = 0
    for _ in range(repetitions):
        indexes = [i for key in rng.choice(keys, size=len(keys), replace=True) for i in groups[key]]
        if len({reference[i].label for i in indexes}) != 2:
            invalid += 1
            continue
        for name, rows in arms.items():
            metric = operating_metrics([rows[i] for i in indexes], thresholds[name])
            values[name].append((metric["attack"]["acceptance_all_originals"],
                                 metric["bonafide"]["acceptance_all_originals"]))
    intervals: dict[str, Any] = {}
    if repetitions - invalid >= 0.95 * repetitions:
        arrays = {name: np.array(samples) for name, samples in values.items()}
        for name, array in arrays.items():
            intervals[name] = np.quantile(array, [0.025, 0.975], axis=0).tolist()
        names = list(arrays)
        for name in names[1:]:
            intervals[f"{name}-minus-{names[0]}"] = np.quantile(
                arrays[name] - arrays[names[0]], [0.025, 0.975], axis=0
            ).tolist()
    return {"repetitions": repetitions, "invalid_one_class": invalid,
            "columns": ["FA", "BFA"], "intervals": intervals,
            "scope": "conditional_on_fixed_models_and_source_thresholds"}
