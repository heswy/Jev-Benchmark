"""Paired, end-to-end comparison of saved predictions against frozen gold data.

The primary score is the unweighted mean of the five task accuracies. For SST-5,
accuracy means the most probable level is exactly right. Failed predictions count
as wrong. All intervals resample paired items within each dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATASET_ORDER, OOS_TAU

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
PRIMARY = "macro_mean_end_to_end_accuracy"
PLANNED_UPPER = ("qwen3.8-27b",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def correctness(gold: dict[str, Any], pred: dict[str, Any] | None) -> int:
    if not pred or not pred.get("format_valid") or pred.get("error"):
        return 0
    task = gold["task_type"]
    probs = pred.get("probabilities") or {}
    if task == "choice":
        answer = pred.get("answer")
        has_oos_option = any(str(label).lower() == "oos" for label in gold.get("labels", []))
        if has_oos_option and probs and max(float(v) for v in probs.values()) < OOS_TAU:
            answer = "oos"
        return int(answer == gold["label"])
    if task == "score":
        if not probs:
            return 0
        return int(max(probs, key=lambda k: probs[k]) == str(gold["label"]))
    if task == "noul":
        return int((float(probs.get("true", 0.5)) >= 0.5) == bool(gold["label"]))
    raise ValueError(f"unknown task type {task}")


def paired_interval(a: list[int], b: list[int], rng: random.Random, draws: int) -> list[float]:
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    estimates = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def macro_interval(pairs: list[tuple[list[int], list[int]]], rng: random.Random, draws: int) -> list[float]:
    datasets = [[y - x for x, y in zip(a, b)] for a, b in pairs]
    estimates = []
    for _ in range(draws):
        estimates.append(sum(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for ds in datasets) / len(datasets))
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def mcnemar_exact(a: list[int], b: list[int]) -> dict[str, Any]:
    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2**n)
        p = min(1.0, 2 * tail)
    return {"reference_only_correct": a_only, "candidate_only_correct": b_only, "two_sided_p": p}


def permutation_p(pairs: list[tuple[list[int], list[int]]], rng: random.Random, draws: int) -> dict[str, float]:
    datasets = [[y - x for x, y in zip(a, b)] for a, b in pairs]
    observed = sum(sum(ds) / len(ds) for ds in datasets) / len(datasets)
    # Under the paired null, swapping the two model names on any item is exchangeable.
    greater = less = farther = 0
    for _ in range(draws):
        permuted = sum(sum(d if rng.randrange(2) else -d for d in ds) / len(ds) for ds in datasets) / len(datasets)
        greater += permuted >= observed - 1e-12
        less += permuted <= observed + 1e-12
        farther += abs(permuted) >= abs(observed) - 1e-12
    return {
        "candidate_superior_one_sided": (greater + 1) / (draws + 1),
        "jev_superior_one_sided": (less + 1) / (draws + 1),
        "two_sided": (farther + 1) / (draws + 1),
    }


def validated_predictions(dataset: str, model: str, gold: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = RAW / f"{dataset}__{model}.jsonl"
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    expected = {row["id"] for row in gold}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"{path}: duplicate, missing, or extra sample IDs")
    if any(row.get("dataset") != dataset or row.get("model") != model for row in rows):
        raise ValueError(f"{path}: incorrect dataset/model identity")
    invalid = [row for row in rows if not row.get("format_valid") or row.get("error")]
    reasons: dict[str, int] = {}
    for row in invalid:
        reason = str(row.get("error") or "format_invalid").split(":", 1)[0]
        reasons[reason] = reasons.get(reason, 0) + 1
    return {row["id"]: row for row in rows}, {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "n": len(rows), "n_valid": len(rows) - len(invalid), "invalid_reasons": reasons,
    }


def build_report(models: list[str], draws: int = 5000, seed: int = 42) -> dict[str, Any]:
    golds: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, Any] = {}
    scores: dict[str, dict[str, list[int]]] = {}
    for ds in DATASET_ORDER:
        path = ROOT / "datasets" / f"{ds}.jsonl"
        gold = read_jsonl(path)
        if len({row["id"] for row in gold}) != len(gold):
            raise ValueError(f"duplicate gold IDs in {path}")
        meta = json.loads((ROOT / "datasets" / f"{ds}.meta.json").read_text(encoding="utf-8"))
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"gold checksum mismatch: {path}")
        golds[ds] = gold
        sources[ds] = {"gold_path": str(path.relative_to(ROOT)), "gold_sha256": sha256(path), "n": len(gold), "raw": {}}
        scores[ds] = {}
        for model in ["jev", *models]:
            preds, info = validated_predictions(ds, model, gold)
            sources[ds]["raw"][model] = info
            scores[ds][model] = [correctness(row, preds[row["id"]]) for row in gold]

    comparisons = []
    for model in models:
        rng = random.Random(f"{seed}:{model}")
        by_dataset = []
        pairs = []
        for ds in DATASET_ORDER:
            ref, cand = scores[ds]["jev"], scores[ds][model]
            pairs.append((ref, cand))
            n = len(ref)
            by_dataset.append({
                "dataset": ds, "n": n, "jev_correct": sum(ref), "candidate_correct": sum(cand),
                "jev_accuracy": sum(ref) / n, "candidate_accuracy": sum(cand) / n,
                "difference_candidate_minus_jev": (sum(cand) - sum(ref)) / n,
                "paired_bootstrap_95_ci": paired_interval(ref, cand, rng, draws),
                "mcnemar": mcnemar_exact(ref, cand),
            })
        delta = sum(row["difference_candidate_minus_jev"] for row in by_dataset) / len(by_dataset)
        p = permutation_p(pairs, rng, draws)
        comparisons.append({
            "candidate": model,
            "jev_macro_accuracy": sum(row["jev_accuracy"] for row in by_dataset) / len(by_dataset),
            "candidate_macro_accuracy": sum(row["candidate_accuracy"] for row in by_dataset) / len(by_dataset),
            "difference_candidate_minus_jev": delta,
            "paired_stratified_bootstrap_95_ci": macro_interval(pairs, rng, draws),
            "permutation_p": p,
            "planned_upper_family_bonferroni_p": min(1.0, p["candidate_superior_one_sided"] * len(PLANNED_UPPER)) if model in PLANNED_UPPER else None,
            "by_dataset": by_dataset,
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "primary": PRIMARY, "reference": "jev", "dataset_order": DATASET_ORDER,
            "failed_prediction": "incorrect", "choice_oos_tau": OOS_TAU,
            "score_accuracy": "argmax of level probabilities", "noul_threshold": 0.5,
            "bootstrap_draws": draws, "permutation_draws": draws, "seed": seed,
            "analysis_source_sha256": sha256(Path(__file__)),
            "config_source_sha256": sha256(ROOT / "jevbench" / "config.py"),
            "upper_bound_candidates_planned": PLANNED_UPPER,
            "upper_bound_decision": "For the completed Qwen3.8-27B comparison, candidate macro difference > 0 and one-sided paired permutation p < 0.05. Other planned runs were cancelled before complete results.",
        },
        "sources": sources, "comparisons": comparisons,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b", "qwen3.5-122b"])
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/statistics.json")
    args = ap.parse_args()
    if args.draws < 100:
        ap.error("--draws must be >= 100")
    report = build_report(args.models, args.draws, args.seed)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for comp in report["comparisons"]:
        print(f"{comp['candidate']}: macro delta={comp['difference_candidate_minus_jev']:+.4f} CI={comp['paired_stratified_bootstrap_95_ci']} p(two-sided)={comp['permutation_p']['two_sided']:.4g}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
