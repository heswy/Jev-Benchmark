"""Verify every published score, paired interval and p-value offline.

Uses the text-free, per-item correctness table so an open-source checkout can
audit the result without downloading third-party benchmark text or API calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

from jevbench.config import DATASET_ORDER
from jevbench.statistics import macro_interval, mcnemar_exact, paired_interval, permutation_p

ROOT = Path(__file__).resolve().parent.parent


def eq(a: object, b: object, path: str) -> None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"{path}: expected {b}, got {a}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: differing list lengths")
        for index, (x, y) in enumerate(zip(a, b)):
            eq(x, y, f"{path}[{index}]")
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: differing keys")
        for key in a:
            eq(a[key], b[key], f"{path}.{key}")
    elif a != b:
        raise AssertionError(f"{path}: expected {b}, got {a}")


def verify() -> None:
    for line in (ROOT / "results" / "ARCHIVES.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if digest != actual:
            raise AssertionError(f"release archive checksum mismatch: {relative}")
    stats = json.loads((ROOT / "results" / "statistics_with_upper.json").read_text())
    with (ROOT / "results" / "reanalysis.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1050 or len({(r["dataset"], r["id"]) for r in rows}) != 1050:
        raise AssertionError("reanalysis table must contain 1,050 unique dataset/sample pairs")
    models = ["jev", *[x["candidate"] for x in stats["comparisons"]]]
    if list(rows[0].keys()) != ["dataset", "id", *models]:
        raise AssertionError("reanalysis columns do not match statistics model order")
    if any(r[model] not in {"0", "1"} for r in rows for model in models):
        raise AssertionError("correctness values must be binary")
    by_ds = {ds: [r for r in rows if r["dataset"] == ds] for ds in DATASET_ORDER}
    if sum(map(len, by_ds.values())) != len(rows):
        raise AssertionError("unknown dataset in reanalysis table")
    for comp in stats["comparisons"]:
        model = comp["candidate"]
        rng = random.Random(f"{stats['protocol']['seed']}:{model}")
        draws = stats["protocol"]["bootstrap_draws"]
        pairs = []
        by_results = []
        for ds in DATASET_ORDER:
            dataset_rows = by_ds[ds]
            ref = [int(r["jev"]) for r in dataset_rows]
            cand = [int(r[model]) for r in dataset_rows]
            pairs.append((ref, cand))
            n = len(ref)
            by_results.append({
                "dataset": ds, "n": n, "jev_correct": sum(ref), "candidate_correct": sum(cand),
                "jev_accuracy": sum(ref)/n, "candidate_accuracy": sum(cand)/n,
                "difference_candidate_minus_jev": (sum(cand)-sum(ref))/n,
                "paired_bootstrap_95_ci": paired_interval(ref, cand, rng, draws),
                "mcnemar": mcnemar_exact(ref, cand),
            })
        eq(by_results, comp["by_dataset"], f"{model}.by_dataset")
        eq(sum(x["jev_accuracy"] for x in by_results)/5, comp["jev_macro_accuracy"], f"{model}.jev_accuracy")
        eq(sum(x["candidate_accuracy"] for x in by_results)/5, comp["candidate_macro_accuracy"], f"{model}.candidate_accuracy")
        eq(sum(x["difference_candidate_minus_jev"] for x in by_results)/5, comp["difference_candidate_minus_jev"], f"{model}.difference")
        eq(permutation_p(pairs, rng, draws), comp["permutation_p"], f"{model}.permutation")
        eq(macro_interval(pairs, rng, draws), comp["paired_stratified_bootstrap_95_ci"], f"{model}.macro_ci")
    print("Verified 1,050 paired rows, all model scores, per-dataset intervals, McNemar tests, and macro intervals/p-values.")


if __name__ == "__main__":
    verify()
