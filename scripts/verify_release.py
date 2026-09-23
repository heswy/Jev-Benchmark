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
import tarfile
from decimal import Decimal
from pathlib import Path

from jevbench.config import DATASET_ORDER
from jevbench.statistics import macro_interval, mcnemar_exact, paired_interval, permutation_p
from jevbench.metrics import percentile

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
    billing = json.loads((ROOT / "results" / "billing_evidence.json").read_text())
    efficiency = json.loads((ROOT / "results" / "efficiency.json").read_text())
    by_model = {r["model"]: r for r in efficiency["rows"]}
    if set(by_model) != set(models):
        raise AssertionError("efficiency model set differs from statistics")
    with tarfile.open(ROOT / "results" / "raw-complete.tar.gz", "r:gz") as archive:
        for model in models:
            raw = []
            for ds in DATASET_ORDER:
                member = archive.extractfile(f"results/raw/{ds}__{model}.jsonl")
                if member is None:
                    raise AssertionError(f"raw archive missing {ds}/{model}")
                raw.extend(json.loads(line) for line in member)
            if len(raw) != 1050:
                raise AssertionError(f"raw archive has {len(raw)} rows for {model}")
            published = by_model[model]
            input_tokens = sum(int(r.get("tokens_in") or 0) for r in raw)
            output_tokens = sum(int(r.get("tokens_out") or 0) for r in raw)
            latencies = [float(r["latency_ms"]) for r in raw if r.get("latency_ms") is not None]
            eq(input_tokens, published["input_tokens"], f"{model}.input_tokens")
            eq(output_tokens, published["output_tokens"], f"{model}.output_tokens")
            eq(len(latencies), published["latency_observations"], f"{model}.latency_count")
            eq(percentile(latencies, 50), published["latency_p50_ms"], f"{model}.latency_p50")
            if model == "jev":
                cost = Decimal(billing["jev"]["matched_cost_usd"]) * Decimal(billing["fx"]["usd_cny"])
                eq(input_tokens, 1222683, "jev.matched_input_tokens")
                eq(output_tokens, 523259, "jev.matched_output_tokens")
            elif model == billing["qwen"]["free_model"]["model"]:
                free = billing["qwen"]["free_model"]
                if Decimal(free["input_cny_per_k_tokens"]) != 0 or Decimal(free["output_cny_per_k_tokens"]) != 0:
                    raise AssertionError("4B published free tariff must be zero")
                cost = Decimal(0)
            elif model in billing["qwen"]["rates"]:
                rate = billing["qwen"]["rates"][model]
                cost = (Decimal(input_tokens) * Decimal(rate["input_cny_per_k_tokens"]) +
                        Decimal(output_tokens) * Decimal(rate["output_cny_per_k_tokens"])) / 1000
                check = billing["qwen"]["account_checks"][model]
                if input_tokens > check["billed_input_tokens"] or output_tokens > check["billed_output_tokens"]:
                    raise AssertionError(f"{model} benchmark use exceeds account bill")
            else:
                cost = None
            eq(float(cost) if cost is not None else None, published["benchmark_cost_cny"], f"{model}.total_cny")
            eq(float(cost * 1000 / 1050) if cost is not None else None, published["cny_per_1000_items"], f"{model}.cny_per_1000")
    print("Verified 1,050 paired rows, scores, intervals, tests, and archived token/latency/cost figures.")


if __name__ == "__main__":
    verify()
