"""Extract publishable billing evidence from private provider CSV exports.

Only aggregate rates and a matched Jev total are written. Account identifiers,
request identifiers and unrelated models never enter the public artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from jevbench.statistics import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "qwen3.5-9b": "qwen/qwen3.5-9b",
    "qwen3.5-27b": "qwen/qwen3.5-27b",
    "qwen3.5-35b": "qwen/qwen3.5-35b-a3b",
    "qwen3.5-122b": "qwen/qwen3.5-122b-a10b",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
}
DATASETS = ("banking77", "clinc150", "sst5", "boolq", "agnews")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal_string(value: Decimal) -> str:
    return format(value, "f")


def reconcile(siliconflow: Path, openrouter: Path, usd_cny: Decimal) -> dict:
    assert usd_cny > 0
    with siliconflow.open(encoding="utf-8-sig", newline="") as stream:
        sf_rows = list(csv.DictReader(stream))
    with openrouter.open(encoding="utf-8-sig", newline="") as stream:
        or_rows = list(csv.DictReader(stream))
    rates: dict[str, dict[str, str]] = {}
    checks: dict[str, dict] = {}
    for model, billed_name in MODELS.items():
        matched = [r for r in sf_rows if r["计费项"].startswith(billed_name + ".online.")]
        assert matched, f"No SiliconFlow rows for {model}"
        assert all(all(Decimal(r[field]) == 0 for field in
                       ("资源包抵扣用量", "折扣优惠金额", "代金券抵扣金额")) for r in matched), \
            f"Credits or discounts require a different attribution method: {model}"
        by_direction: dict[str, list[dict]] = {}
        for direction in ("input", "output"):
            rows = [r for r in matched if r["计费项"].endswith(f".{direction}-tokens")]
            assert rows and all(r["用量单位"] == "K tokens" for r in rows)
            prices = {r["计费项单价"] for r in rows}
            assert len(prices) == 1, f"Rate changed during billing window: {model}/{direction}"
            by_direction[direction] = rows
        assert len(matched) == sum(map(len, by_direction.values()))
        rates[model] = {f"{direction}_cny_per_k_tokens": next(iter({r["计费项单价"] for r in rows}))
                        for direction, rows in by_direction.items()}
        bill_tokens = {direction: sum((Decimal(r["计费项原始用量"]) * 1000 for r in rows), Decimal(0))
                       for direction, rows in by_direction.items()}
        bill_cny = sum((Decimal(r["账单金额"]) for r in matched), Decimal(0))
        checks[model] = {"billed_input_tokens": int(bill_tokens["input"]),
                         "billed_output_tokens": int(bill_tokens["output"]),
                         "account_bill_cny": decimal_string(bill_cny),
                         "billing_rows": len(matched)}
    jev_activity = [r for r in or_rows if r["model_permaslug"] == "typesafe/jev-1.13-20260917"]
    assert jev_activity
    by_pair: dict[tuple[int, int], list[Decimal]] = defaultdict(list)
    for row in jev_activity:
        assert row["cancelled"].lower() in ("", "false")
        by_pair[(int(row["tokens_prompt"]), int(row["tokens_completion"]))].append(Decimal(row["cost_total"]))
    raw_pairs = Counter()
    for dataset in DATASETS:
        for row in read_jsonl(ROOT / "results" / "raw" / f"{dataset}__jev.jsonl"):
            raw_pairs[(int(row["tokens_in"]), int(row["tokens_out"]))] += 1
    assert sum(raw_pairs.values()) == 1050
    assert all(len(by_pair[pair]) >= count for pair, count in raw_pairs.items()), "Jev activity is incomplete"
    # Repeated pairs have one price in this export. A request ID is not in the
    # benchmark raw files, so pairs with conflicting prices are not attributable.
    assert all(len(set(by_pair[pair])) == 1 for pair in raw_pairs), "Ambiguous Jev token-pair price"
    matched_cost = sum((by_pair[pair][0] * count for pair, count in raw_pairs.items()), Decimal(0))
    return {
        "schema_version": 1,
        "source": {
            "siliconflow_csv_sha256": sha256(siliconflow),
            "siliconflow_rows": len(sf_rows),
            "openrouter_csv_sha256": sha256(openrouter),
            "openrouter_rows": len(or_rows),
            "note": "Private exports are not redistributed. Hashes identify the exact source files for independent verification by their owner."
        },
        "fx": {"usd_cny": decimal_string(usd_cny), "date": "2026-09-22",
               "source": "https://www.safe.gov.cn/AppStructured/hlw/RMBQuery.do",
               "method": "PBOC/CFETS midpoint, 100 USD = CNY 674.59; analytical conversion, not a payment settlement rate"},
        "jev": {"provider": "OpenRouter", "currency": "USD", "model_permaslug": "typesafe/jev-1.13-20260917",
                "matched_calls": 1050, "jev_activity_calls": len(jev_activity),
                "unmatched_activity_calls": len(jev_activity) - 1050,
                "matched_cost_usd": decimal_string(matched_cost),
                "matching_method": "multiset of (prompt tokens, completion tokens); all 1050 raw rows covered; same cost for duplicate token pairs",
                "rounding_note": "CSV request costs have six decimal places; their sum inherits per-row rounding."},
        "qwen": {"provider": "SiliconFlow", "currency": "CNY", "unit": "K tokens",
                 "method": "benchmark raw token totals times export-observed input/output unit prices; account totals include extra calls",
                 "rates": rates, "account_checks": checks,
                 "unavailable": {"qwen3.5-4b": "No separately identifiable 4B billing item in this export"}}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siliconflow-csv", type=Path, required=True)
    parser.add_argument("--openrouter-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "billing_evidence.json")
    args = parser.parse_args()
    result = reconcile(args.siliconflow_csv, args.openrouter_csv, Decimal("6.7459"))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Matched {result['jev']['matched_calls']} Jev calls; wrote sanitized evidence to {args.out}")


if __name__ == "__main__":
    main()
