"""Recompute summary metrics from results/raw/*.jsonl + datasets/*.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATASET_ORDER, MODELS, OOS_TAU
from .runner import evaluate_dataset, load_samples
from .schemas import Prediction

ROOT = Path(__file__).resolve().parent.parent


def load_preds(path: Path) -> list[Prediction]:
    allowed = Prediction.__dataclass_fields__.keys()
    out: list[Prediction] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(Prediction(**{k: v for k, v in d.items() if k in allowed}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/summary_merged.json")
    args = ap.parse_args()

    raw_dir = ROOT / "results" / "raw"
    models = sorted({p.stem.split("__", 1)[1] for p in raw_dir.glob("*.jsonl")})
    eligible: set[str] = set()
    excluded: dict[str, str] = {}
    for mid in models:
        reason = None
        for ds in DATASET_ORDER:
            path = raw_dir / f"{ds}__{mid}.jsonl"
            if not path.exists():
                reason = f"missing {ds}"
                break
            gold_ids = {s.id for s in load_samples(ds)}
            preds = load_preds(path)
            ids = [p.id for p in preds]
            if len(ids) != len(gold_ids) or len(set(ids)) != len(ids) or set(ids) != gold_ids:
                reason = f"incomplete or duplicate IDs in {ds}"
                break
            if any("HTTP 402" in (p.error or "") for p in preds):
                reason = f"provider credit failure in {ds}"
                break
        if reason:
            excluded[mid] = reason
        else:
            eligible.add(mid)

    rows = []
    for raw in sorted(raw_dir.glob("*.jsonl")):
        ds, mid = raw.stem.split("__", 1)
        if mid not in eligible:
            continue
        samples = load_samples(ds)
        preds = load_preds(raw)
        if not preds:
            continue
        task = samples[0].task_type
        labels = list(samples[0].labels) if samples[0].labels else None
        summary = evaluate_dataset(samples, preds, task_type=task, labels=labels, oos_tau=OOS_TAU)
        spec = MODELS.get(mid)
        tin, tout = summary["tokens"]["in_total"], summary["tokens"]["out_total"]
        pin = spec.price_in_per_m if spec else 0.0
        pout = spec.price_out_per_m if spec else 0.0
        cost = (tin * pin + tout * pout) / 1_000_000.0 if pin or pout else None
        n = summary["n"] or 1
        summary["cost"] = {
            "total_usd": cost,
            "usd_per_1k": cost / n * 1000.0 if cost is not None else None,
            "price_in_per_m": pin,
            "price_out_per_m": pout,
            "source": "tokens_x_list_price" if cost is not None else "unavailable",
        }
        summary["dataset"] = ds
        summary["task_type"] = task
        summary["model"] = {
            "id": mid,
            "display": spec.display if spec else mid,
            "kind": spec.kind if spec else "?",
            "family": spec.family if spec else "?",
        }
        summary["raw_path"] = str(raw.relative_to(ROOT))
        rows.append(summary)

    order_ds = DATASET_ORDER
    order_md = ["jev", "qwen3.5-4b", "qwen3.5-9b"]
    rows.sort(
        key=lambda r: (
            order_ds.index(r["dataset"]) if r["dataset"] in order_ds else 99,
            order_md.index(r["model"]["id"]) if r["model"]["id"] in order_md else 99,
        )
    )

    out_path = ROOT / args.out
    out_path.write_text(
        json.dumps(
            {
                "note": "recomputed from complete results/raw/*.jsonl + frozen gold JSONL; incomplete models excluded",
                "oos_tau": OOS_TAU,
                "excluded_models": excluded,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
