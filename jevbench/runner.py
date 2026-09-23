"""Batch evaluation runner: load gold JSONL → predict → metrics → results JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import metrics as M
from .adapters import NetworkError, make_adapter
from .config import (
    CONCURRENCY,
    DATASET_ORDER,
    ECE_BINS,
    ENABLE_THINKING,
    MAX_TOKENS_LLM,
    MODELS,
    NETWORK_RETRIES,
    OOS_TAU,
    REASONING_EFFORT,
    SEED,
    TEMPERATURE,
)
from .schemas import Prediction, Sample

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_samples(name: str, limit: int | None = None) -> list[Sample]:
    path = DATASETS / f"{name}.jsonl"
    rows: list[Sample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(Sample(**d))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def evaluate_dataset(
    samples: list[Sample],
    preds: list[Prediction],
    task_type: str,
    labels: list[str] | None = None,
    oos_tau: float = OOS_TAU,
) -> dict[str, Any]:
    """Aggregate metrics over valid predictions (format_invalid excluded from quality)."""
    valid = [p for p in preds if p.format_valid and p.error is None]
    n_all = len(preds)
    n_valid = len(valid)
    gold_by = {s.id: s for s in samples}

    quality: dict[str, Any] = {}
    if task_type == "choice":
        choice_labels = labels if labels is not None else samples[0].labels
        has_oos_option = any(str(label).lower() == "oos" for label in choice_labels)
        y_true, y_pred = [], []
        for p in valid:
            s = gold_by[p.id]
            y_true.append(s.label)
            # Confidence-based abstention is defined only when "oos" is an allowed option.
            probs = p.probabilities or {}
            peak_p = max(probs.values()) if probs else 0.0
            pred_label = p.answer
            if has_oos_option and peak_p < oos_tau:
                pred_label = "oos"
            y_pred.append(pred_label)
        quality["acc"] = M.accuracy(y_true, y_pred)
        quality["macro_f1"] = M.macro_f1(y_true, y_pred)
        # OOS subset metrics
        oos_idx = [i for i, p in enumerate(valid) if str(gold_by[p.id].label).lower() == "oos" or gold_by[p.id].meta.get("is_oos")]
        if oos_idx:
            oos_correct = sum(1 for i in oos_idx if y_pred[i] == "oos")
            quality["oos_recall"] = oos_correct / len(oos_idx)
        in_idx = [i for i, p in enumerate(valid) if not (str(gold_by[p.id].label).lower() == "oos" or gold_by[p.id].meta.get("is_oos"))]
        if in_idx:
            # in-scope accuracy with abstain (pred oos) counted wrong
            quality["in_scope_acc"] = sum(1 for i in in_idx if y_pred[i] == y_true[i]) / len(in_idx)
            quality["abstain_rate_in_scope"] = sum(1 for i in in_idx if y_pred[i] == "oos") / len(in_idx)

    elif task_type == "score":
        gold, pred_exp, pred_exact = [], [], []
        for p in valid:
            s = gold_by[p.id]
            gold.append(float(s.label))
            probs = p.probabilities or {}
            exp = sum(int(k) * float(v) for k, v in probs.items()) if probs else float(p.answer)
            pred_exp.append(float(p.answer if p.answer is not None else exp))
            pred_exact.append(float(max(probs, key=lambda k: probs[k]) if probs else round(pred_exp[-1])))
        quality["mae"] = M.score_mae(gold, pred_exp)
        quality["exact_acc"] = M.accuracy([int(round(g)) for g in gold], [int(round(x)) for x in pred_exact])
        quality["within_1"] = M.within_k_acc(gold, pred_exp, 1.0)

    else:  # noul
        y, ps = [], []
        for p in valid:
            s = gold_by[p.id]
            y.append(bool(s.label))
            ps.append(float(p.probabilities.get("true", 0.5)) if p.probabilities else 0.5)
        b = M.binary_metrics(y, ps, threshold=0.5)
        quality.update(b)

    # calibration on valid preds
    confs, correct_flags = [], []
    for p in valid:
        s = gold_by[p.id]
        if task_type == "choice":
            conf = float(p.confidence if p.confidence is not None else max(p.probabilities.values()))
            peak = max(p.probabilities, key=lambda k: p.probabilities[k]) if p.probabilities else None
            # correct = peak matches gold (before tau abstain)? DESIGN: use whether answer is correct
            correct_flags.append(p.answer == s.label)
            confs.append(conf)
        elif task_type == "score":
            conf = float(p.confidence if p.confidence is not None else max(p.probabilities.values()))
            pred_level = int(round(float(p.answer)))
            correct_flags.append(pred_level == int(s.label))
            confs.append(conf)
        else:
            conf = float(p.confidence if p.confidence is not None else abs(float(p.probabilities.get("true", 0.5)) - 0.5) * 2)
            pred = bool(p.answer) if p.answer is not None else float(p.probabilities.get("true", 0)) >= 0.5
            correct_flags.append(pred == bool(s.label))
            confs.append(conf)

    if task_type in {"choice"} and valid:
        label_set = labels or sorted({str(gold_by[p.id].label) for p in valid} | {k for p in valid for k in p.probabilities})
        brier = M.brier_multiclass(
            [p.probabilities for p in valid],
            [str(gold_by[p.id].label) for p in valid],
            label_set,
        )
    elif task_type == "score" and valid:
        level_set = [str(i) for i in (samples[0].levels or [0, 1, 2, 3, 4])]
        gold_levels = [str(int(gold_by[p.id].label)) for p in valid]
        brier = M.brier_multiclass([p.probabilities for p in valid], gold_levels, level_set)
    elif valid:
        brier = M.brier_binary(
            [float(p.probabilities.get("true", 0.5)) for p in valid],
            [bool(gold_by[p.id].label) for p in valid],
        )
    else:
        brier = float("nan")

    ece_out = M.ece(confs, correct_flags, n_bins=ECE_BINS)

    lats = [p.latency_ms for p in preds if p.latency_ms is not None]
    tin = sum(p.tokens_in or 0 for p in preds)
    tout = sum(p.tokens_out or 0 for p in preds)

    return {
        "n": n_all,
        "n_valid": n_valid,
        "format_valid_pct": (n_valid / n_all * 100.0) if n_all else 0.0,
        "quality": quality,
        "calibration": {
            "brier": brier,
            "ece": ece_out["ece"],
            "mce": ece_out["mce"],
            "bins": ece_out["bins"],
        },
        "latency_ms": {
            "p50": M.percentile(lats, 50),
            "p95": M.percentile(lats, 95),
            "mean": (sum(lats) / len(lats)) if lats else float("nan"),
        },
        "tokens": {"in_total": tin, "out_total": tout, "in_avg": tin / n_all if n_all else 0, "out_avg": tout / n_all if n_all else 0},
    }


def run_one_model_dataset(
    model_id: str,
    dataset: str,
    limit: int | None,
    jev_endpoint: str,
    concurrency: int,
) -> dict[str, Any]:
    samples = load_samples(dataset, limit=limit)
    if not samples:
        raise RuntimeError(f"no samples for {dataset}")
    task_type = samples[0].task_type
    labels = list(samples[0].labels) if samples[0].labels else None
    spec = MODELS[model_id]
    adapter = make_adapter(spec, endpoint=jev_endpoint)
    preds: list[Prediction | None] = [None] * len(samples)

    def work(i: int, s: Sample) -> tuple[int, Prediction]:
        try:
            return i, adapter.predict(s).prediction
        except NetworkError as e:
            return i, Prediction(
                id=s.id,
                dataset=s.dataset,
                model=model_id,
                task_type=s.task_type,
                answer=None,
                format_valid=False,
                label_in_set=False,
                error=f"network: {e}",
            )
        except Exception as e:  # noqa: BLE001
            return i, Prediction(
                id=s.id,
                dataset=s.dataset,
                model=model_id,
                task_type=s.task_type,
                answer=None,
                format_valid=False,
                label_in_set=False,
                error=f"error: {e}",
            )

    t0 = time.perf_counter()
    done = 0
    n_total = len(samples)
    raw_path = RESULTS / "raw" / f"{dataset}__{model_id}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # incremental checkpoint: keep id → row so restarts can resume
    cache: dict[str, dict] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cache[d["id"]] = d

    def flush_raw() -> None:
        with raw_path.open("w", encoding="utf-8") as f:
            for s in samples:
                if s.id in cache:
                    f.write(json.dumps(cache[s.id], ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = []
        for i, s in enumerate(samples):
            if s.id in cache and cache[s.id].get("format_valid") and not cache[s.id].get("error"):
                preds[i] = Prediction(**{k: v for k, v in cache[s.id].items() if k in Prediction.__dataclass_fields__})
                done += 1
            else:
                futs.append(ex.submit(work, i, s))
        if done:
            print(f"    resume cache {done}/{n_total} valid rows", flush=True)
        for fut in as_completed(futs):
            i, p = fut.result()
            preds[i] = p
            d = p.to_dict()
            d.pop("raw", None)
            cache[p.id] = d
            done += 1
            if done == 1 or done % 5 == 0 or done == n_total:
                elapsed = time.perf_counter() - t0
                rate = (done) / elapsed if elapsed > 0 else 0.0
                remaining = n_total - done
                eta = remaining / rate if rate > 0 else float("nan")
                flush_raw()
                print(
                    f"    [{dataset} × {model_id}] {done}/{n_total} "
                    f"({100.0 * done / n_total:.0f}%) {rate:.2f} it/s ETA {eta:.0f}s",
                    flush=True,
                )
    wall = time.perf_counter() - t0
    flush_raw()

    assert all(p is not None for p in preds)
    pred_list = [p for p in preds if p is not None]
    summary = evaluate_dataset(samples, pred_list, task_type=task_type, labels=labels)

    pin, pout = spec.price_in_per_m, spec.price_out_per_m
    cost = M.cost_usd(summary["tokens"]["in_total"], summary["tokens"]["out_total"], pin, pout)
    # Prefer provider-reported cost when present on predictions (OpenRouter usage.cost)
    provider_costs = [p.raw.get("usage", {}).get("cost") for p in pred_list if isinstance(p.raw, dict)]
    provider_costs = [c for c in provider_costs if isinstance(c, (int, float))]
    if provider_costs and len(provider_costs) == len(pred_list):
        cost = float(sum(provider_costs))
        cost_source = "provider_usage.cost"
    elif pin == 0.0 and pout == 0.0:
        cost = None
        cost_source = "unavailable"
    else:
        cost_source = "tokens_x_list_price"
    n = summary["n"] or 1
    summary["cost"] = {
        "total_usd": cost,
        "usd_per_1k": cost / n * 1000.0 if cost is not None else None,
        "price_in_per_m": pin,
        "price_out_per_m": pout,
        "source": cost_source,
    }
    summary["wall_seconds"] = wall
    summary["model"] = {
        "id": spec.id,
        "openrouter_id": spec.openrouter_id,
        "kind": spec.kind,
        "display": spec.display,
        "family": spec.family,
    }
    summary["dataset"] = dataset
    summary["task_type"] = task_type
    summary["oos_tau"] = OOS_TAU

    # raw already checkpointed to raw_path
    summary["raw_path"] = str(raw_path.relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="JEV-Benchmark runner")
    ap.add_argument("--models", nargs="+", default=["jev"], help="model ids (see jevbench.config.MODELS)")
    ap.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    ap.add_argument("--limit", type=int, default=None, help="max samples per dataset (smoke)")
    ap.add_argument("--jev-endpoint", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--tag", default=None, help="label for this run in summary.json")
    args = ap.parse_args(argv)

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        print(f"unknown models: {unknown}. known: {list(MODELS)}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or (f"limit{args.limit}" if args.limit else "full")
    archive = RESULTS / "runs" / f"{run_id}_{tag}"
    archive.mkdir(parents=True, exist_ok=True)
    source_files = ["jevbench/config.py", "jevbench/prompts.py", "jevbench/adapters.py", "jevbench/runner.py", "jevbench/schemas.py"]
    manifest = {
        "run_id": run_id,
        "tag": tag,
        "status": "running",
        "command_argv": sys.argv,
        "python": platform.python_version(),
        "models": {m: {"provider_model_id": MODELS[m].openrouter_id, "kind": MODELS[m].kind} for m in args.models},
        "datasets": args.datasets,
        "limit": args.limit,
        "jev_endpoint": args.jev_endpoint,
        "settings": {"seed": SEED, "temperature": TEMPERATURE, "enable_thinking": ENABLE_THINKING,
                     "max_tokens_llm": MAX_TOKENS_LLM, "network_retries": NETWORK_RETRIES,
                     "oos_tau": OOS_TAU, "concurrency": args.concurrency,
                     "openrouter_reasoning_effort": {m: ("none" if m == "qwen3.5-397b" else REASONING_EFFORT)
                                                      for m in args.models if MODELS[m].kind == "llm"}},
        "source_sha256": {name: _sha256(ROOT / name) for name in source_files},
        "gold_sha256": {ds: _sha256(DATASETS / f"{ds}.jsonl") for ds in args.datasets},
        "raw_snapshots": {},
    }
    manifest_path = archive / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_rows: list[dict[str, Any]] = []
    print(f"JEV-Benchmark run {run_id} tag={tag}")
    for ds in args.datasets:
        for mid in args.models:
            print(f"→ {ds} × {mid} …", flush=True)
            try:
                summary = run_one_model_dataset(mid, ds, args.limit, args.jev_endpoint, args.concurrency)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {ds}×{mid}: {e}", file=sys.stderr)
                all_rows.append({"dataset": ds, "model": mid, "error": str(e)})
                continue
            q = summary.get("quality", {})
            cost_value = summary["cost"]["usd_per_1k"]
            cost_text = f"{cost_value:.4f}" if cost_value is not None else "n/a"
            print(
                "  "
                f"valid={summary['format_valid_pct']:.1f}% "
                f"acc={q.get('acc', q.get('exact_acc', float('nan'))):.3f} "
                f"brier={summary['calibration']['brier']:.3f} "
                f"ece={summary['calibration']['ece']:.3f} "
                f"p50={summary['latency_ms']['p50']:.0f}ms "
                f"$/1k={cost_text}",
                flush=True,
            )
            all_rows.append(summary)
            raw = ROOT / summary["raw_path"]
            snapshot = archive / "raw" / raw.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw, snapshot)
            manifest["raw_snapshots"][f"{ds}__{mid}"] = {
                "path": str(snapshot.relative_to(ROOT)), "sha256": _sha256(snapshot)
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out = {
        "run_id": run_id,
        "tag": tag,
        "limit": args.limit,
        "seed": SEED,
        "oos_tau": OOS_TAU,
        "models": args.models,
        "datasets": args.datasets,
        "jev_endpoint": args.jev_endpoint,
        "results": all_rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"summary_{run_id}_{tag}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_path, archive / "summary.json")
    manifest["status"] = "complete" if all("error" not in row for row in all_rows) else "partial"
    manifest["summary_sha256"] = _sha256(out_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # also write latest
    (RESULTS / "summary_latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
