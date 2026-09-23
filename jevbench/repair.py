"""Re-predict failed slots (network / 429 / format-invalid) and rewrite raw JSONL."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .adapters import NetworkError, make_adapter
from .config import MODELS
from .runner import load_samples
from .schemas import Prediction

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def load_raw(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def needs_retry(row: dict) -> bool:
    if row.get("format_valid") and not row.get("error"):
        return False
    err = (row.get("error") or "").lower()
    # always retry network/429; also retry pure format failures once
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--jev-endpoint", default="openrouter")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args(argv)

    mid, ds = args.model, args.dataset
    spec = MODELS[mid]
    raw_path = RAW / f"{ds}__{mid}.jsonl"
    rows = load_raw(raw_path)
    samples = load_samples(ds)
    by_id = {s.id: s for s in samples}

    todo = [i for i, s in enumerate(samples) if s.id not in rows or needs_retry(rows[s.id])]
    print(f"{ds} × {mid}: total={len(samples)} cached={len(rows)} retry={len(todo)}")
    if not todo:
        print("nothing to retry")
        return 0

    adapter = make_adapter(spec, endpoint=args.jev_endpoint)
    ok = fail = 0
    for n, i in enumerate(todo, 1):
        s = samples[i]
        try:
            pred = adapter.predict(s).prediction
            d = pred.to_dict()
            d.pop("raw", None)
            rows[s.id] = d
            ok += 1
            status = "ok" if d.get("format_valid") else f"fmt:{d.get('error')}"
        except NetworkError as e:
            fail += 1
            status = f"net:{e}"
            rows[s.id] = {
                "id": s.id,
                "dataset": s.dataset,
                "model": mid,
                "task_type": s.task_type,
                "answer": None,
                "probabilities": {},
                "confidence": None,
                "format_valid": False,
                "label_in_set": False,
                "error": f"network: {e}",
            }
        except Exception as e:  # noqa: BLE001
            fail += 1
            status = f"err:{e}"
            rows[s.id] = {
                "id": s.id,
                "dataset": s.dataset,
                "model": mid,
                "task_type": s.task_type,
                "answer": None,
                "probabilities": {},
                "confidence": None,
                "format_valid": False,
                "label_in_set": False,
                "error": f"error: {e}",
            }
        if n % 5 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {status}  ok={ok} fail={fail}", flush=True)
            # checkpoint
            with raw_path.open("w", encoding="utf-8") as f:
                for s in samples:
                    if s.id in rows:
                        f.write(json.dumps(rows[s.id], ensure_ascii=False) + "\n")
        time.sleep(args.sleep)

    with raw_path.open("w", encoding="utf-8") as f:
        for s in samples:
            if s.id in rows:
                f.write(json.dumps(rows[s.id], ensure_ascii=False) + "\n")
    n_valid = sum(1 for s in samples if rows.get(s.id, {}).get("format_valid"))
    print(f"wrote {raw_path} valid={n_valid}/{len(samples)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
