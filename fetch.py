"""Download public gold-label datasets and freeze them as JSONL.

Working sources (network-tested 2026-09):
- jsDelivr GitHub CDN (raw.githubusercontent.com blocked)
- hf-mirror.com (huggingface.co timed out)
- pyarrow from isolated PYTHONPATH for parquet (Tsinghua wheel)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
SEED = 42
USER_AGENT = "JEV-Benchmark/0.1 (academic eval)"
PYARROW_PATH = "/tmp/jev_pydeps"


def _load_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
        return pq
    except ImportError:
        # Compatibility with the original local data-freeze environment.
        if PYARROW_PATH not in sys.path:
            sys.path.insert(0, PYARROW_PATH)
        import pyarrow.parquet as pq  # type: ignore
        return pq


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(
    rows: list[dict[str, Any]], n: int, label_key: str = "label", seed: int = SEED
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r[label_key], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by.keys(), key=lambda k: (-len(by[k]), str(k)))
    total = len(rows)
    quota: dict[Any, int] = {}
    assigned = 0
    for k in keys:
        q = max(1, round(n * len(by[k]) / total))
        quota[k] = min(q, len(by[k]))
        assigned += quota[k]
    while assigned > n:
        k = max(quota, key=lambda x: quota[x])
        if quota[k] > 1:
            quota[k] -= 1
            assigned -= 1
        else:
            break
    while assigned < n:
        k = max(keys, key=lambda x: len(by[x]) - quota.get(x, 0))
        if quota.get(k, 0) < len(by[k]):
            quota[k] = quota.get(k, 0) + 1
            assigned += 1
        else:
            break
    picked: list[dict[str, Any]] = []
    for k in keys:
        picked.extend(by[k][: quota.get(k, 0)])
    rng.shuffle(picked)
    return picked[:n]


def freeze(name: str, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    jl = DATASETS / f"{name}.jsonl"
    write_jsonl(jl, rows)
    meta = {
        **meta,
        "name": name,
        "n": len(rows),
        "seed": SEED,
        "sha256": sha256_file(jl),
        "path": str(jl.relative_to(ROOT)),
        "schema": "jevbench.schemas.Sample v1",
    }
    (DATASETS / f"{name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  wrote {jl.name} n={len(rows)} sha256={meta['sha256'][:12]}…")


def _attach_choice_vocab(rows: list[dict[str, Any]], descriptions: dict[str, str] | None = None) -> list[str]:
    labels = sorted({r["label"] for r in rows})
    desc = descriptions or {lab: str(lab).replace("_", " ") for lab in labels}
    desc = {lab: desc.get(lab, str(lab).replace("_", " ")) for lab in labels}
    for r in rows:
        r["labels"] = labels
        r["label_descriptions"] = desc
    return labels


# ---------------------------------------------------------------------------


def build_banking77(n: int = 800) -> None:
    print("== banking77 ==")
    url = "https://hf-mirror.com/datasets/mteb/banking77/resolve/main/test.jsonl"
    raw = http_get(url).decode("utf-8")
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(raw.strip().splitlines()):
        obj = json.loads(line)
        # mteb format: text, label (int), label_text
        lab = obj.get("label_text") or obj.get("intent") or obj.get("label")
        text = obj.get("text") or obj.get("sentence") or obj.get("query")
        rows.append(
            {
                "id": f"banking77-test-{i:05d}",
                "dataset": "banking77",
                "split": "test",
                "text": text,
                "lang": "en",
                "task_type": "choice",
                "label": str(lab),
                "meta": {
                    "source_row": i,
                    "source_url": url,
                    "license": "CC BY 4.0",
                    "citation": "casanueva-etal-2020-efficient",
                },
            }
        )
    labels = _attach_choice_vocab(rows)
    sample = stratified_sample(rows, n, "label")
    freeze(
        "banking77",
        sample,
        {
            "citation": (
                "Casanueva et al. 2020. Efficient Intent Detection with Dual Sentence Encoders. EMNLP 2020."
            ),
            "license": "CC BY 4.0",
            "task_type": "choice",
            "n_labels": len(labels),
            "labels": labels,
            "official_split": "test",
            "source_url": url,
            "sample_strategy": "stratified by label, seed=42",
            "requested_n": n,
        },
    )


def build_agnews(n: int = 500) -> None:
    print("== agnews ==")
    url = "https://cdn.jsdelivr.net/gh/mhjabreel/CharCnn_Keras@master/data/ag_news_csv/test.csv"
    raw = http_get(url).decode("utf-8", errors="replace")
    label_map = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(raw.strip().splitlines()):
        parts = next(csv.reader([line]))
        if len(parts) < 2:
            continue
        lab_raw = parts[0].strip().strip('"')
        if lab_raw not in label_map:
            continue  # skip header
        title = parts[1].strip()
        desc = parts[2].strip() if len(parts) > 2 else ""
        body = f"{title}\n{desc}".strip() if desc else title
        rows.append(
            {
                "id": f"agnews-test-{i:05d}",
                "dataset": "agnews",
                "split": "test",
                "text": body,
                "lang": "en",
                "task_type": "choice",
                "label": label_map[lab_raw],
                "meta": {
                    "source_row": i,
                    "source_url": url,
                    "title": title,
                    "license": "AG News original terms / research use",
                    "citation": "zhang-etal-2015-character-cnn",
                },
            }
        )
    desc = {
        "World": "International / world news",
        "Sports": "Sports results and coverage",
        "Business": "Business, markets, finance",
        "Sci/Tech": "Science and technology",
    }
    labels = _attach_choice_vocab(rows, desc)
    sample = stratified_sample(rows, n, "label")
    freeze(
        "agnews",
        sample,
        {
            "citation": "Zhang et al. 2015. Character-level Convolutional Networks for Text Classification. NeurIPS 2015.",
            "license": "see source repository",
            "task_type": "choice",
            "n_labels": len(labels),
            "labels": labels,
            "official_split": "test",
            "source_url": url,
            "sample_strategy": "stratified by label, seed=42",
            "requested_n": n,
        },
    )


def build_sst5(n: int = 500) -> None:
    print("== sst5 ==")
    url = "https://hf-mirror.com/datasets/SetFit/sst5/resolve/main/test.jsonl"
    raw = http_get(url).decode("utf-8")
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(raw.strip().splitlines()):
        obj = json.loads(line)
        lab = int(obj["label"])
        rows.append(
            {
                "id": f"sst5-test-{i:05d}",
                "dataset": "sst5",
                "split": "test",
                "text": obj["text"],
                "lang": "en",
                "task_type": "score",
                "label": lab,
                "meta": {
                    "source_row": i,
                    "source_url": url,
                    "label_text": obj.get("label_text"),
                    "citation": "socher-etal-2013-recursive",
                    "license": "see source",
                },
            }
        )
    level_descriptions = {
        "0": "very negative",
        "1": "negative",
        "2": "neutral",
        "3": "positive",
        "4": "very positive",
    }
    for r in rows:
        r["levels"] = [0, 1, 2, 3, 4]
        r["level_descriptions"] = level_descriptions
    sample = stratified_sample(rows, n, "label")
    freeze(
        "sst5",
        sample,
        {
            "citation": (
                "Socher et al. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP 2013."
            ),
            "license": "see source",
            "task_type": "score",
            "n_labels": 5,
            "levels": [0, 1, 2, 3, 4],
            "level_descriptions": level_descriptions,
            "official_split": "test",
            "source_url": url,
            "sample_strategy": "stratified by level, seed=42",
            "requested_n": n,
        },
    )


def build_boolq(n: int = 500) -> None:
    print("== boolq ==")
    url = "https://hf-mirror.com/datasets/google/boolq/resolve/main/data/validation-00000-of-00001.parquet"
    blob = http_get(url, timeout=90)
    tmp = Path("/tmp/boolq_val_download.parquet")
    tmp.write_bytes(blob)
    pq = _load_pyarrow()
    table = pq.read_table(tmp)
    cols = table.to_pydict()
    rows: list[dict[str, Any]] = []
    for i in range(table.num_rows):
        q = cols["question"][i]
        p = cols["passage"][i]
        ans = bool(cols["answer"][i])
        rows.append(
            {
                "id": f"boolq-val-{i:05d}",
                "dataset": "boolq",
                "split": "validation",
                "text": f"passage: {p}\nquestion: {q}",
                "lang": "en",
                "task_type": "noul",
                "label": ans,
                "meta": {
                    "source_row": i,
                    "source_url": url,
                    "question": q,
                    "passage": p,
                    "citation": "clark-etal-2019-boolq",
                    "license": "CC BY 4.0",
                },
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    sample = rows[:n]
    freeze(
        "boolq",
        sample,
        {
            "citation": (
                "Clark et al. 2019. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. NAACL 2019."
            ),
            "license": "CC BY 4.0",
            "task_type": "noul",
            "official_split": "validation",
            "source_url": url,
            "sample_strategy": "random shuffle seed=42, take n",
            "requested_n": n,
        },
    )


def build_clinc150(n: int = 600, oos_frac: float = 0.15) -> None:
    print("== clinc150 ==")
    url = "https://cdn.jsdelivr.net/gh/clinc/oos-eval@master/data/data_full.json"
    obj = json.loads(http_get(url, timeout=90).decode("utf-8"))
    test = obj.get("test") or []
    oos_test = obj.get("oos_test") or []
    rows: list[dict[str, Any]] = []
    labels_set: set[str] = set()
    for i, item in enumerate(test):
        text, lab = item[0], str(item[1])
        labels_set.add(lab)
        rows.append(
            {
                "id": f"clinc150-test-{i:05d}",
                "dataset": "clinc150",
                "split": "test",
                "text": text,
                "lang": "en",
                "task_type": "choice",
                "label": lab,
                "meta": {
                    "source_row": i,
                    "is_oos": False,
                    "source_url": url,
                    "citation": "larson-etal-2019-clinc",
                    "license": "CC BY 4.0",
                },
            }
        )
    labels_set.add("oos")
    base = len(rows)
    for i, item in enumerate(oos_test):
        text = item[0] if isinstance(item, (list, tuple)) else str(item)
        rows.append(
            {
                "id": f"clinc150-oos-{i:05d}",
                "dataset": "clinc150",
                "split": "test",
                "text": text,
                "lang": "en",
                "task_type": "choice",
                "label": "oos",
                "meta": {
                    "source_row": base + i,
                    "is_oos": True,
                    "source_url": url,
                    "citation": "larson-etal-2019-clinc",
                    "license": "CC BY 4.0",
                },
            }
        )
    desc = {lab: lab.replace("_", " ") for lab in labels_set}
    desc["oos"] = "out of scope / none of the defined intents"
    labels = _attach_choice_vocab(rows, desc)

    rng = random.Random(SEED)
    n_oos = max(1, round(n * oos_frac))
    n_in = n - n_oos
    in_rows = [r for r in rows if r["label"] != "oos"]
    oos_rows = [r for r in rows if r["label"] == "oos"]
    rng.shuffle(oos_rows)
    sample = stratified_sample(in_rows, n_in, "label") + oos_rows[:n_oos]
    rng.shuffle(sample)
    freeze(
        "clinc150",
        sample,
        {
            "citation": (
                "Larson et al. 2019. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. "
                "EMNLP-IJCNLP 2019."
            ),
            "license": "CC BY 4.0",
            "task_type": "choice",
            "n_labels": len(labels),
            "labels": labels,
            "official_split": "test + oos_test",
            "source_url": url,
            "sample_strategy": f"stratified in-scope + fixed oos_frac={oos_frac}, seed=42",
            "requested_n": n,
            "oos_frac": oos_frac,
        },
    )


BUILDERS: dict[str, Callable[..., None]] = {
    "banking77": build_banking77,
    "agnews": build_agnews,
    "sst5": build_sst5,
    "boolq": build_boolq,
    "clinc150": build_clinc150,
}

PLAN: dict[str, Callable[[], None]] = {
    "banking77": lambda: build_banking77(300),
    "clinc150": lambda: build_clinc150(200, 0.15),
    "sst5": lambda: build_sst5(200),
    "boolq": lambda: build_boolq(200),
    "agnews": lambda: build_agnews(150),
}


def main(argv: list[str]) -> int:
    which = argv[1:] or list(PLAN.keys())
    print(f"JEV-Benchmark freeze gold sets → {DATASETS}")
    ok: list[str] = []
    fail: list[tuple[str, str]] = []
    for name in which:
        try:
            PLAN[name]()
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}", file=sys.stderr)
            fail.append((name, str(e)))
    print(f"\nOK={ok}")
    if fail:
        print(f"FAIL={fail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
