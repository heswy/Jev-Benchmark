"""Build publication figures and a text-free paired reanalysis table.

All outputs are derived from the frozen local gold and prediction files. The
public table carries correctness bits, not third-party dataset text.
"""
from __future__ import annotations

import csv
import json
import statistics
from html import escape
from pathlib import Path

from jevbench.config import DATASET_ORDER
from jevbench.metrics import percentile
from jevbench.statistics import correctness, read_jsonl, validated_predictions

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["jev", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b", "qwen3.5-122b", "qwen3.8-27b"]
NAMES = {
    "jev": "Jev 1.13", "qwen3.5-4b": "Qwen3.5 4B", "qwen3.5-9b": "Qwen3.5 9B",
    "qwen3.5-27b": "Qwen3.5 27B", "qwen3.5-35b": "Qwen3.5 35B-A3B",
    "qwen3.5-122b": "Qwen3.5 122B-A10B", "qwen3.8-27b": "Qwen3.8 27B",
}
DATASET_NAMES = {"banking77": "BANKING77", "clinc150": "CLINC150", "sst5": "SST-5", "boolq": "BoolQ", "agnews": "AG News"}
FIGURES = ROOT / "report" / "figures"


def svg_text(x: float, y: float, value: str, **attrs: object) -> str:
    extra = " ".join(f'{("class" if key == "class_" else key.replace("_", "-"))}="{escape(str(val))}"' for key, val in attrs.items())
    return f'<text x="{x}" y="{y}" {extra}>{escape(value)}</text>'


def wrap_svg(width: int, height: int, title: str, description: str, content: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title)}</title>", f"<desc id=\"desc\">{escape(description)}</desc>",
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#1e293b} .muted{fill:#64748b}.grid{stroke:#dbe2ea;stroke-width:1}.blue{fill:#3062c4}.orange{fill:#b76027}.thin{stroke:#1e293b;stroke-width:2}</style>',
        *content, "</svg>", "",
    ])


def build_accuracy(stats: dict) -> str:
    comps = {x["candidate"]: x for x in stats["comparisons"]}
    scores = {"jev": comps["qwen3.5-4b"]["jev_macro_accuracy"] * 100}
    scores.update({key: value["candidate_macro_accuracy"] * 100 for key, value in comps.items()})
    order = sorted(MODELS, key=lambda model: scores[model], reverse=True)
    out = [svg_text(32, 41, "Five-dataset accuracy", font_size=25, font_weight=700),
           svg_text(32, 64, "End-to-end; equal dataset weights; 1,050 items per model", font_size=13, class_="muted")]
    # Absolute bars keep the zero baseline; the right plot shows paired differences.
    left, span = 196, 400
    for tick in (0, 25, 50, 75, 100):
        x = left + span * tick / 100
        out.append(f'<line x1="{x}" y1="96" x2="{x}" y2="430" class="grid"/>')
        out.append(svg_text(x, 89, f"{tick}%", font_size=11, text_anchor="middle", class_="muted"))
    out.append(svg_text(750, 89, "Difference vs Jev (percentage points), 95% paired CI", font_size=11, class_="muted"))
    diff_left, diff_span = 750, 285
    for tick in (-15, -10, -5, 0, 5):
        x = diff_left + (tick + 15) / 20 * diff_span
        out.append(f'<line x1="{x}" y1="103" x2="{x}" y2="430" class="grid"/>')
        out.append(svg_text(x, 447, f"{tick:+d}" if tick else "0", font_size=11, text_anchor="middle", class_="muted"))
    for i, model in enumerate(order):
        y = 123 + i * 47
        out.append(svg_text(32, y + 4, NAMES[model], font_size=13, font_weight=600 if model == "jev" else 400))
        color = "#3062c4" if model == "jev" else "#8694a7"
        out.append(f'<rect x="{left}" y="{y-10}" width="{span*scores[model]/100:.2f}" height="18" rx="2" fill="{color}"/>')
        out.append(svg_text(610, y + 4, f"{scores[model]:.2f}%", font_size=13, font_weight=600))
        if model == "jev":
            out.append(svg_text(750, y + 4, "reference", font_size=12, class_="muted"))
            continue
        comp = comps[model]
        delta = comp["difference_candidate_minus_jev"] * 100
        lo, hi = (v * 100 for v in comp["paired_stratified_bootstrap_95_ci"])
        x0 = diff_left + (lo + 15) / 20 * diff_span
        x1 = diff_left + (hi + 15) / 20 * diff_span
        xd = diff_left + (delta + 15) / 20 * diff_span
        out.append(f'<line x1="{x0:.2f}" y1="{y}" x2="{x1:.2f}" y2="{y}" class="thin"/>')
        out.append(f'<line x1="{x0:.2f}" y1="{y-5}" x2="{x0:.2f}" y2="{y+5}" class="thin"/>')
        out.append(f'<line x1="{x1:.2f}" y1="{y-5}" x2="{x1:.2f}" y2="{y+5}" class="thin"/>')
        out.append(f'<circle cx="{xd:.2f}" cy="{y}" r="5" fill="#b76027"/>')
    out.append(svg_text(32, 480, "Bars start at 0. Intervals are for the paired difference, not each model's standalone accuracy.", font_size=12, class_="muted"))
    return wrap_svg(1080, 502, "Benchmark accuracy and paired uncertainty", "Five-dataset equal-weight accuracy for seven models. Right panel shows each candidate minus Jev with paired 95% bootstrap intervals.", out)


def build_tasks(stats: dict) -> str:
    comps = {x["candidate"]: x for x in stats["comparisons"]}
    ref = comps["qwen3.5-4b"]["by_dataset"]
    values = {"jev": [x["jev_accuracy"] * 100 for x in ref]}
    values.update({model: [x["candidate_accuracy"] * 100 for x in comp["by_dataset"]] for model, comp in comps.items()})
    out = [svg_text(30, 41, "Accuracy by dataset", font_size=25, font_weight=700),
           svg_text(30, 65, "The five task scores behind the equal-weight overall result", font_size=13, class_="muted")]
    start, cw = 230, 156
    for j, ds in enumerate(DATASET_ORDER):
        out.append(svg_text(start + j*cw + 66, 100, DATASET_NAMES[ds], font_size=13, text_anchor="middle", font_weight=600))
    for i, model in enumerate(MODELS):
        y = 115+i*48
        out.append(svg_text(30, y+29, NAMES[model], font_size=13, font_weight=600 if model=="jev" else 400))
        for j, val in enumerate(values[model]):
            refval = values["jev"][j]
            fill = "#e8effb" if model=="jev" else "#e4f1eb" if val>refval else "#f8eee7" if val<refval else "#f0f2f5"
            x = start+j*cw
            out.append(f'<rect x="{x}" y="{y}" width="132" height="38" rx="3" fill="{fill}"/>')
            out.append(svg_text(x+66, y+25, f"{val:.1f}%", font_size=14, text_anchor="middle", font_weight=600))
    out.append(svg_text(30, 476, "Blue = Jev; pale green = above Jev on that dataset; pale orange = below Jev. Numbers are primary.", font_size=12, class_="muted"))
    return wrap_svg(1030, 498, "Per-dataset benchmark accuracy", "Direct percentages for five datasets and seven models. Colors identify whether a peer scored above or below Jev on each dataset.", out)


def build_reanalysis() -> list[dict[str, str]]:
    target = ROOT / "results" / "reanalysis.csv"
    rows: list[dict[str, str]] = []
    for ds in DATASET_ORDER:
        gold = read_jsonl(ROOT / "datasets" / f"{ds}.jsonl")
        preds = {model: validated_predictions(ds, model, gold)[0] for model in MODELS}
        for item in gold:
            row = {"dataset": ds, "id": item["id"]}
            row.update({model: str(correctness(item, preds[model][item["id"]])) for model in MODELS})
            rows.append(row)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "id", *MODELS], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_efficiency() -> list[dict]:
    summary = json.loads((ROOT / "results" / "summary_merged.json").read_text())
    rows = []
    for model in MODELS:
        lats: list[float] = []
        for ds in DATASET_ORDER:
            lats.extend(float(row["latency_ms"]) for row in read_jsonl(ROOT / "results" / "raw" / f"{ds}__{model}.jsonl") if row.get("latency_ms") is not None)
        summaries = [x for x in summary["rows"] if x["model"]["id"] == model]
        assert len(summaries) == len(DATASET_ORDER)
        costs = [x["cost"]["total_usd"] for x in summaries]
        rows.append({
            "model": model, "latency_observations": len(lats), "latency_p50_ms": percentile(lats, 50),
            "latency_p95_ms": percentile(lats, 95),
            "estimated_usd_per_1000_items": sum(costs) / 1050 * 1000 if all(x is not None for x in costs) else None,
            "cost_source": "recorded tokens × unverified historical list-price placeholders" if all(x is not None for x in costs) else "unavailable",
        })
    (ROOT / "results" / "efficiency.json").write_text(json.dumps({
        "scope": "All five datasets, 1,050 attempted items per model",
        "latency_note": "Client-observed request latency where recorded; excludes missing latency on some network failures. Provider, queue, payload and date are not controlled.",
        "cost_note": "Illustrative estimate, not an invoice or current public price. Historical price inputs were not independently archived. Qwen3.8-27B is unavailable.",
        "rows": rows,
    }, indent=2) + "\n")
    return rows


def build_latency(rows: list[dict]) -> str:
    scores = {row["model"]: row for row in rows}
    order = sorted(MODELS, key=lambda model: scores[model]["latency_p50_ms"])
    out = [svg_text(30, 41, "Observed API latency", font_size=25, font_weight=700),
           svg_text(30, 64, "Median client-observed request time across five datasets; available observations only", font_size=13, class_="muted")]
    left, span = 200, 720
    for tick in (0, 2, 4, 6):
        x = left+span*tick/6
        out.append(f'<line x1="{x}" y1="100" x2="{x}" y2="425" class="grid"/>')
        out.append(svg_text(x, 89, f"{tick} s", font_size=11, text_anchor="middle", class_="muted"))
    for i, model in enumerate(order):
        row=scores[model]; y=120+i*45; seconds=row["latency_p50_ms"]/1000
        out.append(svg_text(30, y+4, NAMES[model], font_size=13, font_weight=600 if model=="jev" else 400))
        out.append(f'<rect x="{left}" y="{y-10}" width="{span*seconds/6:.2f}" height="18" rx="2" fill="{("#3062c4" if model=="jev" else "#8694a7")}"/>')
        out.append(svg_text(935, y+4, f"{seconds:.2f} s", font_size=13, font_weight=600))
    out.append(svg_text(30, 467, "Descriptive API measurements, not intrinsic model speed; missing failure latencies are excluded.", font_size=12, class_="muted"))
    return wrap_svg(1030, 489, "Observed API latency by model", "Median client-side request latency across five datasets; source results/raw JSONL.", out)


def main() -> None:
    stats = json.loads((ROOT / "results" / "statistics_with_upper.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "accuracy.svg").write_text(build_accuracy(stats), encoding="utf-8")
    (FIGURES / "datasets.svg").write_text(build_tasks(stats), encoding="utf-8")
    rows = build_reanalysis()
    efficiency = build_efficiency()
    (FIGURES / "latency.svg").write_text(build_latency(efficiency), encoding="utf-8")
    assert len(rows) == 1050
    for comp in stats["comparisons"]:
        model = comp["candidate"]
        observed = statistics.mean(sum(int(row[model]) for row in rows if row["dataset"] == ds) / sum(row["dataset"] == ds for row in rows) for ds in DATASET_ORDER)
        assert abs(observed - comp["candidate_macro_accuracy"]) < 1e-12
    print("Built report/figures/*.svg, results/reanalysis.csv and results/efficiency.json from frozen inputs")


if __name__ == "__main__":
    main()
