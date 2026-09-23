"""Build publication figures and a text-free paired reanalysis table.

All outputs are derived from the frozen local gold and prediction files. The
public table carries correctness bits, not third-party dataset text.
"""
from __future__ import annotations

import csv
import json
import statistics
from decimal import Decimal
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
    billing = json.loads((ROOT / "results" / "billing_evidence.json").read_text())
    assert billing["jev"]["matched_calls"] == 1050
    rows = []
    for model in MODELS:
        lats: list[float] = []
        raw = []
        for ds in DATASET_ORDER:
            raw.extend(read_jsonl(ROOT / "results" / "raw" / f"{ds}__{model}.jsonl"))
        assert len(raw) == 1050
        lats = [float(row["latency_ms"]) for row in raw if row.get("latency_ms") is not None]
        tokens_in = sum(int(row.get("tokens_in") or 0) for row in raw)
        tokens_out = sum(int(row.get("tokens_out") or 0) for row in raw)
        if model == "jev":
            total_usd = Decimal(billing["jev"]["matched_cost_usd"])
            total_cny = total_usd * Decimal(billing["fx"]["usd_cny"])
            method = "matched OpenRouter activity costs, then 2026-09-22 USD/CNY midpoint"
        elif model == billing["qwen"]["free_model"]["model"]:
            free = billing["qwen"]["free_model"]
            assert Decimal(free["input_cny_per_k_tokens"]) == Decimal(free["output_cny_per_k_tokens"]) == 0
            total_cny = Decimal(0)
            total_usd = None
            method = "SiliconFlow official free tariff; no individually identifiable 4B bill line"
        elif model in billing["qwen"]["rates"]:
            rate = billing["qwen"]["rates"][model]
            total_cny = (Decimal(tokens_in) * Decimal(rate["input_cny_per_k_tokens"]) +
                         Decimal(tokens_out) * Decimal(rate["output_cny_per_k_tokens"])) / 1000
            total_usd = None
            method = "raw tokens × observed SiliconFlow billing rates"
            check = billing["qwen"]["account_checks"][model]
            assert tokens_in <= check["billed_input_tokens"] and tokens_out <= check["billed_output_tokens"]
        else:
            raise AssertionError(f"Missing billing basis for {model}")
        rows.append({
            "model": model, "attempted_items": len(raw), "input_tokens": tokens_in, "output_tokens": tokens_out,
            "latency_observations": len(lats), "latency_p50_ms": percentile(lats, 50),
            "latency_p95_ms": percentile(lats, 95),
            "benchmark_cost_cny": float(total_cny) if total_cny is not None else None,
            "benchmark_cost_usd": float(total_usd) if total_usd is not None else None,
            "cny_per_1000_items": float(total_cny * 1000 / 1050) if total_cny is not None else None,
            "cost_method": method,
        })
    (ROOT / "results" / "efficiency.json").write_text(json.dumps({
        "scope": "All five datasets, 1,050 attempted items per model",
        "latency_note": "Client-observed request latency where recorded; excludes missing latency on some network failures. Provider, queue, payload and date are not controlled.",
        "cost_note": "Jev: matched historical OpenRouter activity charge, converted at the 2026-09-22 USD/CNY midpoint. Paid Qwen: benchmark tokens times observed SiliconFlow billing rates, excluding extra account calls. 4B: zero under SiliconFlow's officially announced free tariff, not individually matched to a named bill line. Cost excludes credit-purchase fees, taxes, and unreported overhead; figures are historical and are not current quotes.",
        "billing_evidence": "results/billing_evidence.json",
        "rows": rows,
    }, indent=2) + "\n")
    return rows


def build_score_time_cost(stats: dict, rows: list[dict]) -> str:
    scores = {"jev": stats["comparisons"][0]["jev_macro_accuracy"] * 100}
    scores.update({x["candidate"]: x["candidate_macro_accuracy"] * 100 for x in stats["comparisons"]})
    efficiency = {r["model"]: r for r in rows}
    cols = [(247, 220, "Score", "Equal-weight accuracy · %", 100),
            (520, 200, "Time", "Median API request · seconds", 6),
            (770, 210, "Cost", "CNY / 1,000 items", 3)]
    out = [svg_text(28, 39, "Score · time · cost", font_size=25, font_weight=700),
           svg_text(28, 64, "Same 1,050 attempts per model; all rows use the same task mix", font_size=13, class_="muted")]
    for left, span, title, subtitle, maximum in cols:
        out.append(svg_text(left, 105, title, font_size=16, font_weight=700))
        out.append(svg_text(left, 124, subtitle, font_size=11, class_="muted"))
        out.append(f'<line x1="{left}" y1="145" x2="{left+span}" y2="145" class="grid"/>')
        out.append(svg_text(left, 140, "0", font_size=10, class_="muted"))
        out.append(svg_text(left+span, 140, str(maximum), font_size=10, text_anchor="end", class_="muted"))
    for i, model in enumerate(MODELS):
        y = 178 + i*54
        r = efficiency[model]
        color = "#3062c4" if model == "jev" else "#8a97aa"
        out.append(svg_text(28, y+6, NAMES[model], font_size=12, font_weight=700 if model == "jev" else 400))
        for left, span, value, maximum, label in [
            (247, 220, scores[model], 100, f"{scores[model]:.1f}%"),
            (520, 200, r["latency_p50_ms"] / 1000, 6, f'{r["latency_p50_ms"]/1000:.2f} s'),
            (770, 210, r["cny_per_1000_items"], 3, "¥0 · free*" if model == "qwen3.5-4b" else f'¥{r["cny_per_1000_items"]:.2f}')]:
            if value == 0:
                out.append(f'<circle cx="{left+2}" cy="{y+1}" r="4" fill="#8a97aa"/>')
            else:
                out.append(f'<rect x="{left}" y="{y-9}" width="{span*value/maximum:.2f}" height="20" rx="3" fill="{color}"/>')
            out.append(svg_text(left+span+11, y+5, label, font_size=12, font_weight=600))
        out.append(f'<line x1="28" y1="{y+22}" x2="1060" y2="{y+22}" class="grid"/>')
    out.append(svg_text(28, 580, "Cost: historical OpenRouter charge for Jev; SiliconFlow bill rates × benchmark tokens for Qwen; USD→CNY at 6.7459.", font_size=11, class_="muted"))
    out.append(svg_text(28, 599, "*4B uses SiliconFlow's announced free tariff; no separately named 4B bill line. Time is cross-provider client latency.", font_size=11, class_="muted"))
    return wrap_svg(1120, 620, "Score, API time and cost for seven benchmarked models", "Aligned comparison of equal-weight accuracy, median observed API latency and CNY cost per thousand attempts.", out)


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
    (FIGURES / "score-time-cost.svg").write_text(build_score_time_cost(stats, efficiency), encoding="utf-8")
    assert len(rows) == 1050
    for comp in stats["comparisons"]:
        model = comp["candidate"]
        observed = statistics.mean(sum(int(row[model]) for row in rows if row["dataset"] == ds) / sum(row["dataset"] == ds for row in rows) for ds in DATASET_ORDER)
        assert abs(observed - comp["candidate_macro_accuracy"]) < 1e-12
    print("Built report/figures/*.svg, results/reanalysis.csv and results/efficiency.json from frozen inputs")


if __name__ == "__main__":
    main()
