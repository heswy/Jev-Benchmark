"""Quality and calibration metrics (exact formulas per DESIGN.md §4)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true:
        return float("nan")
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
    if not labels:
        return float("nan")
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def score_mae(gold: Sequence[float], pred: Sequence[float]) -> float:
    if not gold:
        return float("nan")
    return sum(abs(float(g) - float(p)) for g, p in zip(gold, pred)) / len(gold)


def within_k_acc(gold: Sequence[float], pred: Sequence[float], k: float = 1.0) -> float:
    if not gold:
        return float("nan")
    return sum(1 for g, p in zip(gold, pred) if abs(float(g) - float(p)) <= k) / len(gold)


def binary_metrics(y_true: Sequence[bool], p_true: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    y = [bool(t) for t in y_true]
    pred = [p >= threshold for p in p_true]
    acc = accuracy(y, pred)
    tp = sum(1 for t, p in zip(y, pred) if t and p)
    fp = sum(1 for t, p in zip(y, pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y, pred) if t and (not p))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc_roc": auc_roc(y, p_true),
    }


def auc_roc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """Rank-based AUC-ROC; 0.5 if degenerate."""
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank with average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_ranks_pos = sum(ranks[k] for k in range(n) if pairs[k][1])
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier_multiclass(probs: Sequence[dict[str, float]], y_true: Sequence[str], label_set: Sequence[str]) -> float:
    """mean_k sum_c (p_c - y_c)^2  over fixed label set."""
    if not probs:
        return float("nan")
    s = 0.0
    for p, y in zip(probs, y_true):
        acc = 0.0
        for c in label_set:
            pc = float(p.get(c, 0.0))
            yc = 1.0 if c == y else 0.0
            acc += (pc - yc) ** 2
        s += acc
    return s / len(probs)


def brier_binary(p_true: Sequence[float], y_true: Sequence[bool]) -> float:
    if not p_true:
        return float("nan")
    return sum((float(p) - (1.0 if y else 0.0)) ** 2 for p, y in zip(p_true, y_true)) / len(p_true)


def ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Expected Calibration Error with equal-width bins on [0, 1]."""
    n = len(confidences)
    if n == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": []}
    bins: list[dict[str, Any]] = []
    ece_val = 0.0
    mce_val = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i
            for i, c in enumerate(confidences)
            if (c >= lo and (c < hi or (b == n_bins - 1 and c <= hi)))
        ]
        n_b = len(idx)
        if n_b == 0:
            bins.append({"bin": b, "lo": lo, "hi": hi, "n": 0, "acc": None, "conf": None, "gap": None})
            continue
        acc_b = sum(1 for i in idx if correct[i]) / n_b
        conf_b = sum(confidences[i] for i in idx) / n_b
        gap = abs(acc_b - conf_b)
        ece_val += (n_b / n) * gap
        mce_val = max(mce_val, gap)
        bins.append(
            {
                "bin": b,
                "lo": lo,
                "hi": hi,
                "n": n_b,
                "acc": acc_b,
                "conf": conf_b,
                "gap": gap,
            }
        )
    return {"ece": ece_val, "mce": mce_val, "bins": bins}


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def cost_usd(tokens_in: int, tokens_out: int, pin_per_m: float, pout_per_m: float) -> float:
    return (tokens_in * pin_per_m + tokens_out * pout_per_m) / 1_000_000.0


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))
