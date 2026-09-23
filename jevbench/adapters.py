"""Model adapters: TypeSafe/OpenRouter System One (Jev) and OpenRouter chat (LLM)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import (
    CHAT_ENDPOINT,
    ENABLE_THINKING,
    JEV_ENDPOINTS,
    MAX_TOKENS_LLM,
    NETWORK_RETRIES,
    PROB_SUM_TOL,
    REASONING_EXCLUDE,
    REASONING_EFFORT,
    SILICONFLOW_CHAT,
    TEMPERATURE,
    THINKING_BUDGET,
    ModelSpec,
)
from .prompts import build_jev_request, build_llm_messages, parse_llm_json
from .schemas import Prediction, Sample


class NetworkError(RuntimeError):
    """Transient transport failure (retryable). Not a format error."""


def _read_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def _load_api_key(provider: str = "openrouter") -> str:
    """provider: openrouter | siliconflow | typesafe"""
    root = Path(__file__).resolve().parent.parent
    file_env: dict[str, str] = {}
    for path in (
        os.environ.get("JEV_ENV_FILE"),
        str(root / ".env"),
    ):
        if path:
            file_env.update(_read_env_file(path))

    def pick(*names: str) -> str:
        for n in names:
            if os.environ.get(n):
                return os.environ[n].strip()
            if file_env.get(n):
                return file_env[n]
        return ""

    if provider == "siliconflow":
        key = pick("SILICONFLOW_API_KEY")
    elif provider == "typesafe":
        key = pick("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")
    else:
        key = pick("OPENROUTER_API_KEY", "TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError(f"No API key for {provider}. Set env or .env.")
    return key


def _http_post_json(url: str, payload: dict[str, Any], api_key: str, timeout: int = 60) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JEV-Benchmark/0.1",
            "HTTP-Referer": "https://localhost/jev-benchmark",
            "X-Title": "JEV-Benchmark",
        },
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(NETWORK_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            # Rate limit / overload: retry with longer backoff (SiliconFlow TPM/TPD)
            if e.code in {408, 425, 429, 500, 502, 503, 504, 529} and attempt < NETWORK_RETRIES:
                time.sleep(min(30.0, 3.0 * (attempt + 1) ** 2))
                last = NetworkError(f"HTTP {e.code}: {detail}")
                continue
            if e.code == 429:
                raise NetworkError(f"HTTP 429: {detail}") from e
            raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e
        except Exception as e:  # noqa: BLE001 — transport
            last = NetworkError(str(e))
            if attempt < NETWORK_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise NetworkError(str(e)) from e
    raise NetworkError(str(last))


def _normalize_probs(raw: dict[str, Any], keys: list[str]) -> dict[str, float] | None:
    """Normalize to a distribution over `keys`.

    Models often omit low-probability options on large label sets. Missing keys
    default to 0; the provided mass is renormalized over the full key set.
    """
    if not isinstance(raw, dict):
        return None
    raw_s = {str(k): v for k, v in raw.items()}
    vals: dict[str, float] = {}
    any_hit = False
    for k in keys:
        if k in raw_s:
            try:
                v = float(raw_s[k])
            except Exception:
                return None
            if v < -1e-9:
                return None
            vals[k] = v
            if v > 0:
                any_hit = True
        else:
            vals[k] = 0.0
    if not any_hit:
        return None
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in vals.items()}


def _peak(probs: dict[str, float]) -> tuple[str, float]:
    key = max(probs, key=lambda k: probs[k])
    return key, probs[key]


@dataclass
class AdapterResult:
    prediction: Prediction
    network_attempts: int = 1
    extras: dict[str, Any] = field(default_factory=dict)


class BaseModelAdapter:
    kind: str = "base"

    def __init__(self, spec: ModelSpec, api_key: str | None = None, provider: str = "openrouter"):
        self.spec = spec
        self.provider = provider
        self.api_key = api_key or _load_api_key(provider)

    def predict(self, sample: Sample) -> AdapterResult:
        raise NotImplementedError


class JevAdapter(BaseModelAdapter):
    """System One via OpenRouter systemone (default) or TypeSafe."""

    kind = "jev"

    def __init__(self, spec: ModelSpec, api_key: str | None = None, endpoint: str = "openrouter"):
        super().__init__(spec, api_key=api_key, provider="typesafe" if endpoint == "typesafe" else "openrouter")
        self.url = JEV_ENDPOINTS[endpoint]
        self.model_id = spec.openrouter_id if endpoint == "openrouter" else "jev-latest"
        if endpoint == "typesafe" and spec.openrouter_id.startswith("typesafe/"):
            self.model_id = spec.openrouter_id.split("/", 1)[1]

    def predict(self, sample: Sample) -> AdapterResult:
        t0 = time.perf_counter()
        payload = build_jev_request(sample, model=self.model_id)
        try:
            data = _http_post_json(self.url, payload, self.api_key)
        except NetworkError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(str(e)) from e
        latency = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        answers = data.get("answers") or data.get("response") or {}
        ans = answers.get("q") or next(iter(answers.values()), {}) if answers else {}
        pred = self._map_answer(sample, ans, latency, tokens_in, tokens_out, data)
        return AdapterResult(prediction=pred)

    def _map_answer(
        self,
        sample: Sample,
        ans: dict[str, Any],
        latency: float,
        tokens_in: int,
        tokens_out: int,
        raw: dict[str, Any],
    ) -> Prediction:
        base = dict(
            id=sample.id,
            dataset=sample.dataset,
            model=self.spec.id,
            task_type=sample.task_type,
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            raw=raw,
        )
        if not ans:
            return Prediction(
                **base,
                answer=None,
                probabilities={},
                confidence=None,
                format_valid=False,
                label_in_set=False,
                error="empty jev answer",
            )

        if sample.task_type == "choice":
            choice = ans.get("choice")
            probs_raw = ans.get("probabilities") or {}
            keys = list(sample.labels)
            probs = {k: float(probs_raw.get(k, 0.0)) for k in keys}
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}
            conf = float(ans.get("confidence") if ans.get("confidence") is not None else max(probs.values()))
            ok_fmt = choice in sample.labels
            return Prediction(
                **base,
                answer=choice,
                probabilities=probs,
                confidence=conf,
                format_valid=ok_fmt,
                label_in_set=ok_fmt,
            )

        if sample.task_type == "score":
            # score answer: score float + probabilities over level indices as strings
            score = ans.get("score")
            probs_raw = ans.get("probabilities") or {}
            level_keys = [str(i) for i in (sample.levels or [0, 1, 2, 3, 4])]
            probs = {k: float(probs_raw.get(k, probs_raw.get(int(k) if str(k).isdigit() else k, 0.0))) for k in level_keys}
            # also accept int keys from JSON (unlikely)
            if sum(probs.values()) == 0:
                for i in level_keys:
                    if i in probs_raw:
                        probs[i] = float(probs_raw[i])
                    elif int(i) in probs_raw:
                        probs[i] = float(probs_raw[int(i)])
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}
            expected = sum(int(k) * v for k, v in probs.items())
            try:
                pred_score = float(score) if score is not None else expected
            except Exception:
                pred_score = expected
            conf = float(ans.get("confidence") if ans.get("confidence") is not None else max(probs.values()))
            ok = (sample.levels or [0, 1, 2, 3, 4])
            return Prediction(
                **base,
                answer=pred_score,
                probabilities=probs,
                confidence=conf,
                format_valid=True,
                label_in_set=min(ok) <= pred_score <= max(ok),
            )

        # noul
        noul = ans.get("noul")
        try:
            p_true = float(noul)
        except Exception:
            p_true = float("nan")
        ok = p_true == p_true and 0.0 <= p_true <= 1.0
        probs = {"true": p_true, "false": 1.0 - p_true} if ok else {}
        return Prediction(
            **base,
            answer=bool(p_true >= 0.5) if ok else None,
            probabilities=probs,
            confidence=abs(p_true - 0.5) * 2.0 if ok else None,
            format_valid=ok,
            label_in_set=ok,
        )


class LLMAdapter(BaseModelAdapter):
    """Chat/completions (OpenRouter or SiliconFlow) with one-shot JSON decision contract."""

    kind = "llm"

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str | None = None,
        provider: str = "openrouter",
        endpoint: str | None = None,
    ):
        super().__init__(spec, api_key=api_key, provider=provider)
        self.endpoint = endpoint or (SILICONFLOW_CHAT if provider == "siliconflow" else CHAT_ENDPOINT)

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.openrouter_id,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS_LLM,
            "response_format": {"type": "json_object"},
        }
        if self.provider == "siliconflow":
            # Force non-thinking decision band (user requirement).
            payload["enable_thinking"] = ENABLE_THINKING
            payload["thinking_budget"] = THINKING_BUDGET
        else:
            effort = "none" if self.spec.id == "qwen3.5-397b" else REASONING_EFFORT
            payload["reasoning"] = {"effort": effort, "exclude": REASONING_EXCLUDE}
        return payload

    def predict(self, sample: Sample) -> AdapterResult:
        messages = build_llm_messages(sample)
        payload = self._payload(messages)
        t0 = time.perf_counter()
        data = _http_post_json(self.endpoint, payload, self.api_key)
        latency = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        content = ""
        try:
            msg = data["choices"][0]["message"] or {}
            content = msg.get("content") or ""
            if not content:
                content = msg.get("reasoning") or msg.get("reasoning_content") or ""
        except Exception:
            content = (data.get("choices") or [{}])[0].get("text", "")
        cost_usd = usage.get("cost")
        return AdapterResult(
            prediction=self._map(sample, content, latency, tokens_in, tokens_out, data),
            extras={"provider_cost_usd": cost_usd},
        )

    def _map(
        self,
        sample: Sample,
        content: str,
        latency: float,
        tokens_in: int,
        tokens_out: int,
        raw: dict[str, Any],
    ) -> Prediction:
        base = dict(
            id=sample.id,
            dataset=sample.dataset,
            model=self.spec.id,
            task_type=sample.task_type,
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            raw=raw,
        )
        try:
            obj = parse_llm_json(content)
        except Exception as e:  # noqa: BLE001 — format error, no retry
            return Prediction(
                **base,
                answer=None,
                probabilities={},
                confidence=None,
                format_valid=False,
                label_in_set=False,
                error=f"format: {e}",
            )

        if sample.task_type == "choice":
            keys = list(sample.labels)
            answer = obj.get("answer")
            raw_p = obj.get("probabilities") or {}
            if isinstance(raw_p, dict) and answer in keys and answer not in raw_p:
                # ensure the chosen answer has mass even if omitted
                conf_hint = obj.get("confidence")
                raw_p = dict(raw_p)
                raw_p[answer] = float(conf_hint) if conf_hint is not None else 1.0
            probs = _normalize_probs(raw_p, keys)
            if answer not in keys:
                return Prediction(
                    **base,
                    answer=None,
                    probabilities=probs or {},
                    confidence=obj.get("confidence"),
                    format_valid=False,
                    label_in_set=False,
                    error="answer not in label set",
                )
            if probs is None:
                # last resort: one-hot on answer (still counts as format-valid; calibration degrades)
                if isinstance(obj.get("confidence"), (int, float)):
                    conf = float(obj["confidence"])
                    rest = max(0.0, 1.0 - conf)
                    share = rest / max(1, len(keys) - 1)
                    probs = {k: (conf if k == answer else share) for k in keys}
                    s = sum(probs.values()) or 1.0
                    probs = {k: v / s for k, v in probs.items()}
                else:
                    return Prediction(
                        **base,
                        answer=answer,
                        probabilities={},
                        confidence=None,
                        format_valid=False,
                        label_in_set=True,
                        error="bad choice contract",
                    )
            conf = float(obj.get("confidence", max(probs.values())))
            return Prediction(
                **base,
                answer=answer,
                probabilities=probs,
                confidence=conf,
                format_valid=True,
                label_in_set=True,
            )

        if sample.task_type == "score":
            level_keys = [str(i) for i in (sample.levels or [0, 1, 2, 3, 4])]
            raw_p = obj.get("probabilities") or {}
            # coerce int keys to str
            raw_p = {str(k): v for k, v in raw_p.items()}
            probs = _normalize_probs(raw_p, level_keys)
            if probs is None:
                return Prediction(
                    **base,
                    answer=None,
                    probabilities={},
                    confidence=None,
                    format_valid=False,
                    label_in_set=False,
                    error="bad score probabilities",
                )
            expected = sum(int(k) * v for k, v in probs.items())
            try:
                pred_score = float(obj.get("answer"))
            except Exception:
                pred_score = expected
            lo, hi = 0, max(int(k) for k in level_keys)
            in_set = lo - 1e-6 <= pred_score <= hi + 1e-6
            conf = float(obj.get("confidence", max(probs.values())))
            return Prediction(
                **base,
                answer=pred_score,
                probabilities=probs,
                confidence=conf,
                format_valid=in_set,
                label_in_set=in_set,
                error=None if in_set else "score out of range",
            )

        # noul
        raw_p = obj.get("probabilities") or {}
        if "true" in raw_p or "false" in raw_p:
            p_true = float(raw_p.get("true", raw_p.get("false", 0.0)))
            if "true" not in raw_p and "false" in raw_p:
                p_true = 1.0 - float(raw_p["false"])
        elif "p_true" in obj:
            p_true = float(obj["p_true"])
        else:
            return Prediction(
                **base,
                answer=None,
                probabilities={},
                confidence=None,
                format_valid=False,
                label_in_set=False,
                error="bad noul contract",
            )
        p_true = float(p_true)
        if not (0.0 <= p_true <= 1.0):
            return Prediction(
                **base,
                answer=None,
                probabilities={},
                confidence=None,
                format_valid=False,
                label_in_set=False,
                error="p_true out of range",
            )
        answer = bool(p_true >= 0.5)
        # allow explicit answer field consistency
        if "answer" in obj and isinstance(obj["answer"], bool):
            answer = obj["answer"]
        conf = float(obj.get("confidence", abs(p_true - 0.5) * 2.0))
        return Prediction(
            **base,
            answer=answer,
            probabilities={"true": p_true, "false": 1.0 - p_true},
            confidence=conf,
            format_valid=True,
            label_in_set=True,
        )


def make_adapter(spec: ModelSpec, endpoint: str = "openrouter") -> BaseModelAdapter:
    if spec.kind == "jev":
        return JevAdapter(spec, endpoint=endpoint)
    if spec.kind == "llm-sf":
        return LLMAdapter(spec, provider="siliconflow")
    return LLMAdapter(spec, provider="openrouter")
