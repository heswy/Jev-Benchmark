"""Model registry and run constants for JEV-Benchmark."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    openrouter_id: str
    kind: str  # "jev" | "llm"
    display: str
    price_in_per_m: float  # USD per 1M input tokens
    price_out_per_m: float  # USD per 1M output tokens
    family: str


MODELS: dict[str, ModelSpec] = {
    "jev": ModelSpec(
        id="jev",
        openrouter_id="typesafe/jev-1.13",
        kind="jev",
        display="Jev 1.13",
        price_in_per_m=0.042,
        price_out_per_m=0.0,
        family="System One",
    ),
    "lfm-2.6b": ModelSpec(
        id="lfm-2.6b",
        openrouter_id="liquid/lfm-2.5-2.6b",
        kind="llm",
        display="LFM2.5 2.6B",
        price_in_per_m=0.02,
        price_out_per_m=0.05,
        family="small-open",
    ),
    "granite-8b": ModelSpec(
        id="granite-8b",
        openrouter_id="ibm-granite/granite-4.2-8b",
        kind="llm",
        display="Granite 4.2 8B",
        price_in_per_m=0.06,
        price_out_per_m=0.25,
        family="small-open",
    ),
    "qwen3.5-4b": ModelSpec(
        id="qwen3.5-4b",
        openrouter_id="Qwen/Qwen3.5-4B",
        kind="llm-sf",
        display="Qwen3.5 4B",
        # List-price placeholders (USD / 1M); SiliconFlow usage payload has no cost field.
        price_in_per_m=0.02,
        price_out_per_m=0.08,
        family="small-open-siliconflow",
    ),
    "qwen3.5-9b": ModelSpec(
        id="qwen3.5-9b",
        openrouter_id="Qwen/Qwen3.5-9B",
        kind="llm-sf",
        display="Qwen3.5 9B",
        price_in_per_m=0.04,
        price_out_per_m=0.16,
        family="small-open-siliconflow",
    ),
    "qwen3.5-27b": ModelSpec(
        id="qwen3.5-27b",
        openrouter_id="Qwen/Qwen3.5-27B",
        kind="llm-sf",
        display="Qwen3.5 27B",
        price_in_per_m=0.10,
        price_out_per_m=0.30,
        family="small-open-siliconflow",
    ),
    "qwen3.5-35b": ModelSpec(
        id="qwen3.5-35b",
        openrouter_id="Qwen/Qwen3.5-35B-A3B",
        kind="llm-sf",
        display="Qwen3.5 35B-A3B",
        price_in_per_m=0.15,
        price_out_per_m=0.45,
        family="small-open-siliconflow",
    ),
    "qwen3.5-122b": ModelSpec(
        id="qwen3.5-122b",
        openrouter_id="Qwen/Qwen3.5-122B-A10B",
        kind="llm-sf",
        display="Qwen3.5 122B-A10B",
        price_in_per_m=0.35,
        price_out_per_m=0.70,
        family="small-open-siliconflow",
    ),
    "qwen3.8-27b": ModelSpec(
        id="qwen3.8-27b",
        openrouter_id="Qwen/Qwen3.8-27B",
        kind="llm-sf",
        display="Qwen3.8 27B",
        # SiliconFlow publishes CNY prices; USD cost is unavailable here.
        price_in_per_m=0.0,
        price_out_per_m=0.0,
        family="upper-bound-siliconflow",
    ),
    "qwen3.5-397b": ModelSpec(
        id="qwen3.5-397b",
        openrouter_id="qwen/qwen3.5-397b-a17b",
        kind="llm",
        display="Qwen3.5 397B-A17B",
        price_in_per_m=0.39,
        price_out_per_m=2.34,
        family="upper-bound-openrouter",
    ),
    "qwen3.5-9b-or": ModelSpec(
        id="qwen3.5-9b-or",
        openrouter_id="qwen/qwen3.5-9b",
        kind="llm",
        display="Qwen3.5 9B (OR)",
        price_in_per_m=0.10,
        price_out_per_m=0.15,
        family="small-open",
    ),
    "gemma4-moe": ModelSpec(
        id="gemma4-moe",
        openrouter_id="google/gemma-4-26b-a4b-it",
        kind="llm",
        display="Gemma 4 26B-A4B",
        price_in_per_m=0.09,
        price_out_per_m=0.30,
        family="small-open",
    ),
    "gpt-5.6-luna": ModelSpec(
        id="gpt-5.6-luna",
        openrouter_id="openai/gpt-5.6-luna",
        kind="llm",
        display="GPT-5.6 Luna",
        price_in_per_m=0.20,
        price_out_per_m=1.20,
        family="cloud-fast",
    ),
    "mimo-flash": ModelSpec(
        id="mimo-flash",
        openrouter_id="xiaomi/mimo-v2.6-flash",
        kind="llm",
        display="MiMo-V2.6-Flash",
        price_in_per_m=0.14,
        price_out_per_m=0.28,
        family="cloud-fast",
    ),
}

DEFAULT_MODEL_ORDER = [
    "jev",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "lfm-2.6b",
    "granite-8b",
    "qwen3.5-9b-or",
    "gemma4-moe",
    "gpt-5.6-luna",
    "mimo-flash",
]

# SiliconFlow (LLM peers for this run)
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
SILICONFLOW_CHAT = f"{SILICONFLOW_BASE}/chat/completions"
ENABLE_THINKING = False
THINKING_BUDGET = 128

DATASET_ORDER = ["banking77", "clinc150", "sst5", "boolq", "agnews"]

TEMPERATURE = 0.0
MAX_TOKENS_LLM = 800
# Keep LLM peers in the same "fast decision" band as Jev (not long reasoning).
REASONING_EFFORT = "low"
REASONING_EXCLUDE = True
CONCURRENCY = 4
NETWORK_RETRIES = 4  # network / 429 / 5xx only; format errors never retried
PROB_SUM_TOL = 0.03
OOS_TAU = 0.35
ECE_BINS = 10
SEED = 42

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TYPESAFE_BASE = "https://api.typesafe.ai/v1"
JEV_ENDPOINTS = {
    "openrouter": f"{OPENROUTER_BASE}/systemone",
    "typesafe": f"{TYPESAFE_BASE}/systemone",
}
CHAT_ENDPOINT = f"{OPENROUTER_BASE}/chat/completions"
