#!/usr/bin/env python3
"""Run OpenEvolve with prompt-level code deduplication enabled."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Set, Tuple

from openevolve import cli as openevolve_cli
from openevolve.evaluator import Evaluator
from openevolve.iteration import run_iteration_with_shared_db as _ORIGINAL_RUN_ITERATION
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.openai import OpenAILLM
from openevolve.prompt.sampler import PromptSampler

LOGGER = logging.getLogger("openevolve_dedup_runner")
_PROMPT_MIN_INTERVAL_SEC = float(os.environ.get("OE_PROMPT_MIN_INTERVAL_SEC", "0") or 0.0)
_LAST_PROMPT_TS = 0.0
_PROMPT_LOCK = threading.Lock()


def _normalize_source(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _program_payload(program: Dict[str, Any], prefer_changes: bool) -> str:
    if not isinstance(program, dict):
        return ""
    primary = program.get("changes_description") if prefer_changes else program.get("code")
    fallback = program.get("code") if prefer_changes else program.get("changes_description")
    text = primary if isinstance(primary, str) and primary.strip() else fallback
    return _normalize_source(text if isinstance(text, str) else "")


def _fingerprint(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dedupe_programs(
    programs: List[Dict[str, Any]],
    seen: Set[str],
    prefer_changes: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    if not programs:
        return [], 0
    out: List[Dict[str, Any]] = []
    removed = 0
    for p in programs:
        fp = _fingerprint(_program_payload(p, prefer_changes))
        if fp in seen:
            removed += 1
            continue
        seen.add(fp)
        out.append(p)
    return out, removed


_ORIGINAL_BUILD_PROMPT = PromptSampler.build_prompt
_BUILD_PROMPT_SIG = inspect.signature(_ORIGINAL_BUILD_PROMPT)


def _build_prompt_with_dedup(self: PromptSampler, *args: Any, **kwargs: Any) -> Dict[str, str]:
    global _LAST_PROMPT_TS
    if _PROMPT_MIN_INTERVAL_SEC > 0.0:
        with _PROMPT_LOCK:
            now = time.monotonic()
            wait_sec = _PROMPT_MIN_INTERVAL_SEC - (now - _LAST_PROMPT_TS)
            if wait_sec > 0.0:
                LOGGER.info("Prompt throttling sleep %.2fs (OE_PROMPT_MIN_INTERVAL_SEC)", wait_sec)
                time.sleep(wait_sec)
            _LAST_PROMPT_TS = time.monotonic()

    bound = _BUILD_PROMPT_SIG.bind_partial(self, *args, **kwargs)
    prefer_changes = bool(getattr(self.config, "programs_as_changes_description", False))

    seen: Set[str] = set()
    previous = list(bound.arguments.get("previous_programs", []) or [])
    top = list(bound.arguments.get("top_programs", []) or [])
    inspirations = list(bound.arguments.get("inspirations", []) or [])

    previous_dedup, removed_prev = _dedupe_programs(previous, seen, prefer_changes)
    top_dedup, removed_top = _dedupe_programs(top, seen, prefer_changes)
    inspirations_dedup, removed_insp = _dedupe_programs(inspirations, seen, prefer_changes)

    bound.arguments["previous_programs"] = previous_dedup
    bound.arguments["top_programs"] = top_dedup
    bound.arguments["inspirations"] = inspirations_dedup

    # Keep telemetry in trace/db, but never feed it back into model prompts.
    program_artifacts = bound.arguments.get("program_artifacts")
    if isinstance(program_artifacts, dict) and "qos_iteration_stats_json" in program_artifacts:
        filtered = {k: v for k, v in program_artifacts.items() if k != "qos_iteration_stats_json"}
        bound.arguments["program_artifacts"] = filtered
        LOGGER.info("Prompt artifacts filter removed 'qos_iteration_stats_json'")

    removed_total = removed_prev + removed_top + removed_insp
    if removed_total > 0:
        LOGGER.info(
            "Prompt dedup removed %d duplicated program snippets "
            "(previous=%d, top=%d, inspirations=%d)",
            removed_total,
            removed_prev,
            removed_top,
            removed_insp,
        )

    return _ORIGINAL_BUILD_PROMPT(*bound.args, **bound.kwargs)


def _apply_prompt_dedup_patch() -> None:
    if getattr(PromptSampler, "_qos_prompt_dedup_patched", False):
        return
    PromptSampler.build_prompt = _build_prompt_with_dedup
    PromptSampler._qos_prompt_dedup_patched = True


_apply_prompt_dedup_patch()


_ORIGINAL_EVALUATE_PROGRAM = Evaluator.evaluate_program


def _normalize_failed_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics or {})
    has_failure = ("error" in out) or bool(out.get("timeout"))
    if has_failure:
        # Prevent failed candidates from being treated as competitive programs.
        out["score"] = -1e12
        out["combined_score"] = -1e12
        out.setdefault("avg_rank", 1e12)
        out.setdefault("inv_avg_rank", 0.0)
        out.setdefault("rank_score_corr", -1.0)
        return out
    if "combined_score" not in out and "score" in out:
        out["combined_score"] = out["score"]
    return out


async def _evaluate_program_with_guardrail(self: Evaluator, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    metrics = await _ORIGINAL_EVALUATE_PROGRAM(self, *args, **kwargs)
    if isinstance(metrics, dict):
        return _normalize_failed_metrics(metrics)
    return metrics


def _apply_evaluator_guardrail_patch() -> None:
    if getattr(Evaluator, "_qos_failure_guardrail_patched", False):
        return
    Evaluator.evaluate_program = _evaluate_program_with_guardrail
    Evaluator._qos_failure_guardrail_patched = True


_apply_evaluator_guardrail_patch()


_ORIGINAL_OPENAI_CALL_API = OpenAILLM._call_api
_RESPONSES_FORCE_ENV = os.environ.get("OE_OPENAI_USE_RESPONSES", "auto").strip().lower()
_RESPONSES_MODEL_HINTS = (
    "gpt-5-codex",
    "gpt-5.3-codex",
)
_OPENAI_SERVICE_TIER = (os.environ.get("OPENAI_SERVICE_TIER", "") or "").strip()
_OPENAI_SERVICE_TIER_NORM = _OPENAI_SERVICE_TIER.lower()
_PRICE_TABLE_RAW = os.environ.get("OE_TOKEN_PRICE_TABLE_JSON", "").strip()
_DEFAULT_INPUT_USD_PER_1M = float(os.environ.get("OE_PRICE_INPUT_USD_PER_1M", "0") or 0.0)
_DEFAULT_OUTPUT_USD_PER_1M = float(os.environ.get("OE_PRICE_OUTPUT_USD_PER_1M", "0") or 0.0)

# Built-in fallback prices used when OE_TOKEN_PRICE_TABLE_JSON is not provided.
# Sources (queried 2026-03-05):
# - OpenAI API pricing: https://platform.openai.com/docs/pricing
# - Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
_BUILTIN_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {
        "input_usd_per_1m": 0.25,
        "output_usd_per_1m": 2.0,
    },
    "gpt-5-mini:flex": {
        "input_usd_per_1m": 0.125,
        "output_usd_per_1m": 1.0,
    },
    # Gemini 3 Flash Preview (Gemini Developer API, Standard tier, text/image/video input).
    "gemini-3-flash-preview": {
        "input_usd_per_1m": 0.50,
        "output_usd_per_1m": 3.00,
    },
    # Gemini 3 Pro Preview (Gemini Developer API, Standard tier, <=200k token pricing tier).
    "gemini-3-pro-preview": {
        "input_usd_per_1m": 2.50,
        "output_usd_per_1m": 15.00,
    },
}


def _estimate_tokens_from_text(text: str) -> int:
    """Provider-agnostic fallback token estimate."""
    if not isinstance(text, str) or not text:
        return 0
    # Simple and stable heuristic for mixed providers: ~4 chars/token.
    return max(1, int(round(len(text) / 4.0)))


def _to_json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_json_safe(model_dump())
        except Exception:
            pass
    dct = getattr(value, "__dict__", None)
    if isinstance(dct, dict):
        return _to_json_safe(dct)
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def _load_price_table() -> Dict[str, Dict[str, float]]:
    if not _PRICE_TABLE_RAW:
        return {}
    try:
        obj = json.loads(_PRICE_TABLE_RAW)
    except Exception:
        LOGGER.warning("Invalid OE_TOKEN_PRICE_TABLE_JSON; ignoring.")
        return {}
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        in_rate = v.get("input_usd_per_1m", v.get("in_usd_per_1m"))
        out_rate = v.get("output_usd_per_1m", v.get("out_usd_per_1m"))
        try:
            out[k.lower()] = {
                "input_usd_per_1m": float(in_rate),
                "output_usd_per_1m": float(out_rate),
            }
        except Exception:
            continue
    return out


_PRICE_TABLE = {**_BUILTIN_PRICE_TABLE, **_load_price_table()}


def _resolve_token_prices(model: str, provider: str) -> Dict[str, Any]:
    ml = str(model or "").strip().lower()
    pl = str(provider or "").strip().lower()

    def _norm_key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    ml_n = _norm_key(ml)
    pl_n = _norm_key(pl)

    # Canonical aliases for requested model families.
    alias_candidates = [
        ml,
        ml_n,
    ]
    if ("gemini" in ml_n and "3" in ml_n and "pro" in ml_n) or ("gemini3pro" in ml_n):
        alias_candidates.extend(
            [
                "gemini3pro",
                "gemini-3-pro-preview",
                "gemini 3 pro preview",
                "gemini-3-pro",
                "gemini 3 pro",
            ]
        )
    if ("gemini" in ml_n and "3" in ml_n and "flash" in ml_n) or ("gemini3flash" in ml_n):
        alias_candidates.extend(["gemini3flash", "gemini-3-flash", "gemini 3 flash"])
    if ("gpt" in ml_n and "5" in ml_n and "mini" in ml_n) or ("gpt5mini" in ml_n):
        alias_candidates.extend(["gpt5mini", "gpt-5-mini", "gpt 5 mini"])
    if ("gpt" in ml_n and "53" in ml_n and "codex" in ml_n) or ("gpt53codex" in ml_n):
        alias_candidates.extend(["gpt53codex", "gpt-5.3-codex", "gpt 5.3 codex"])
    if ("claude" in ml_n and "sonnet" in ml_n and "46" in ml_n) or ("claudesonnet46" in ml_n):
        alias_candidates.extend(
            [
                "claudesonnet46",
                "claude-sonnet-4-6",
                "claude sonnet 4 6",
            ]
        )
    if ("claude" in ml_n and "opus" in ml_n and "46" in ml_n) or ("claudeopus46" in ml_n):
        alias_candidates.extend(
            [
                "claudeopus46",
                "claude-opus-4-6",
                "claude opus 4 6",
            ]
        )

    # Also allow provider-level aliases.
    provider_aliases = [pl, pl_n]
    if "gemini" in pl_n:
        provider_aliases.extend(["gemini_openai_compat", "gemini"])
    if "claude" in pl_n or "anthropic" in pl_n:
        provider_aliases.extend(["claude_openai_compat", "claude", "anthropic"])
    if "openai" in pl_n or "gpt" in pl_n:
        provider_aliases.extend(["openai_compatible", "openai"])

    # Deduplicate while preserving order.
    alias_candidates = list(dict.fromkeys([x for x in alias_candidates if x]))
    provider_aliases = list(dict.fromkeys([x for x in provider_aliases if x]))

    # Prefer tier-specific pricing keys when service tier is configured, e.g. "gpt-5-mini:flex".
    if _OPENAI_SERVICE_TIER_NORM:
        alias_candidates = (
            [f"{x}:{_OPENAI_SERVICE_TIER_NORM}" for x in alias_candidates] + alias_candidates
        )
        provider_aliases = (
            [f"{x}:{_OPENAI_SERVICE_TIER_NORM}" for x in provider_aliases] + provider_aliases
        )

    # Build a normalized lookup view once per call.
    table_norm: Dict[str, Dict[str, float]] = {}
    for k, v in _PRICE_TABLE.items():
        table_norm[k] = v
        table_norm[_norm_key(k)] = v

    # 1) exact model key
    for key in alias_candidates:
        if key in table_norm:
            v = table_norm[key]
            return {
                "input_usd_per_1m": v["input_usd_per_1m"],
                "output_usd_per_1m": v["output_usd_per_1m"],
                "pricing_source": f"table:model:{key}",
            }
    # 2) provider key
    for key in provider_aliases:
        if key in table_norm:
            v = table_norm[key]
            return {
                "input_usd_per_1m": v["input_usd_per_1m"],
                "output_usd_per_1m": v["output_usd_per_1m"],
                "pricing_source": f"table:provider:{key}",
            }
    # 3) prefix key, e.g. "gpt-5", "claude", "gemini"
    for k, v in _PRICE_TABLE.items():
        if ml.startswith(k) or ml_n.startswith(_norm_key(k)):
            return {
                "input_usd_per_1m": v["input_usd_per_1m"],
                "output_usd_per_1m": v["output_usd_per_1m"],
                "pricing_source": f"table:prefix:{k}",
            }
    # 4) global fallback
    return {
        "input_usd_per_1m": _DEFAULT_INPUT_USD_PER_1M,
        "output_usd_per_1m": _DEFAULT_OUTPUT_USD_PER_1M,
        "pricing_source": "env_default",
    }


def _estimate_cost_usd(
    prompt_tokens: Any,
    output_tokens: Any,
    model: str,
    provider: str,
) -> Dict[str, Any]:
    try:
        pt = int(prompt_tokens or 0)
        ot = int(output_tokens or 0)
    except Exception:
        pt, ot = 0, 0
    prices = _resolve_token_prices(model=model, provider=provider)
    in_rate = float(prices["input_usd_per_1m"])
    out_rate = float(prices["output_usd_per_1m"])
    in_cost = (pt / 1_000_000.0) * in_rate
    out_cost = (ot / 1_000_000.0) * out_rate
    return {
        "price_input_usd_per_1m": in_rate,
        "price_output_usd_per_1m": out_rate,
        "pricing_source": prices["pricing_source"],
        "cost_input_usd": in_cost,
        "cost_output_usd": out_cost,
        "cost_total_usd": in_cost + out_cost,
    }


def _extract_usage_tokens(response: Any) -> Dict[str, Any]:
    """Extract token usage from OpenAI-compatible responses when available."""
    usage = getattr(response, "usage", None)
    out: Dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_prompt_tokens": None,
        "reasoning_tokens": None,
        "thinking_tokens": None,
        "usage_raw": None,
        "usage_source": None,
    }
    if usage is None:
        return out

    # Keep a compact raw snapshot for debugging/reporting.
    raw_snapshot: Dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "output_tokens_details",
    ):
        val = getattr(usage, key, None)
        if val is not None:
            raw_snapshot[key] = val
    out["usage_raw"] = _to_json_safe(raw_snapshot)

    for k in ("prompt_tokens", "input_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out["prompt_tokens"] = v
            break
    for k in ("completion_tokens", "output_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out["completion_tokens"] = v
            break
    for k in ("total_tokens",):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out["total_tokens"] = v
            break
    # OpenAI detail fields (if present)
    pdet = getattr(usage, "prompt_tokens_details", None)
    if pdet is not None:
        c = getattr(pdet, "cached_tokens", None)
        if isinstance(c, int):
            out["cached_prompt_tokens"] = c
    cdet = getattr(usage, "completion_tokens_details", None) or getattr(
        usage, "output_tokens_details", None
    )
    if cdet is not None:
        r = getattr(cdet, "reasoning_tokens", None)
        if isinstance(r, int):
            out["reasoning_tokens"] = r
        t = getattr(cdet, "thinking_tokens", None)
        if isinstance(t, int):
            out["thinking_tokens"] = t
    out["usage_source"] = "api_usage"
    return out


def _extract_usage_from_raw_payload(raw: Any) -> Dict[str, Any]:
    """
    Provider-agnostic raw usage parser.
    Supports:
      - Gemini native: raw["usageMetadata"]
      - OpenAI compatible: raw["usage"]
    """
    out: Dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_prompt_tokens": None,
        "reasoning_tokens": None,
        "thinking_tokens": None,
        "usage_raw": None,
        "usage_source": None,
    }
    if not isinstance(raw, dict):
        return out

    # Gemini native payload
    gm = raw.get("usageMetadata")
    if isinstance(gm, dict):
        out["prompt_tokens"] = gm.get("promptTokenCount")
        out["completion_tokens"] = gm.get("candidatesTokenCount")
        out["total_tokens"] = gm.get("totalTokenCount")
        out["cached_prompt_tokens"] = gm.get("cachedContentTokenCount")
        # Optional thinking/reasoning style counters
        out["thinking_tokens"] = gm.get("thinkingTokenCount", gm.get("thoughtsTokenCount"))
        out["reasoning_tokens"] = gm.get("reasoningTokenCount", gm.get("reasoningTokens"))
        out["usage_raw"] = _to_json_safe(gm)
        out["usage_source"] = "gemini_native_usageMetadata"
        return out

    # OpenAI-compatible payload
    us = raw.get("usage")
    if isinstance(us, dict):
        out["prompt_tokens"] = us.get("prompt_tokens", us.get("input_tokens"))
        out["completion_tokens"] = us.get("completion_tokens", us.get("output_tokens"))
        out["total_tokens"] = us.get("total_tokens")
        pdet = us.get("prompt_tokens_details") or {}
        if isinstance(pdet, dict):
            out["cached_prompt_tokens"] = pdet.get("cached_tokens")
        cdet = us.get("completion_tokens_details") or us.get("output_tokens_details") or {}
        if isinstance(cdet, dict):
            out["reasoning_tokens"] = cdet.get("reasoning_tokens")
            out["thinking_tokens"] = cdet.get("thinking_tokens")
        out["usage_raw"] = _to_json_safe(us)
        out["usage_source"] = "openai_usage"
        return out

    return out


def _should_use_responses_endpoint(llm: OpenAILLM) -> bool:
    model = str(getattr(llm, "model", "") or "").strip().lower()
    if _RESPONSES_FORCE_ENV in ("1", "true", "yes", "y", "on"):
        return True
    if _RESPONSES_FORCE_ENV in ("0", "false", "no", "n", "off"):
        return False
    return any(hint in model for hint in _RESPONSES_MODEL_HINTS)


def _extract_responses_text(response: Any) -> str:
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                ctype = getattr(c, "type", None)
                if ctype == "output_text":
                    ctext = getattr(c, "text", None)
                    if isinstance(ctext, str) and ctext:
                        chunks.append(ctext)
        if chunks:
            return "".join(chunks)

    raise RuntimeError("responses endpoint returned no textual output")


def _to_responses_params(chat_params: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": chat_params.get("model"),
        "input": chat_params.get("messages", []),
    }
    max_out = chat_params.get("max_completion_tokens", chat_params.get("max_tokens"))
    if max_out is not None:
        params["max_output_tokens"] = max_out

    reasoning_effort = chat_params.get("reasoning_effort")
    if reasoning_effort is not None:
        params["reasoning"] = {"effort": reasoning_effort}

    # Keep passthrough for compatibility when provided by caller/config.
    verbosity = chat_params.get("verbosity")
    if verbosity is not None:
        params["verbosity"] = verbosity
    service_tier = chat_params.get("service_tier")
    if service_tier is not None:
        params["service_tier"] = service_tier

    return params


async def _openai_call_api_with_responses_fallback(self: OpenAILLM, params: Dict[str, Any]) -> str:
    if self.client is None:
        raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

    # Default metadata (will be overwritten on successful call)
    self._qos_last_usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_prompt_tokens": None,
        "reasoning_tokens": None,
        "thinking_tokens": None,
        "usage_raw": None,
        "usage_source": None,
    }
    self._qos_last_model = str(getattr(self, "model", "") or "")
    self._qos_last_provider = "openai_compatible"
    if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
        self._qos_last_provider = "gemini_openai_compat"
    elif "anthropic" in str(self.api_base).lower():
        self._qos_last_provider = "claude_openai_compat"

    # Optional OpenAI Flex/Scale tier selection.
    # Only attach for native OpenAI provider path.
    if (
        _OPENAI_SERVICE_TIER
        and self._qos_last_provider == "openai_compatible"
        and "service_tier" not in params
    ):
        params["service_tier"] = _OPENAI_SERVICE_TIER

    if not _should_use_responses_endpoint(self):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        usage = _extract_usage_tokens(response)
        self._qos_last_usage = usage
        text = response.choices[0].message.content
        if usage.get("prompt_tokens") is None:
            # Fallback estimate if provider omits usage.
            msg_txt = "\n".join(
                str(m.get("content", "")) for m in params.get("messages", []) if isinstance(m, dict)
            )
            self._qos_last_usage["prompt_tokens"] = _estimate_tokens_from_text(msg_txt)
            self._qos_last_usage["completion_tokens"] = _estimate_tokens_from_text(text)
            self._qos_last_usage["total_tokens"] = (
                self._qos_last_usage["prompt_tokens"] + self._qos_last_usage["completion_tokens"]
            )
            self._qos_last_usage["usage_source"] = "heuristic_char_div4"
        self._qos_last_raw_response = {"usage": usage.get("usage_raw")}
        return text

    if not getattr(self, "_qos_responses_logged", False):
        LOGGER.info("OpenAI model '%s' will use /v1/responses endpoint", getattr(self, "model", ""))
        self._qos_responses_logged = True

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: self.client.responses.create(**_to_responses_params(params))
    )
    text = _extract_responses_text(response)
    usage = _extract_usage_tokens(response)
    if usage.get("prompt_tokens") is None:
        msg_txt = "\n".join(
            str(m.get("content", "")) for m in params.get("messages", []) if isinstance(m, dict)
        )
        usage["prompt_tokens"] = _estimate_tokens_from_text(msg_txt)
        usage["completion_tokens"] = _estimate_tokens_from_text(text)
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage["usage_source"] = "heuristic_char_div4"
    self._qos_last_usage = usage
    self._qos_last_raw_response = {"usage": usage.get("usage_raw")}
    return text


def _apply_openai_responses_patch() -> None:
    if getattr(OpenAILLM, "_qos_openai_responses_patched", False):
        return
    OpenAILLM._call_api = _openai_call_api_with_responses_fallback
    OpenAILLM._qos_openai_responses_patched = True


_apply_openai_responses_patch()


_ORIGINAL_ENSEMBLE_GEN_WITH_CONTEXT = LLMEnsemble.generate_with_context


async def _ensemble_generate_with_context_timed(
    self: LLMEnsemble, system_message: str, messages: List[Dict[str, str]], **kwargs: Any
) -> str:
    # Mirror original behavior while capturing provider/model, token usage, and LLM latency.
    model = self._sample_model()
    t0 = time.time()
    response = await model.generate_with_context(system_message, messages, **kwargs)
    dt = time.time() - t0
    usage = getattr(model, "_qos_last_usage", {}) or {}
    if usage.get("prompt_tokens") is None:
        # Try provider-native raw payloads if model exposes them.
        for attr in ("_qos_last_raw_response", "last_raw_response", "raw_response", "last_response"):
            raw = getattr(model, attr, None)
            parsed = _extract_usage_from_raw_payload(raw if isinstance(raw, dict) else {})
            if parsed.get("prompt_tokens") is not None or parsed.get("completion_tokens") is not None:
                usage = parsed
                break
    self._qos_last_llm_meta = {
        "llm_time_sec": float(dt),
        "provider": getattr(model, "_qos_last_provider", "unknown"),
        "model": getattr(model, "_qos_last_model", getattr(model, "model", "")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_prompt_tokens": usage.get("cached_prompt_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "thinking_tokens": usage.get("thinking_tokens"),
        "usage_raw": _to_json_safe(usage.get("usage_raw")),
        "token_usage_source": usage.get("usage_source"),
    }
    return response


def _apply_ensemble_timing_patch() -> None:
    if getattr(LLMEnsemble, "_qos_timing_patched", False):
        return
    LLMEnsemble.generate_with_context = _ensemble_generate_with_context_timed
    LLMEnsemble._qos_timing_patched = True


_apply_ensemble_timing_patch()


# Re-wrap evaluator patch to include timing while preserving failure guardrail behavior.
async def _evaluate_program_with_guardrail(self: Evaluator, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    t0 = time.time()
    metrics = await _ORIGINAL_EVALUATE_PROGRAM(self, *args, **kwargs)
    self._qos_last_eval_time_sec = float(time.time() - t0)
    if isinstance(metrics, dict):
        return _normalize_failed_metrics(metrics)
    return metrics


Evaluator.evaluate_program = _evaluate_program_with_guardrail


_ORIGINAL_RUN_ITERATION_FUNC = _ORIGINAL_RUN_ITERATION


async def _run_iteration_with_stats(
    iteration: int,
    config: Any,
    database: Any,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    result = await _ORIGINAL_RUN_ITERATION_FUNC(
        iteration, config, database, evaluator, llm_ensemble, prompt_sampler
    )
    if result is None:
        return result

    llm_meta = getattr(llm_ensemble, "_qos_last_llm_meta", {}) or {}
    eval_time = float(getattr(evaluator, "_qos_last_eval_time_sec", 0.0) or 0.0)

    # Fallback token estimate if provider-level usage was unavailable.
    prompt_text = ""
    if isinstance(result.prompt, dict):
        prompt_text = f"{result.prompt.get('system', '')}\n{result.prompt.get('user', '')}"
    output_text = result.llm_response if isinstance(result.llm_response, str) else ""
    if llm_meta.get("prompt_tokens") is None:
        llm_meta["prompt_tokens"] = _estimate_tokens_from_text(prompt_text)
        llm_meta["output_tokens"] = _estimate_tokens_from_text(output_text)
        llm_meta["total_tokens"] = llm_meta["prompt_tokens"] + llm_meta["output_tokens"]
        llm_meta["token_usage_source"] = "heuristic_char_div4"

    cost = _estimate_cost_usd(
        prompt_tokens=llm_meta.get("prompt_tokens"),
        output_tokens=llm_meta.get("output_tokens"),
        model=str(llm_meta.get("model") or ""),
        provider=str(llm_meta.get("provider") or ""),
    )

    stats = {
        "input_prompt_tokens": llm_meta.get("prompt_tokens"),
        "output_response_tokens": llm_meta.get("output_tokens"),
        "prompt_tokens": llm_meta.get("prompt_tokens"),
        "output_tokens": llm_meta.get("output_tokens"),
        "total_tokens": llm_meta.get("total_tokens"),
        "cached_prompt_tokens": llm_meta.get("cached_prompt_tokens"),
        "reasoning_tokens": llm_meta.get("reasoning_tokens"),
        "thinking_tokens": llm_meta.get("thinking_tokens"),
        "usage_raw": _to_json_safe(llm_meta.get("usage_raw")),
        "token_usage_source": llm_meta.get("token_usage_source"),
        "llm_time_sec": llm_meta.get("llm_time_sec"),
        "evaluation_time_sec": eval_time,
        "model": llm_meta.get("model"),
        "provider": llm_meta.get("provider"),
        **cost,
    }

    # Attach to artifacts so it appears in evolution_trace.jsonl entry per iteration.
    if not isinstance(result.artifacts, dict):
        result.artifacts = {}
    result.artifacts["qos_iteration_stats_json"] = stats
    return result


def _apply_iteration_stats_patch() -> None:
    import openevolve.iteration as it_mod

    if getattr(it_mod, "_qos_iteration_stats_patched", False):
        return
    it_mod.run_iteration_with_shared_db = _run_iteration_with_stats
    it_mod._qos_iteration_stats_patched = True


_apply_iteration_stats_patch()


_QOS_ORIGINAL_PP_WORKER = None


def _qos_run_iteration_worker_with_stats(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
):
    """
    Pickle-safe wrapper for openevolve.process_parallel._run_iteration_worker.

    Must be module-top-level so ProcessPool can pickle and ship it to workers.
    """
    global _QOS_ORIGINAL_PP_WORKER
    if _QOS_ORIGINAL_PP_WORKER is None:
        raise RuntimeError("_QOS_ORIGINAL_PP_WORKER is not initialized")

    result = _QOS_ORIGINAL_PP_WORKER(iteration, db_snapshot, parent_id, inspiration_ids)
    try:
        if result is None or getattr(result, "error", None):
            return result

        artifacts = getattr(result, "artifacts", None)
        if not isinstance(artifacts, dict):
            artifacts = {}
            result.artifacts = artifacts
        if "qos_iteration_stats_json" in artifacts:
            return result

        import openevolve.process_parallel as pp_mod

        llm_meta: Dict[str, Any] = {}
        eval_time = 0.0

        worker_ensemble = getattr(pp_mod, "_worker_llm_ensemble", None)
        if worker_ensemble is not None:
            llm_meta = getattr(worker_ensemble, "_qos_last_llm_meta", {}) or {}

        worker_evaluator = getattr(pp_mod, "_worker_evaluator", None)
        if worker_evaluator is not None:
            eval_time = float(getattr(worker_evaluator, "_qos_last_eval_time_sec", 0.0) or 0.0)

        prompt_text = ""
        prompt_obj = getattr(result, "prompt", None)
        if isinstance(prompt_obj, dict):
            prompt_text = f"{prompt_obj.get('system', '')}\n{prompt_obj.get('user', '')}"
        output_text = getattr(result, "llm_response", "") or ""

        prompt_tokens = llm_meta.get("prompt_tokens")
        output_tokens = llm_meta.get("output_tokens")
        total_tokens = llm_meta.get("total_tokens")
        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens_from_text(prompt_text)
            output_tokens = _estimate_tokens_from_text(output_text)
            total_tokens = prompt_tokens + output_tokens
            llm_meta["token_usage_source"] = "heuristic_char_div4"

        cost = _estimate_cost_usd(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=str(llm_meta.get("model") or ""),
            provider=str(llm_meta.get("provider") or ""),
        )

        artifacts["qos_iteration_stats_json"] = {
            "input_prompt_tokens": prompt_tokens,
            "output_response_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": llm_meta.get("cached_prompt_tokens"),
            "reasoning_tokens": llm_meta.get("reasoning_tokens"),
            "thinking_tokens": llm_meta.get("thinking_tokens"),
            "usage_raw": _to_json_safe(llm_meta.get("usage_raw")),
            "token_usage_source": llm_meta.get("token_usage_source"),
            "llm_time_sec": llm_meta.get("llm_time_sec"),
            "evaluation_time_sec": eval_time,
            "model": llm_meta.get("model"),
            "provider": llm_meta.get("provider"),
            **cost,
        }
    except Exception:
        # Non-critical telemetry path: never fail iteration due to stats collection.
        pass
    return result


def _apply_process_parallel_worker_stats_patch() -> None:
    """
    Patch process-parallel worker path so iteration stats are persisted in artifacts.

    OpenEvolve commonly executes iterations through openevolve.process_parallel._run_iteration_worker,
    which bypasses openevolve.iteration.run_iteration_with_shared_db. We patch the worker entrypoint
    to attach qos_iteration_stats_json so evolution_trace.jsonl always records token/time/cost.
    """
    import openevolve.process_parallel as pp_mod

    if getattr(pp_mod, "_qos_worker_stats_patched", False):
        return

    global _QOS_ORIGINAL_PP_WORKER
    _QOS_ORIGINAL_PP_WORKER = pp_mod._run_iteration_worker
    pp_mod._run_iteration_worker = _qos_run_iteration_worker_with_stats
    pp_mod._qos_worker_stats_patched = True


_apply_process_parallel_worker_stats_patch()


def main() -> int:
    return openevolve_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
