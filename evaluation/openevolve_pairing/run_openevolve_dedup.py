#!/usr/bin/env python3
"""Run OpenEvolve with prompt-level code deduplication enabled."""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import threading
import time
from typing import Any, Dict, List, Set, Tuple

from openevolve import cli as openevolve_cli
from openevolve.evaluator import Evaluator
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


def main() -> int:
    return openevolve_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
