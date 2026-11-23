"""Thin client for AssemblyAI's LLM Gateway (OpenAI-compatible chat completions).

Design choices:
- Structured outputs (``response_format`` + strict JSON schema) so the voice loop
  never has to guess at the shape of a reply, plus the Gateway's ``json-repair``
  post-processing step as a safety net.
- A model chain, not a single model. Models are tried in order. A model this
  account cannot call (most Gateway models need a paid AssemblyAI account) is
  remembered and skipped, so only the first request pays for finding out, and
  the startup probe in main.py usually means no learner turn pays at all.
- Models without ``response_format`` get a prompt-only JSON request with the
  schema written into the system prompt. Every reply is checked against the
  schema's required keys and the gaps are returned in ``meta["missing"]`` so the
  caller can repair them deterministically instead of trusting the model.
A tutor that stalls mid-lesson is worse than a slightly slower answer.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger("lumi.llm")

# Substrings of the Gateway's error bodies that decide what happens next.
NO_ACCESS = ("does not have access",)
NO_SCHEMA = ("response_format",)
UNKNOWN_MODEL = ("model not found", "unknown model", "invalid model", "does not exist", "not a valid model")

PAID_ACCOUNT_HINT = (
    "Most LLM Gateway models need a paid AssemblyAI account (add a card in the dashboard; "
    "unused free credits carry over). Or set LLM_MODEL / LLM_FALLBACK_MODELS to models this key can call."
)

JSON_ONLY_INSTRUCTION = (
    "Output format: reply with ONE JSON object and nothing else - no prose, no markdown, no code fences. "
    "Include every required key; use empty strings or empty lists when a value does not apply. "
    "It must validate against this JSON Schema:\n{schema}"
)


class LLMGatewayError(RuntimeError):
    pass


def _mentions(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(n in lowered for n in needles)


def parse_json_loose(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating code fences or chatter."""
    if not text:
        return None
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def missing_required(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Paths of required keys that are absent, or present with the wrong container type."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [path or "<root>"]
        props = schema.get("properties", {})
        out: list[str] = []
        for key in schema.get("required", []):
            child = f"{path}.{key}" if path else key
            if key not in value:
                out.append(child)
            else:
                out.extend(missing_required(value[key], props.get(key, {}), child))
        return out
    if kind == "array":
        if not isinstance(value, list):
            return [path or "<root>"]
        out = []
        for i, item in enumerate(value):
            out.extend(missing_required(item, schema.get("items", {}), f"{path}[{i}]"))
        return out
    return []


def fill_defaults(value: Any, schema: dict[str, Any]) -> Any:
    """Return ``value`` with every missing required key filled with an empty value of the right type."""
    kind = schema.get("type")
    if kind == "object":
        filled = dict(value) if isinstance(value, dict) else {}
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            filled[key] = fill_defaults(filled.get(key), props.get(key, {}))
        return filled
    if kind == "array":
        return [fill_defaults(v, schema.get("items", {})) for v in value] if isinstance(value, list) else []
    if value is not None:
        return value
    return {"string": "", "integer": 0, "number": 0, "boolean": False}.get(kind)


def with_json_instruction(messages: list[dict[str, str]], schema: dict[str, Any]) -> list[dict[str, str]]:
    """Spell the schema out in the system prompt, for models that cannot take ``response_format``."""
    note = JSON_ONLY_INSTRUCTION.format(schema=json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f'{out[0]["content"]}\n\n{note}'
    else:
        out.insert(0, {"role": "system", "content": note})
    return out


class LLMGateway:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, retry_unavailable_after_s: float = 600):
        self.settings = settings
        self.client = client
        self.retry_unavailable_after_s = retry_unavailable_after_s
        self.unavailable: dict[str, tuple[str, float]] = {}  # model -> (reason, when)
        self.no_schema: set[str] = set()  # models that only take prompt-only JSON
        self.active_model: str | None = None  # last model that answered

    # ------------------------------------------------------------ model chain
    @property
    def models(self) -> list[str]:
        seen: list[str] = []
        for model in (self.settings.llm_model, *self.settings.llm_fallback_models):
            if model and model not in seen:
                seen.append(model)
        return seen

    @property
    def usable_models(self) -> list[str]:
        """The chain minus models known to be unusable (re-checked after a while, e.g. after an upgrade)."""
        now = time.monotonic()
        return [
            m for m in self.models
            if m not in self.unavailable or now - self.unavailable[m][1] >= self.retry_unavailable_after_s
        ]

    def mark_unavailable(self, model: str, reason: str) -> None:
        if model not in self.unavailable:
            log.warning("LLM model %s skipped: %s", model, reason)
        self.unavailable[model] = (reason, time.monotonic())

    def status(self) -> dict[str, Any]:
        return {
            "chain": self.models,
            "active": self.active_model,
            "unavailable": {m: why for m, (why, _) in self.unavailable.items()},
            "prompt_only_json": [m for m in self.models if m in self.no_schema],
        }

    def _nothing_usable(self) -> str:
        reasons = "; ".join(f"{m}: {why}" for m, (why, _) in self.unavailable.items()) or "none configured"
        return f"No usable LLM Gateway model ({reasons}). {PAID_ACCOUNT_HINT}"

    # ---------------------------------------------------------------- catalog
    async def catalog(self) -> list[dict[str, Any]]:
        """The public models endpoint: ids, supported parameters and regions (no auth needed)."""
        resp = await self.client.get(f"{self.settings.llm_base}/models", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data", payload) if isinstance(payload, dict) else payload
        out: list[dict[str, Any]] = []
        for item in items or []:
            if isinstance(item, dict) and (item.get("id") or item.get("model") or item.get("name")):
                out.append({**item, "id": str(item.get("id") or item.get("model") or item.get("name"))})
            elif isinstance(item, str):
                out.append({"id": item})
        return out

    async def list_models(self) -> list[str]:
        return [m["id"] for m in await self.catalog()]

    def apply_catalog(self, catalog: list[dict[str, Any]]) -> None:
        """Learn from the catalog which models need prompt-only JSON or are not served in our region."""
        by_id = {m["id"]: m for m in catalog}
        for model in self.models:
            info = by_id.get(model)
            if not info:
                continue
            params = info.get("supported_parameters")
            if isinstance(params, list) and "response_format" not in params:
                self.no_schema.add(model)
            regions = info.get("available_regions")
            if self.settings.is_eu and isinstance(regions, list) and "eu" not in regions:
                self.mark_unavailable(model, "not served from the EU region")

    async def probe(self) -> str | None:
        """Find the first model in the chain this key can call (a few tokens each). Returns its id."""
        if not self.settings.api_key:
            return None
        for model in self.usable_models:
            try:
                resp = await self.client.post(
                    f"{self.settings.llm_base}/chat/completions",
                    headers={"authorization": self.settings.api_key},
                    json={"model": model, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 8},
                    timeout=15,
                )
            except httpx.HTTPError as exc:
                log.info("LLM probe could not reach the Gateway (%r)", exc)
                return None
            if resp.status_code == 200:
                self.active_model = model
                return model
            if _mentions(resp.text, NO_ACCESS):
                self.mark_unavailable(model, "this account has no access to it")
            elif resp.status_code in (401, 403):
                log.warning("LLM probe: API key rejected (%s)", resp.status_code)
                return None
            else:
                log.info("LLM probe: %s answered HTTP %s; will still try it", model, resp.status_code)
        return None

    # ------------------------------------------------------------------- chat
    async def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 500,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(parsed_reply, meta)``; meta records model, latency, mode and any missing keys."""
        if not self.settings.api_key:
            raise LLMGatewayError("ASSEMBLYAI_API_KEY is not set")
        candidates = self.usable_models
        if not candidates:
            raise LLMGatewayError(self._nothing_usable())

        headers = {"authorization": self.settings.api_key, "content-type": "application/json"}
        last_error = "no models configured"

        for model in candidates:
            schema_rejected = False
            for structured in ((False,) if model in self.no_schema else (True, False)):
                body: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
                if structured:
                    body["messages"] = messages
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                    }
                    body["post_processing_steps"] = [{"type": "json-repair"}]
                else:
                    body["messages"] = with_json_instruction(messages, schema)

                started = time.perf_counter()
                try:
                    resp = await self.client.post(
                        f"{self.settings.llm_base}/chat/completions", headers=headers, json=body, timeout=25
                    )
                except httpx.HTTPError as exc:
                    last_error = f"{model}: network error {exc!r}"
                    log.warning(last_error)
                    break  # try the next model
                elapsed_ms = round((time.perf_counter() - started) * 1000)

                if resp.status_code == 200:
                    try:
                        content = resp.json()["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError, ValueError):
                        content = ""
                    parsed = parse_json_loose(content or "")
                    if parsed is not None:
                        if schema_rejected:  # the schema, not the request, was the problem
                            self.no_schema.add(model)
                        self.active_model = model
                        missing = missing_required(parsed, schema)
                        if missing:
                            log.info("%s reply is missing %s", model, ", ".join(missing))
                        return parsed, {"model": model, "llm_ms": elapsed_ms, "structured": structured, "missing": missing}
                    last_error = f"{model}: reply was not valid JSON"
                    log.warning(last_error)
                    continue  # structured -> retry this model prompt-only; prompt-only -> next model

                detail = resp.text[:300]
                if _mentions(detail, NO_ACCESS):
                    self.mark_unavailable(model, "this account has no access to it")
                    last_error = f"{model}: no access for this account"
                    break  # retrying without a schema would fail the same way
                if resp.status_code in (401, 403):
                    raise LLMGatewayError(f"LLM Gateway rejected the API key ({resp.status_code}): {detail[:200]}")

                last_error = f"{model}: HTTP {resp.status_code} {detail}"
                log.warning(last_error)
                if structured and resp.status_code in (400, 422):
                    schema_rejected = True
                    if _mentions(detail, NO_SCHEMA):
                        self.no_schema.add(model)
                    continue  # schema dialects differ between providers - retry this model without one
                if resp.status_code in (400, 404) and _mentions(detail, UNKNOWN_MODEL):
                    self.mark_unavailable(model, "the Gateway does not know this model id")
                break  # rate limit, server error... try the next model

        if not self.usable_models:
            raise LLMGatewayError(self._nothing_usable())
        raise LLMGatewayError(last_error)
