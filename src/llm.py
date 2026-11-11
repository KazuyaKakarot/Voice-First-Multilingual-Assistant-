"""Thin LLM wrapper: any OpenAI-compatible API, disk-cached, with retries.

Caching matters on free tiers: identical prompts (temperature 0) are never
paid for twice, so re-running experiments is free. Turn it off with set_cache_enabled(False)
when you want honest latency numbers.
"""
import base64
import hashlib
import json

from diskcache import Cache
import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import config

_client = None
_cache = None
_cache_enabled = True

# Counts every LLM request (cached or not). Differences between two readings give "LLM calls per
# question", which is the cost proxy reported by the evaluation harness.
stats = {"calls": 0, "cache_hits": 0}


def set_cache_enabled(enabled: bool) -> None:
    global _cache_enabled
    _cache_enabled = enabled


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is empty. Copy .env.example to .env and set it.")
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache(config.CACHE_DIR)
    return _cache


# Only transient problems are worth retrying. A wrong model name or a bad key will never fix itself.
TRANSIENT = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


@retry(retry=retry_if_exception_type(TRANSIENT), stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def _request(model: str, messages: list[dict]) -> str:
    resp = _get_client().chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content or ""


def _call(model: str, messages: list[dict]) -> str:
    try:
        return _request(model, messages)
    except openai.NotFoundError as e:
        raise RuntimeError(f"The model '{model}' does not exist or your key cannot use it. Providers retire models: "
                           f"check the provider's current model list and fix LLM_MODEL / VISION_MODEL in .env. ({e})") from None
    except openai.AuthenticationError as e:
        raise RuntimeError(f"The LLM API key was rejected. Check LLM_API_KEY and LLM_BASE_URL in .env. ({e})") from None


def _cached(model: str, messages: list[dict], key_extra: str = "") -> str:
    stats["calls"] += 1
    key = hashlib.sha256((json.dumps([model, messages]) + key_extra).encode()).hexdigest()
    if _cache_enabled:
        cache = _get_cache()
        if key in cache:
            stats["cache_hits"] += 1
            return cache[key]
    out = _call(model, messages)
    if _cache_enabled:
        _get_cache()[key] = out
    return out


def chat(system: str, user: str, model: str | None = None) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _cached(model or config.LLM_MODEL, messages)


def chat_vision(system: str, user: str, image: bytes, mime: str = "image/jpeg") -> str:
    """Multimodal call: an image plus an instruction, via an OpenAI-compatible vision model."""
    data_uri = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]},
    ]
    # The data URI is huge; key the cache on the image hash instead of storing it twice.
    short = [messages[0], {"role": "user", "content": user}]
    return _cached(config.VISION_MODEL, messages, key_extra=json.dumps(short) + hashlib.sha256(image).hexdigest())


def parse_json(raw: str, default: dict) -> dict:
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return default


def chat_json(system: str, user: str, default: dict, model: str | None = None) -> dict:
    """Ask for JSON; fall back to `default` if the model returns something unparseable."""
    raw = chat(system + "\nRespond with a single valid JSON object and nothing else.", user, model=model)
    return parse_json(raw, default)
