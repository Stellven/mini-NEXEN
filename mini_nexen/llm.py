from __future__ import annotations

import json
import os
import sys
import time
import re
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from .config import TASK_LOG_PATH, ensure_dirs

_ECHO_LOG = False
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_LINE: str | None = None
_PROGRESS_ACTIVE = False
_RETRY_NOTICE: dict[str, str] | None = None
_PROGRESS_LAST_PERCENT: dict[tuple[str, str], int] = {}
BACKOFF_INITIAL_SECONDS = 2
BACKOFF_MAX_SECONDS = 30
BACKOFF_TOTAL_MAX_SECONDS = 180
TIMEOUT_RETRY_DELAY_SECONDS = 1
TIMEOUT_RETRY_MULTIPLIER = 0.5
MAX_RETRY_ATTEMPTS = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 60
_HTTP_CODE_RE = re.compile(r"\b([45]\d{2})\b")

TASK_TIMEOUT_OVERRIDES: dict[str, float] = {
    "outline": 300,
    "outline revision": 240,
    "outline expansion": 240,
    "plan draft": 150,
    "plan refinement": 150,
    "profile signals": 120,
    "profile extraction": 120,
    "kg triples": 120,
}


def set_log_echo(enabled: bool = True) -> None:
    global _ECHO_LOG
    _ECHO_LOG = enabled


def _should_echo(line: str) -> bool:
    lowered = line.lower()
    return "raw response" not in lowered


def _clear_progress_locked() -> None:
    if not sys.stdout.isatty():
        return
    print("\r\033[2K", end="")


def _render_progress_locked() -> None:
    if not sys.stdout.isatty() or not _PROGRESS_LINE:
        return
    print("\r" + _PROGRESS_LINE + " " * 6, end="", flush=True)


def update_progress_line(line: str, *, done: bool = False) -> None:
    if not sys.stdout.isatty():
        return
    global _PROGRESS_LINE, _PROGRESS_ACTIVE
    with _PROGRESS_LOCK:
        _PROGRESS_LINE = line
        _PROGRESS_ACTIVE = not done
        _render_progress_locked()
        if done:
            _PROGRESS_LINE = None
            print("", flush=True)

def note_retry_notice(agent: str, task: str, wait_seconds: float, category: str) -> None:
    if category != "rate_limit":
        return
    suffix = f"(rate limit, retrying in {int(wait_seconds)}s)"
    with _PROGRESS_LOCK:
        global _RETRY_NOTICE
        _RETRY_NOTICE = {
            "agent": agent,
            "task": task,
            "suffix": suffix,
        }


def consume_retry_notice(agent: str, task: str) -> str | None:
    with _PROGRESS_LOCK:
        global _RETRY_NOTICE
        if not _RETRY_NOTICE:
            return None
        if _RETRY_NOTICE.get("agent") != agent or _RETRY_NOTICE.get("task") != task:
            return None
        suffix = _RETRY_NOTICE.get("suffix")
        _RETRY_NOTICE = None
        return suffix


def has_retry_notice(agent: str, task: str) -> bool:
    with _PROGRESS_LOCK:
        if not _RETRY_NOTICE:
            return False
        return _RETRY_NOTICE.get("agent") == agent and _RETRY_NOTICE.get("task") == task


def format_progress_line(
    agent: str,
    model: str,
    task: str,
    current: int,
    total: int,
    percent: int | None = None,
) -> str:
    percent = percent if percent is not None else int((current / total) * 100) if total else 0
    stamp = datetime.now(timezone.utc).isoformat()
    label = f"{stamp} | {agent} | Model {model} is generating {task}... ({percent}%)"
    suffix = consume_retry_notice(agent, task)
    if suffix:
        label = f"{label} {suffix}"
    return label


def resolve_task_timeout(task: str, base_timeout: float) -> float:
    override = TASK_TIMEOUT_OVERRIDES.get((task or "").casefold())
    if override:
        return max(float(base_timeout), float(override))
    return float(base_timeout)


def emit_progress(
    agent: str,
    model: str,
    task: str,
    current: int,
    total: int,
    *,
    done: bool = False,
) -> None:
    if total <= 0:
        return
    percent = int((current / total) * 100)
    key = (agent, task)
    last = _PROGRESS_LAST_PERCENT.get(key)
    if last == percent and not done and not has_retry_notice(agent, task):
        return
    line = format_progress_line(agent, model, task, current, total, percent=percent)
    update_progress_line(line, done=done)
    if done:
        _PROGRESS_LAST_PERCENT.pop(key, None)
    else:
        _PROGRESS_LAST_PERCENT[key] = percent


def _write_log_line(line: str, echo: bool = True) -> None:
    ensure_dirs()
    with TASK_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if echo and _ECHO_LOG and _should_echo(line):
        with _PROGRESS_LOCK:
            if _PROGRESS_ACTIVE and sys.stdout.isatty():
                _clear_progress_locked()
            print(line, flush=True)
            if _PROGRESS_ACTIVE:
                _render_progress_locked()

@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    discover_model: bool = True


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _log(self, agent: str, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        message = f"{stamp} | {agent} | {message}"
        _write_log_line(message)

    def _should_log_completion(self, agent: str, task: str) -> bool:
        if agent == "KGExtractor" and task == "kg triples":
            return False
        if agent == "Profiler" and task == "profile extraction":
            return False
        return True

    def _should_log_start(self, agent: str, task: str) -> bool:
        if agent == "KGExtractor" and task == "kg triples":
            return False
        if agent == "Profiler" and task == "profile extraction":
            return False
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "content",
        agent: str = "Agent",
    ) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise LLMClientError(
                "google-genai is not installed. Install it to use Gemini models."
            ) from exc
        self._types = types

        api_key = config.api_key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "content",
        agent: str = "Agent",
    ) -> str:
        attempt = 0
        timeout_attempts = 0
        rate_limit_wait = 0.0
        effective_timeout = resolve_task_timeout(task, self.config.timeout)
        def _classify_error(message: str) -> str:
            lowered = message.lower()
            if "timeout" in lowered or "timed out" in lowered:
                return "timeout"
            if "429" in message or "rate" in lowered or "resourceexhausted" in lowered:
                return "rate_limit"
            match = _HTTP_CODE_RE.search(message)
            if match:
                return f"http_{match.group(1)}"
            return "error"

        def _log_failure(category: str, exc: Exception, detail: str | None = None) -> None:
            info = detail if detail is not None else str(exc)
            if category == "timeout":
                return
            if category == "rate_limit":
                self._log(
                    agent,
                    f"Model {self.config.model} failed {task}: {category} ({info})",
                )
                return
            self._log(
                agent,
                f"Model {self.config.model} failed {task}: {category} ({type(exc).__name__}: {info})",
            )

        def _generate_with_timeout(timeout_seconds: float) -> str:
            result: dict[str, str] = {}
            error: dict[str, Exception] = {}

            def _runner() -> None:
                try:
                    response = self.client.models.generate_content(
                        model=self.config.model,
                        contents=user_prompt,
                        config=self._types.GenerateContentConfig(system_instruction=system_prompt),
                    )
                    result["text"] = getattr(response, "text", "") or ""
                except Exception as exc:
                    error["exc"] = exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(f"Gemini request timeout after {timeout_seconds}s")
            if "exc" in error:
                raise error["exc"]
            return result.get("text", "")
        while True:
            attempt += 1
            if attempt == 1:
                if self._should_log_start(agent, task):
                    self._log(agent, f"Model {self.config.model} is generating {task}...")
            try:
                attempt_timeout = effective_timeout * (1 + TIMEOUT_RETRY_MULTIPLIER * timeout_attempts)
                text = _generate_with_timeout(attempt_timeout)
                if self._should_log_completion(agent, task):
                    self._log(agent, f"Model {self.config.model} finished generating {task}.")
                return text
            except Exception as exc:  # pragma: no cover - provider-specific errors
                message = str(exc)
                category = _classify_error(message)
                if category in {"rate_limit", "timeout"}:
                    _log_failure(category, exc, detail=message)
                    if attempt >= MAX_RETRY_ATTEMPTS:
                        raise LLMClientError(f"{category} retry limit exceeded")
                    if category == "timeout":
                        timeout_attempts += 1
                        sleep_for = TIMEOUT_RETRY_DELAY_SECONDS
                        self._log(
                            agent,
                            f"{self.config.model} timeout on {task}; "
                            f"retrying (attempt {attempt}) in {sleep_for}s.",
                        )
                    else:
                        sleep_for = min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * (2 ** (attempt - 1)))
                        if rate_limit_wait + sleep_for > BACKOFF_TOTAL_MAX_SECONDS:
                            raise LLMClientError(f"{category} retry budget exceeded")
                        note_retry_notice(agent, task, sleep_for, category)
                        self._log(
                            agent,
                            f"Model {self.config.model} {category} on {task}; "
                            f"retrying (attempt {attempt}) in {sleep_for}s.",
                        )
                        rate_limit_wait += sleep_for
                    time.sleep(sleep_for)
                    continue
                _log_failure(category, exc, detail=message)
                raise LLMClientError(message) from exc


class LMStudioClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise LLMClientError("requests is required for LM Studio usage.") from exc
        self._requests = requests

    def _resolve_model(self, agent: str, base_url: str) -> None:
        if not self.config.discover_model:
            return
        if self.config.model != "your-local-model":
            return
        url = base_url.rstrip("/") + "/models"
        try:
            response = self._requests.get(url, timeout=10)
            if response.status_code >= 400:
                self._log(agent, f"LM Studio model discovery failed: {response.text}")
                return
            data = response.json()
            models = data.get("data") if isinstance(data, dict) else None
            if isinstance(models, list) and models:
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    model_id = item.get("id")
                    if not model_id:
                        continue
                    lowered = str(model_id).lower()
                    if "embed" in lowered or "embedding" in lowered:
                        continue
                    self.config.model = model_id
                    self._log(agent, f"LM Studio resolved model to {model_id}.")
                    return
                self._log(agent, "LM Studio returned only embedding models; using configured model.")
                return
            self._log(agent, "LM Studio returned no models; using configured model.")
        except Exception as exc:
            self._log(agent, f"LM Studio model discovery error: {exc}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "content",
        agent: str = "Agent",
    ) -> str:
        base_url = self.config.base_url or "http://localhost:1234/v1"
        self._resolve_model(agent, base_url)
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        attempt = 0
        timeout_attempts = 0
        rate_limit_wait = 0.0
        effective_timeout = resolve_task_timeout(task, self.config.timeout)
        while True:
            attempt += 1
            if attempt == 1:
                if self._should_log_start(agent, task):
                    self._log(agent, f"Model {self.config.model} is generating {task}...")
            try:
                attempt_timeout = effective_timeout * (1 + TIMEOUT_RETRY_MULTIPLIER * timeout_attempts)
                response = self._requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=attempt_timeout,
                )
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                category = "timeout" if "timeout" in lowered or "timed out" in lowered else "error"
                if category == "timeout":
                    if attempt >= MAX_RETRY_ATTEMPTS:
                        raise LLMClientError("timeout retry limit exceeded") from exc
                    timeout_attempts += 1
                    sleep_for = TIMEOUT_RETRY_DELAY_SECONDS
                    self._log(
                        agent,
                        f"{self.config.model} timeout on {task}; "
                        f"retrying (attempt {attempt}) in {sleep_for}s.",
                    )
                    time.sleep(sleep_for)
                    continue
                self._log(agent, f"Model {self.config.model} failed {task}: {category} ({message})")
                raise LLMClientError(message) from exc
            if response.status_code == 429:
                self._log(agent, f"Model {self.config.model} rate limit on {task}.")
                sleep_for = min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * (2 ** (attempt - 1)))
                if attempt >= MAX_RETRY_ATTEMPTS:
                    raise LLMClientError("rate limit retry limit exceeded")
                if rate_limit_wait + sleep_for > BACKOFF_TOTAL_MAX_SECONDS:
                    raise LLMClientError("rate limit retry budget exceeded")
                note_retry_notice(agent, task, sleep_for, "rate_limit")
                self._log(
                    agent,
                    f"Model {self.config.model} rate limit on {task}; "
                    f"retrying (attempt {attempt}) in {sleep_for}s.",
                )
                time.sleep(sleep_for)
                rate_limit_wait += sleep_for
                continue
            if response.status_code >= 400:
                self._log(agent, f"Model {self.config.model} failed {task}: {response.text}")
                raise LLMClientError(f"LM Studio error {response.status_code}: {response.text}")
            data = response.json()
            try:
                if isinstance(data, dict):
                    actual_model = data.get("model")
                    if actual_model and actual_model != self.config.model:
                        self.config.model = actual_model
                        self._log(agent, f"LM Studio reported model {actual_model}.")
                text = data["choices"][0]["message"]["content"]
                if self._should_log_completion(agent, task):
                    self._log(agent, f"Model {self.config.model} finished generating {task}.")
                return text
            except (KeyError, IndexError, TypeError) as exc:
                self._log(agent, f"Model {self.config.model} failed {task}: unexpected response")
                raise LLMClientError("Unexpected LM Studio response format") from exc


def log_task_event(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    _write_log_line(f"{stamp} | {message}")


def log_task_event_quiet(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    _write_log_line(f"{stamp} | {message}", echo=False)


def _load_timeout_env() -> Optional[float]:
    value = os.getenv("MINI_NEXEN_LLM_TIMEOUT")
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(1.0, parsed)


def load_llm_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    discover_model: Optional[bool] = None,
) -> Optional[LLMConfig]:
    provider = provider or os.getenv("MINI_NEXEN_PROVIDER")
    if not provider:
        return None

    provider = provider.lower().strip()
    model = model or os.getenv("MINI_NEXEN_MODEL")

    timeout_value = timeout or _load_timeout_env()
    if provider == "gemini":
        model = model or "gemini-2.5-flash"
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature or 0.2,
            max_tokens=max_tokens or 2048,
            timeout=timeout_value or DEFAULT_LLM_TIMEOUT_SECONDS,
            api_key=api_key,
            discover_model=True,
        )

    if provider == "lmstudio":
        model = model or "local-model"
        base_url = base_url or os.getenv("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1"
        api_key = api_key or os.getenv("LMSTUDIO_API_KEY")
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature or 0.2,
            max_tokens=max_tokens or 2048,
            timeout=timeout_value or DEFAULT_LLM_TIMEOUT_SECONDS,
            base_url=base_url,
            api_key=api_key,
            discover_model=bool(discover_model) if discover_model is not None else True,
        )

    raise LLMClientError(f"Unsupported provider: {provider}")


def build_client(config: Optional[LLMConfig]) -> Optional[LLMClient]:
    if not config:
        return None
    if config.provider == "gemini":
        return GeminiClient(config)
    if config.provider == "lmstudio":
        return LMStudioClient(config)
    raise LLMClientError(f"Unsupported provider: {config.provider}")
