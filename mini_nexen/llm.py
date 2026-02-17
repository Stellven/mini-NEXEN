from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from .config import TASK_LOG_PATH, ensure_dirs

_ECHO_LOG = False
PLANNING_THROTTLE_BASE_SECONDS = 2
PLANNING_THROTTLE_MAX_SECONDS = 30
PLANNING_THROTTLE_IDLE_RESET_SECONDS = 60.0
PLANNING_RETRY_MAX_SECONDS = 90.0


def set_log_echo(enabled: bool = True) -> None:
    global _ECHO_LOG
    _ECHO_LOG = enabled


def _should_echo(line: str) -> bool:
    lowered = line.lower()
    return "raw response" not in lowered


def _write_log_line(line: str, echo: bool = True) -> None:
    ensure_dirs()
    with TASK_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if echo and _ECHO_LOG and _should_echo(line):
        print(line, flush=True)

@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
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
        self._planning_last_call = 0.0
        self._planning_call_count = 0

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
        self._maybe_throttle_planning(agent, task)
        use_time_budget = agent in {"Planner", "Outliner"}
        started_at = time.time()
        attempt = 0
        while True:
            attempt += 1
            if attempt == 1:
                self._log(agent, f"Model {self.config.model} is generating {task}...")
            else:
                if use_time_budget:
                    elapsed = time.time() - started_at
                    remaining = max(0.0, PLANNING_RETRY_MAX_SECONDS - elapsed)
                    self._log(
                        agent,
                        f"Model {self.config.model} retrying {task} (attempt {attempt}, "
                        f"{remaining:.0f}s budget remaining)."
                    )
                else:
                    self._log(
                        agent,
                        f"Model {self.config.model} retrying {task} (attempt {attempt}/3)."
                    )
            try:
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=user_prompt,
                    config=self._types.GenerateContentConfig(system_instruction=system_prompt),
                )
                text = getattr(response, "text", "") or ""
                self._log(agent, f"Model {self.config.model} finished generating {task}.")
                return text
            except Exception as exc:  # pragma: no cover - provider-specific errors
                message = str(exc)
                if "429" in message or "rate" in message.lower() or "resourceexhausted" in message.lower():
                    self._log(agent, f"Model {self.config.model} rate limit on {task}.")
                    if use_time_budget:
                        elapsed = time.time() - started_at
                        if elapsed >= PLANNING_RETRY_MAX_SECONDS:
                            raise LLMClientError("Planning retry budget exceeded")
                        sleep_for = 2**attempt
                        remaining = PLANNING_RETRY_MAX_SECONDS - elapsed
                        time.sleep(min(sleep_for, max(0.0, remaining)))
                        continue
                    if attempt < 3:
                        time.sleep(2**attempt)
                        continue
                    raise LLMClientError("Rate limit reached")
                self._log(agent, f"Model {self.config.model} failed {task}: {message}")
                raise LLMClientError(message) from exc

    def _maybe_throttle_planning(self, agent: str, task: str) -> None:
        if agent not in {"Planner", "Outliner"}:
            return
        now = time.time()
        if self._planning_last_call and now - self._planning_last_call > PLANNING_THROTTLE_IDLE_RESET_SECONDS:
            self._planning_call_count = 0
        if self._planning_last_call:
            self._planning_call_count += 1
            power = min(self._planning_call_count, 5)
            delay = min(PLANNING_THROTTLE_MAX_SECONDS, PLANNING_THROTTLE_BASE_SECONDS**power)
            self._log(agent, f"Planner throttle: sleeping {delay}s before {task}.")
            time.sleep(delay)
        self._planning_last_call = time.time()


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
        use_time_budget = agent in {"Planner", "Outliner"}
        started_at = time.time()
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
        while True:
            attempt += 1
            if attempt == 1:
                self._log(agent, f"Model {self.config.model} is generating {task}...")
            else:
                if use_time_budget:
                    elapsed = time.time() - started_at
                    remaining = max(0.0, PLANNING_RETRY_MAX_SECONDS - elapsed)
                    self._log(
                        agent,
                        f"Model {self.config.model} retrying {task} (attempt {attempt}, "
                        f"{remaining:.0f}s budget remaining)."
                    )
                else:
                    self._log(
                        agent,
                        f"Model {self.config.model} retrying {task} (attempt {attempt}/3)."
                    )
            response = self._requests.post(url, headers=headers, data=json.dumps(payload), timeout=1200)
            if response.status_code == 429:
                self._log(agent, f"Model {self.config.model} rate limit on {task}.")
                if use_time_budget:
                    elapsed = time.time() - started_at
                    if elapsed >= PLANNING_RETRY_MAX_SECONDS:
                        raise LLMClientError("Planning retry budget exceeded")
                    retry_after = response.headers.get("Retry-After")
                    sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    remaining = PLANNING_RETRY_MAX_SECONDS - elapsed
                    time.sleep(min(sleep_for, max(0.0, remaining)))
                    continue
                if attempt < 3:
                    retry_after = response.headers.get("Retry-After")
                    sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    time.sleep(sleep_for)
                    continue
                raise LLMClientError("Rate limit reached")
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


def load_llm_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    discover_model: Optional[bool] = None,
) -> Optional[LLMConfig]:
    provider = provider or os.getenv("MINI_NEXEN_PROVIDER")
    if not provider:
        return None

    provider = provider.lower().strip()
    model = model or os.getenv("MINI_NEXEN_MODEL")

    if provider == "gemini":
        model = model or "gemini-2.5-flash"
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature or 0.2,
            max_tokens=max_tokens or 2048,
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
