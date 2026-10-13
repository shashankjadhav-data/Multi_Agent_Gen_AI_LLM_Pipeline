import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from utils.logger import logger

class LLMClient:

    def __init__(self):
        from config import settings
        self._cfg = settings

        self._ollama_client = None
        self._anthropic_client = None
        self._anthropic_async_client = None
        self._hf_client = None
        self._hf_async_client = None
        self._groq_client = None
        self._groq_async_client = None
        self._nvidia_client = None
        self._nvidia_async_client = None

        default_backend  = getattr(self._cfg, "LLM_BACKEND", "ollama")
        has_anthropic_key = bool(getattr(self._cfg, "ANTHROPIC_API_KEY", ""))
        has_hf_token       = bool(getattr(self._cfg, "HUGGINGFACE_API_TOKEN", ""))
        has_groq_key       = bool(getattr(self._cfg, "GROQ_API_KEY", ""))
        has_nvidia_key     = bool(getattr(self._cfg, "NVIDIA_API_KEY", ""))

        skip_ollama = (
            (default_backend == "anthropic" and has_anthropic_key) or
            (default_backend == "huggingface" and has_hf_token) or
            (default_backend == "groq" and has_groq_key) or
            (default_backend == "nvidia" and has_nvidia_key)
        )
        if not skip_ollama:
            self._ollama_client = self._build_ollama_client()
            if self._ollama_client:
                self._check_models()

        if has_anthropic_key:
            self._init_anthropic()
        if has_hf_token:
            self._init_huggingface()
        if has_groq_key:
            self._init_groq()
        if has_nvidia_key:
            self._init_nvidia()

    def _build_ollama_client(self):
        try:
            import ollama
            return ollama.Client(host=self._cfg.OLLAMA_BASE_URL)
        except Exception as exc:
            logger.warning(
                f"[LLM] Cannot connect to Ollama at {self._cfg.OLLAMA_BASE_URL}: {exc}"
            )
            return None

    def _init_anthropic(self):
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self._cfg.ANTHROPIC_API_KEY
            )
            self._anthropic_async_client = anthropic.AsyncAnthropic(
                api_key=self._cfg.ANTHROPIC_API_KEY
            )
            logger.info(
                f"[LLM] Anthropic backend ready — model="
                f"'{getattr(self._cfg, 'ANTHROPIC_MODEL', 'claude-sonnet-5')}'"
            )
        except Exception as exc:
            logger.warning(f"[LLM] Cannot initialise Anthropic client: {exc}")
            self._anthropic_client = None
            self._anthropic_async_client = None

    def _init_huggingface(self):
        try:
            from huggingface_hub import InferenceClient, AsyncInferenceClient
            model = getattr(self._cfg, "HUGGINGFACE_MODEL", None)
            provider = getattr(self._cfg, "HUGGINGFACE_PROVIDER", "auto") or "auto"
            self._hf_client = InferenceClient(
                model=model,
                provider=provider,
                token=self._cfg.HUGGINGFACE_API_TOKEN,
            )
            self._hf_async_client = AsyncInferenceClient(
                model=model,
                provider=provider,
                token=self._cfg.HUGGINGFACE_API_TOKEN,
            )
            logger.info(
                f"[LLM] Hugging Face backend ready — model='{model}' "
                f"provider={provider}"
            )
        except Exception as exc:
            logger.warning(f"[LLM] Cannot initialise Hugging Face client: {exc}")
            self._hf_client = None
            self._hf_async_client = None

    def _build_openai_compatible_clients(self, api_key: str, base_url: str, timeout: float = 120.0):
        import httpx
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        sync_client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        async_client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        return sync_client, async_client

    def _init_groq(self):
        try:
            base_url = getattr(self._cfg, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            timeout = getattr(self._cfg, "GROQ_TIMEOUT", 180.0)
            self._groq_client, self._groq_async_client = self._build_openai_compatible_clients(
                self._cfg.GROQ_API_KEY, base_url, timeout
            )
            logger.info(f"[LLM] Groq backend ready — model='{getattr(self._cfg, 'GROQ_MODEL', '?')}'")
        except Exception as exc:
            logger.warning(f"[LLM] Cannot initialise Groq client: {exc}")
            self._groq_client = None
            self._groq_async_client = None

    def _init_nvidia(self):
        try:
            base_url = getattr(self._cfg, "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            timeout = getattr(self._cfg, "NVIDIA_TIMEOUT", 300.0)
            self._nvidia_client, self._nvidia_async_client = self._build_openai_compatible_clients(
                self._cfg.NVIDIA_API_KEY, base_url, timeout
            )
            logger.info(f"[LLM] NVIDIA NIM backend ready — model='{getattr(self._cfg, 'NVIDIA_MODEL', '?')}'")
        except Exception as exc:
            logger.warning(f"[LLM] Cannot initialise NVIDIA client: {exc}")
            self._nvidia_client = None
            self._nvidia_async_client = None

    @property
    def anthropic_available(self) -> bool:
        return self._anthropic_client is not None

    @property
    def ollama_available(self) -> bool:
        return self._ollama_client is not None

    @property
    def huggingface_available(self) -> bool:
        return self._hf_client is not None

    @property
    def groq_available(self) -> bool:
        return self._groq_client is not None

    @property
    def nvidia_available(self) -> bool:
        return self._nvidia_client is not None

    def _extract_content(self, response: Any) -> str:
        if isinstance(response, dict):
            msg = response.get("message", {})
            if isinstance(msg, dict):
                return msg.get("content", "").strip()
            try:
                return msg.content.strip()
            except AttributeError:
                return str(msg).strip()
        try:
            return response.message.content.strip()
        except AttributeError:
            logger.warning("[LLM] Unexpected response type — falling back to str().")
            return str(response).strip()

    def _model_name(self, m: Any) -> str:
        if isinstance(m, dict):
            return m.get("model", m.get("name", ""))
        return getattr(m, "model", getattr(m, "name", str(m)))

    def _check_models(self) -> None:
        try:
            response = self._ollama_client.list()
            if isinstance(response, dict):
                raw = response.get("models", [])
            else:
                raw = getattr(response, "models", [])
            available = [self._model_name(m) for m in raw if self._model_name(m)]
            model = getattr(self._cfg, "OLLAMA_MODEL", self._cfg.OLLAMA_PRIMARY_MODEL)
            if not any(model in a for a in available if a):
                logger.warning(
                    f"[LLM] Model '{model}' not found locally. "
                    f"Pull it with:  ollama pull {model}"
                )
            else:
                logger.info(f"[LLM] Model '{model}' — OK")
        except Exception as exc:
            logger.warning(f"[LLM] Cannot verify Ollama models: {exc}")

    def _is_backend_available(self, name: str) -> bool:
        return {
            "ollama": self.ollama_available,
            "anthropic": self.anthropic_available,
            "huggingface": self.huggingface_available,
            "groq": self.groq_available,
            "nvidia": self.nvidia_available,
        }.get(name, False)

    def _first_available_backend(self, exclude: Optional[str] = None) -> str:
        for candidate in ("ollama", "anthropic", "huggingface", "groq", "nvidia"):
            if candidate != exclude and self._is_backend_available(candidate):
                return candidate
        raise RuntimeError(
            "No LLM backend available — none of Ollama, Anthropic, Hugging "
            "Face, Groq, or NVIDIA is reachable/configured. Set up a local "
            "Ollama server, or set ANTHROPIC_API_KEY / HUGGINGFACE_API_TOKEN "
            "/ GROQ_API_KEY / NVIDIA_API_KEY."
        )

    def _resolve_backend(self, backend: Optional[str]) -> str:
        b = backend or getattr(self._cfg, "LLM_BACKEND", "ollama")
        if not self._is_backend_available(b):
            logger.warning(
                f"[LLM] Backend '{b}' requested but not configured/reachable "
                "— falling back to the first available backend."
            )
            b = self._first_available_backend(exclude=b)
        return b

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> str:
        b = self._resolve_backend(backend)
        if b == "anthropic":
            return self._chat_anthropic(messages, model, temperature, max_tokens)
        if b == "huggingface":
            return self._chat_huggingface(messages, model, temperature, max_tokens)
        if b == "groq":
            return self._chat_groq(messages, model, temperature, max_tokens)
        if b == "nvidia":
            return self._chat_nvidia(messages, model, temperature, max_tokens)
        return self._chat_ollama(messages, model, temperature, max_tokens)

    async def achat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> str:
        b = self._resolve_backend(backend)
        if b == "anthropic":
            return await self._achat_anthropic(messages, model, temperature, max_tokens)
        if b == "huggingface":
            return await self._achat_huggingface(messages, model, temperature, max_tokens)
        if b == "groq":
            return await self._achat_groq(messages, model, temperature, max_tokens)
        if b == "nvidia":
            return await self._achat_nvidia(messages, model, temperature, max_tokens)
        return await asyncio.to_thread(
            self._chat_ollama, messages, model, temperature, max_tokens
        )

    def _chat_ollama(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.ollama_available:
            raise RuntimeError("Ollama backend not available.")
        model = model or self._cfg.OLLAMA_PRIMARY_MODEL
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        try:
            response = self._chat_ollama_request(model, messages, temperature, max_tokens, think=False)
        except TypeError as exc:
            if "think" not in str(exc):
                raise
            logger.warning(
                "[LLM] Installed 'ollama' package doesn't support the "
                "'think' parameter (needs >=0.5.1) — thinking mode could "
                "not be disabled for this call. Run 'pip install -U "
                "ollama' to fix this permanently."
            )
            response = self._chat_ollama_request(model, messages, temperature, max_tokens, think=None)
        try:
            content = self._extract_content(response)
            if not content:
                raise ValueError("Empty response from LLM.")
            return content
        except Exception as exc:
            logger.error(f"[LLM] chat() failed  model={model}: {exc}")
            raise

    def _chat_ollama_request(self, model, messages, temperature, max_tokens, think):
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": getattr(self._cfg, "OLLAMA_NUM_CTX", 8192),
            },
        }
        if think is not None:
            kwargs["think"] = think
        return self._ollama_client.chat(**kwargs)

    def _to_anthropic_messages(self, messages: List[Dict]):
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]
        if not turns or turns[0]["role"] != "user":
            turns = [{"role": "user", "content": "Continue."}] + turns
        return "\n\n".join(system_parts), turns

    _ANTHROPIC_NO_TEMP = ("claude-sonnet-5", "claude-opus-4-6", "claude-3-7")

    def _anthropic_no_temp(self, model: str) -> bool:
        return any(model.startswith(prefix) for prefix in self._ANTHROPIC_NO_TEMP)

    def _chat_anthropic(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        model = model or getattr(self._cfg, "ANTHROPIC_MODEL", "claude-sonnet-5")
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        system, turns = self._to_anthropic_messages(messages)

        create_kwargs = dict(model=model, system=system, messages=turns, max_tokens=max_tokens)
        if not self._anthropic_no_temp(model):
            create_kwargs["temperature"] = temperature
        try:
            resp = self._anthropic_client.messages.create(**create_kwargs)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            if not text.strip():
                raise ValueError("Empty response from Anthropic.")
            return text.strip()
        except Exception as exc:
            logger.error(f"[LLM] Anthropic chat() failed  model={model}: {exc}")
            raise

    async def _achat_anthropic(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        model = model or getattr(self._cfg, "ANTHROPIC_MODEL", "claude-sonnet-5")
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        system, turns = self._to_anthropic_messages(messages)
        create_kwargs = dict(model=model, system=system, messages=turns, max_tokens=max_tokens)
        if not self._anthropic_no_temp(model):
            create_kwargs["temperature"] = temperature
        try:
            resp = await self._anthropic_async_client.messages.create(**create_kwargs)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            if not text.strip():
                raise ValueError("Empty response from Anthropic.")
            return text.strip()
        except Exception as exc:
            logger.error(f"[LLM] Anthropic achat() failed  model={model}: {exc}")
            raise

    def _chat_huggingface(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.huggingface_available:
            raise RuntimeError("Hugging Face backend not available.")
        model = model or getattr(self._cfg, "HUGGINGFACE_MODEL", None)
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS

        needs_thinking_disable = bool(
            model and (
                "qwen3" in model.lower()
                or "qwen/qwen3" in model.lower()
                or "/qwen3" in model.lower()
            )
        )

        import time
        max_attempts = 4
        last_exc: Exception = RuntimeError("No attempt made.")
        for attempt in range(max_attempts):
            try:
                try:
                    if needs_thinking_disable:
                        resp = self._hf_client.chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    else:
                        resp = self._hf_client.chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                except TypeError as te:
                    if "extra_body" not in str(te):
                        raise
                    logger.warning(
                        "[LLM] Installed 'huggingface_hub' package doesn't "
                        "support 'extra_body' on chat_completion() (needs "
                        ">=0.29.0) — thinking mode could not be disabled for "
                        "this call. Run 'pip install -U huggingface_hub' to "
                        "fix this permanently."
                    )
                    resp = self._hf_client.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                content = resp.choices[0].message.content
                reasoning = getattr(resp.choices[0].message, "reasoning", None)
                if reasoning and reasoning.strip():
                    logger.warning(
                        f"[LLM] Hugging Face returned {len(reasoning)} chars of "
                        f"reasoning content despite enable_thinking=False "
                        f"(model={model})."
                    )
                if not content or not content.strip():
                    raise ValueError("Empty response from Hugging Face.")
                return content.strip()
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                is_retryable = (
                    "connectionerror" in exc_str
                    or "remotedisconnected" in exc_str
                    or "connection aborted" in exc_str
                    or "connection reset" in exc_str
                    or "broken pipe" in exc_str
                    or "empty response" in exc_str
                    or "429" in exc_str
                    or "too many requests" in exc_str
                    or "rate limit" in exc_str
                    or "503" in exc_str
                    or "502" in exc_str
                    or "504" in exc_str
                )
                if is_retryable and attempt < max_attempts - 1:
                    is_rate_limit = "429" in exc_str or "too many requests" in exc_str or "rate limit" in exc_str
                    if is_rate_limit:
                        wait_s = 3.0 * (attempt + 1)
                    else:
                        wait_s = 5.0 * (attempt + 1)
                    logger.warning(
                        f"[LLM] Hugging Face chat() hit a transient connection "
                        f"error (attempt {attempt + 1}/{max_attempts}) — waiting "
                        f"{wait_s:.0f}s before retrying. Error: {exc}"
                    )
                    time.sleep(wait_s)
                    continue
                logger.error(f"[LLM] Hugging Face chat() failed  model={model}: {exc}")
                raise
        logger.error(
            f"[LLM] Hugging Face chat() failed after {max_attempts} attempts  "
            f"model={model}: {last_exc}"
        )
        raise last_exc

    async def _achat_huggingface(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.huggingface_available:
            raise RuntimeError("Hugging Face backend not available.")
        model = model or getattr(self._cfg, "HUGGINGFACE_MODEL", None)
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS

        needs_thinking_disable = bool(
            model and (
                "qwen3" in model.lower()
                or "qwen/qwen3" in model.lower()
                or "/qwen3" in model.lower()
            )
        )

        max_attempts = 4
        last_exc: Exception = RuntimeError("No attempt made.")
        for attempt in range(max_attempts):
            try:
                try:
                    if needs_thinking_disable:
                        resp = await self._hf_async_client.chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    else:
                        resp = await self._hf_async_client.chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                except TypeError as te:
                    if "extra_body" not in str(te):
                        raise
                    logger.warning(
                        "[LLM] Installed 'huggingface_hub' package doesn't "
                        "support 'extra_body' on chat_completion() (needs "
                        ">=0.29.0) — thinking mode could not be disabled for "
                        "this call. Run 'pip install -U huggingface_hub' to "
                        "fix this permanently."
                    )
                    resp = await self._hf_async_client.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                content = resp.choices[0].message.content
                reasoning = getattr(resp.choices[0].message, "reasoning", None)
                if reasoning and reasoning.strip():
                    logger.warning(
                        f"[LLM] Hugging Face returned {len(reasoning)} chars of "
                        f"reasoning content despite enable_thinking=False "
                        f"(model={model})."
                    )
                if not content or not content.strip():
                    raise ValueError("Empty response from Hugging Face.")
                return content.strip()
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "aiohttp" in msg and "install" in msg:
                    logger.warning(
                        "[LLM] AsyncInferenceClient needs aiohttp (pip install "
                        "aiohttp) — falling back to the synchronous Hugging "
                        "Face client via a background thread for this call."
                    )
                    return await asyncio.to_thread(
                        self._chat_huggingface, messages, model, temperature, max_tokens
                    )
                is_retryable = (
                    "connectionerror" in msg
                    or "remotedisconnected" in msg
                    or "connection aborted" in msg
                    or "connection reset" in msg
                    or "broken pipe" in msg
                    or "empty response" in msg
                    or "429" in msg
                    or "too many requests" in msg
                    or "rate limit" in msg
                    or "503" in msg
                    or "502" in msg
                    or "504" in msg
                )
                if is_retryable and attempt < max_attempts - 1:
                    is_rate_limit = "429" in msg or "too many requests" in msg or "rate limit" in msg
                    if is_rate_limit:
                        wait_s = 3.0 * (attempt + 1)
                    else:
                        wait_s = 5.0 * (attempt + 1)
                    logger.warning(
                        f"[LLM] Hugging Face achat() hit a transient connection "
                        f"error (attempt {attempt + 1}/{max_attempts}) — waiting "
                        f"{wait_s:.0f}s before retrying. Error: {exc}"
                    )
                    await asyncio.sleep(wait_s)
                    continue
                logger.error(f"[LLM] Hugging Face achat() failed  model={model}: {exc}")
                raise
        logger.error(
            f"[LLM] Hugging Face achat() failed after {max_attempts} attempts  "
            f"model={model}: {last_exc}"
        )
        raise last_exc

    def _shrink_messages(self, messages: List[Dict], factor: float = 0.4) -> List[Dict]:
        shrunk = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                new_len = max(500, int(len(content) * factor))
                content = content[:new_len] + "\n\n[...truncated to fit the provider's request-size limit...]"
            shrunk.append({**m, "content": content})
        return shrunk

    def _retry_after_seconds(self, exc: Exception, default: float) -> float:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if headers:
            header = headers.get("Retry-After") or headers.get("retry-after")
            if header:
                try:
                    return max(0.5, float(header))
                except (TypeError, ValueError):
                    pass
        return default

    def _post_openai_compatible(self, client, messages, model, temperature, max_tokens):
        resp = client.post("/chat/completions", json={
            "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError("Empty response.")
        return content.strip()

    async def _apost_openai_compatible(self, async_client, messages, model, temperature, max_tokens):
        resp = await async_client.post("/chat/completions", json={
            "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError("Empty response.")
        return content.strip()

    def _chat_openai_compatible(self, client, messages, model, temperature, max_tokens, provider_name: str) -> str:
        import time
        import httpx
        current_messages = messages
        shrunk_once = False
        max_attempts = 4
        retryable_statuses = {404, 429, 500, 502, 503, 504}
        last_exc: Exception = RuntimeError("No attempt made.")
        for attempt in range(max_attempts):
            try:
                return self._post_openai_compatible(client, current_messages, model, temperature, max_tokens)
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                is_transport_error = isinstance(exc, httpx.TransportError)
                if status == 413 and not shrunk_once:
                    logger.warning(
                        f"[LLM] {provider_name} returned 413 Payload Too Large — "
                        f"retrying with truncated message content."
                    )
                    current_messages = self._shrink_messages(current_messages)
                    shrunk_once = True
                    continue
                if (status in retryable_statuses or is_transport_error) and attempt < max_attempts - 1:
                    wait_s = self._retry_after_seconds(exc, default=5.0 * (attempt + 1))
                    reason = f"HTTP {status}" if status else f"a transport error ({type(exc).__name__})"
                    logger.warning(
                        f"[LLM] {provider_name} hit {reason} — waiting "
                        f"{wait_s:.1f}s before retrying (attempt {attempt + 1}/{max_attempts})."
                    )
                    time.sleep(wait_s)
                    continue
                logger.error(f"[LLM] {provider_name} chat() failed  model={model}: {exc}")
                raise
        logger.error(f"[LLM] {provider_name} chat() failed after retries  model={model}: {last_exc}")
        raise last_exc

    async def _achat_openai_compatible(self, async_client, messages, model, temperature, max_tokens, provider_name: str) -> str:
        import httpx
        current_messages = messages
        shrunk_once = False
        max_attempts = 4
        retryable_statuses = {404, 429, 500, 502, 503, 504}
        last_exc: Exception = RuntimeError("No attempt made.")
        for attempt in range(max_attempts):
            try:
                return await self._apost_openai_compatible(async_client, current_messages, model, temperature, max_tokens)
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                is_transport_error = isinstance(exc, httpx.TransportError)
                if status == 413 and not shrunk_once:
                    logger.warning(
                        f"[LLM] {provider_name} returned 413 Payload Too Large — "
                        f"retrying with truncated message content."
                    )
                    current_messages = self._shrink_messages(current_messages)
                    shrunk_once = True
                    continue
                if (status in retryable_statuses or is_transport_error) and attempt < max_attempts - 1:
                    wait_s = self._retry_after_seconds(exc, default=5.0 * (attempt + 1))
                    reason = f"HTTP {status}" if status else f"a transport error ({type(exc).__name__})"
                    logger.warning(
                        f"[LLM] {provider_name} hit {reason} — waiting "
                        f"{wait_s:.1f}s before retrying (attempt {attempt + 1}/{max_attempts})."
                    )
                    await asyncio.sleep(wait_s)
                    continue
                logger.error(f"[LLM] {provider_name} achat() failed  model={model}: {exc}")
                raise
        logger.error(f"[LLM] {provider_name} achat() failed after retries  model={model}: {last_exc}")
        raise last_exc

    def _chat_groq(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.groq_available:
            raise RuntimeError("Groq backend not available.")
        model = model or getattr(self._cfg, "GROQ_MODEL", None)
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        return self._chat_openai_compatible(self._groq_client, messages, model, temperature, max_tokens, "Groq")

    async def _achat_groq(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.groq_available:
            raise RuntimeError("Groq backend not available.")
        model = model or getattr(self._cfg, "GROQ_MODEL", None)
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        return await self._achat_openai_compatible(self._groq_async_client, messages, model, temperature, max_tokens, "Groq")

    _NO_SYSTEM_ROLE_MODELS = {
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    }

    _MODEL_MAX_TOKENS_CAP = {
        "google/gemma-2-2b-it": 1024,
        "google/gemma-2-9b-it": 1024,
        "google/gemma-2-27b-it": 1024,
    }

    def _merge_system_into_user(self, messages: List[Dict]) -> List[Dict]:
        if not messages or messages[0].get("role") != "system":
            return messages
        system_content = messages[0].get("content", "")
        rest = messages[1:]
        if not rest:
            return [{"role": "user", "content": system_content}]
        merged = dict(rest[0])
        merged["content"] = f"{system_content}\n\n{merged.get('content', '')}"
        return [merged] + rest[1:]

    def _chat_nvidia(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.nvidia_available:
            raise RuntimeError("NVIDIA backend not available.")
        model = model or getattr(self._cfg, "NVIDIA_MODEL", None)
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        if model in self._NO_SYSTEM_ROLE_MODELS:
            messages = self._merge_system_into_user(messages)
        cap = self._MODEL_MAX_TOKENS_CAP.get(model)
        if cap is not None and max_tokens > cap:
            max_tokens = cap
        return self._chat_openai_compatible(self._nvidia_client, messages, model, temperature, max_tokens, "NVIDIA")

    async def _achat_nvidia(self, messages, model=None, temperature=0.3, max_tokens=None) -> str:
        if not self.nvidia_available:
            raise RuntimeError("NVIDIA backend not available.")
        model = model or getattr(self._cfg, "NVIDIA_MODEL", None)
        max_tokens = max_tokens or self._cfg.LLM_MAX_TOKENS
        if model in self._NO_SYSTEM_ROLE_MODELS:
            messages = self._merge_system_into_user(messages)
        cap = self._MODEL_MAX_TOKENS_CAP.get(model)
        if cap is not None and max_tokens > cap:
            max_tokens = cap
        return await self._achat_openai_compatible(self._nvidia_async_client, messages, model, temperature, max_tokens, "NVIDIA")

    def fast_chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        default_backend = getattr(self._cfg, "LLM_BACKEND", "ollama")
        if default_backend == "anthropic" and self.anthropic_available:
            return self.chat(messages, temperature=temperature, backend="anthropic")
        if default_backend == "huggingface" and self.huggingface_available:
            return self.chat(
                messages, model=getattr(self._cfg, "HUGGINGFACE_FAST_MODEL", None),
                temperature=temperature, backend="huggingface",
            )
        if self.ollama_available:
            return self.chat(
                messages, model=self._cfg.OLLAMA_FAST_MODEL,
                temperature=temperature, backend="ollama",
            )
        return self.chat(messages, temperature=temperature)

    def generate_with_retry(self, messages, model=None, max_attempts=3) -> str:
        last_exc: Exception = RuntimeError("No attempts made.")
        for attempt in range(1, max_attempts + 1):
            try:
                return self.chat(messages, model=model)
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(f"[LLM] Attempt {attempt}/{max_attempts} failed: {exc}. Retrying...")
        raise last_exc

    def extract_json(self, text: str) -> Any:
        filtered = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(filtered).strip()
        _FAILED = object()

        def _try_parse(s: str):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            for open_b, close_b in [("{", "}"), ("[", "]")]:
                pos = s.find(open_b)
                if pos < 0:
                    continue
                depth, in_str, esc = 0, False, False
                for i, ch in enumerate(s[pos:], start=pos):
                    if esc:
                        esc = False
                        continue
                    if ch == "\\" and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                    if not in_str:
                        if ch == open_b:
                            depth += 1
                        elif ch == close_b:
                            depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(s[pos: i + 1])
                            except json.JSONDecodeError:
                                break
            return _FAILED

        result = _try_parse(cleaned)
        if result is not _FAILED:
            return result

        repaired = self._repair_near_json(cleaned)
        if repaired != cleaned:
            fixed = _try_parse(repaired)
            if fixed is not _FAILED:
                logger.warning(
                    "[LLM] extract_json: strict parse failed but a "
                    "near-JSON repair (unquoted keys/ID-like values) "
                    "rescued the response."
                )
                return fixed

        repaired2 = self._repair_orphan_nutrient_blocks(cleaned)
        if repaired2 != cleaned:
            fixed2 = _try_parse(repaired2)
            if fixed2 is not _FAILED:
                logger.warning(
                    "[LLM] extract_json: strict parse failed but removing "
                    "a duplicated, unlabeled nutrient block rescued the "
                    "response."
                )
                return fixed2

        logger.warning(
            f"[LLM] extract_json full failed response ({len(text)} chars, "
            f"showing up to 4000):\n{text[:4000]}"
        )
        raise ValueError(f"No valid JSON found in LLM response.\nFirst 2500 chars:\n{text[:2500]}")

    @staticmethod
    def _repair_near_json(s: str) -> str:
        _FRACTIONS = {
            "½", "¼", "¾", "⅓", "⅔", "⅕", "⅖", "⅗", "⅘",
            "⅙", "⅚", "⅛", "⅜", "⅝", "⅞",
        }
        _NUM_RE = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
        out = []
        i, n = 0, len(s)
        in_str, esc = False, False
        while i < n:
            ch = s[i]
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < n and s[i + 1] == "/":
                j = s.find("\n", i)
                i = j if j != -1 else n
                continue
            if ch == ",":
                k = i + 1
                while k < n and s[k] in " \t\r\n":
                    k += 1
                if k < n and s[k] in "}]":
                    i += 1
                    continue
                out.append(ch)
                i += 1
                continue
            if ch.isdigit() or (ch == "-" and i + 1 < n and s[i + 1].isdigit()):
                m = _NUM_RE.match(s[i:])
                if m:
                    end = i + len(m.group(0))
                    if end >= n or not (s[end].isalpha() or s[end] == "_"):
                        out.append(m.group(0))
                        i = end
                        continue
            if ch == "." and i + 1 < n and s[i + 1].isdigit():
                out.append("0.")
                i += 1
                continue
            if ch in _FRACTIONS:
                out.append(f'"{ch}"')
                i += 1
                continue
            if ch == "'":
                j = i + 1
                content = []
                closed = False
                while j < n:
                    cj = s[j]
                    if cj == "\\" and j + 1 < n and s[j + 1] == "'":
                        content.append("'")
                        j += 2
                        continue
                    if cj == "'":
                        closed = True
                        j += 1
                        break
                    if cj == "\n":
                        break
                    content.append(cj)
                    j += 1
                if closed:
                    inner = "".join(content).replace("\\", "\\\\").replace('"', '\\"')
                    out.append(f'"{inner}"')
                    i = j
                    continue
            m = re.match(r"[A-Za-z0-9_]+", s[i:])
            if m:
                word = m.group(0)
                j = i + len(word)
                if word in ("true", "false", "null"):
                    out.append(word)
                    i = j
                    continue
                if re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", word):
                    out.append(word)
                    i = j
                    continue
                k = j
                while k < n and s[k] in " \t\r\n":
                    k += 1
                if k < n and s[k] == ":":
                    out.append(f'"{word}"')
                    i = j
                    continue
                if re.fullmatch(r"[0-9a-fA-F]{12,}", word) or (
                    word[:1].isalpha() and re.fullmatch(r"[A-Za-z0-9_]{2,}", word)
                ):
                    out.append(f'"{word}"')
                    i = j
                    continue
                out.append(word)
                i = j
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _repair_orphan_nutrient_blocks(s: str) -> str:
        pattern = re.compile(r',\{"([A-Z][A-Z0-9_]{2,})":\{"label"')
        out = []
        i, n = 0, len(s)
        while i < n:
            m = pattern.match(s, i)
            if m:
                brace_start = m.start() + 1
                depth = 0
                j = brace_start
                in_string = False
                escape = False
                while j < n:
                    c = s[j]
                    if in_string:
                        if escape:
                            escape = False
                        elif c == "\\":
                            escape = True
                        elif c == '"':
                            in_string = False
                    else:
                        if c == '"':
                            in_string = True
                        elif c == "{":
                            depth += 1
                        elif c == "}":
                            depth -= 1
                            if depth == 0:
                                j += 1
                                break
                    j += 1
                if j < n and s[j] == "]" and j + 1 < n and s[j + 1] == ",":
                    j += 1
                i = j
                continue
            out.append(s[i])
            i += 1
        return "".join(out)

    def extract_json_dict(self, text: str, fallback: Optional[Dict] = None) -> Dict:
        try:
            result = self.extract_json(text)
        except ValueError as exc:
            if fallback is not None:
                logger.warning(f"[LLM] extract_json_dict: parse failed ({exc}). Using fallback.")
                return fallback
            raise
        if isinstance(result, list):
            logger.warning(
                f"[LLM] extract_json_dict: LLM returned a list ({len(result)} items) "
                "instead of a JSON object — unwrapping first dict element."
            )
            for item in result:
                if isinstance(item, dict):
                    return item
            if fallback is not None:
                return fallback
            raise ValueError("extract_json_dict: expected object, got list with no dict items.")
        if isinstance(result, dict):
            return result
        if fallback is not None:
            logger.warning(f"[LLM] extract_json_dict: unexpected type {type(result).__name__}. Using fallback.")
            return fallback
        raise ValueError(f"extract_json_dict: expected dict, got {type(result).__name__}: {str(result)[:200]}")

llm = LLMClient()
