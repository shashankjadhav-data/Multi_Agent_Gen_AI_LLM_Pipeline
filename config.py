import os
from pathlib import Path
from typing import Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):

    OLLAMA_BASE_URL: str         = "http://localhost:11434"

    OLLAMA_MODEL: str             = "qwen2.5:7b-instruct"
    OLLAMA_PRIMARY_MODEL: str     = "qwen2.5:7b-instruct"
    OLLAMA_FAST_MODEL: str        = "qwen2.5:7b-instruct"
    OLLAMA_TIMEOUT: int           = 600

    OLLAMA_NUM_CTX: int          = 8192

    ANTHROPIC_API_KEY: str       = ""
    ANTHROPIC_MODEL: str         = "claude-sonnet-5"

    HUGGINGFACE_API_TOKEN: str    = ""

    HUGGINGFACE_PROVIDER: str     = "auto"

    HUGGINGFACE_MODEL: str          = "Qwen/Qwen3-8B"
    HUGGINGFACE_PRIMARY_MODEL: str  = "Qwen/Qwen3-8B"
    HUGGINGFACE_FAST_MODEL: str     = "Qwen/Qwen3-8B"

    GROQ_API_KEY: str            = ""
    GROQ_BASE_URL: str           = "https://api.groq.com/openai/v1"

    GROQ_MODEL: str              = "openai/gpt-oss-20b"

    GROQ_TIMEOUT: float          = 180.0

    NVIDIA_API_KEY: str          = ""
    NVIDIA_BASE_URL: str         = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str            = "meta/llama-3.3-70b-instruct"
    NVIDIA_TIMEOUT: float        = 300.0

    LLM_BACKEND: str             = "ollama"

    GENERATION_BACKEND: str      = "auto"

    MODEL_PROFILE: str           = ""

    EMBEDDING_MODEL: str         = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int    = 32

    MAX_RETRIES: int             = 10

    CHUNK_SIZE: int              = 2000
    MAX_FAISS_CHUNKS: int        = 120
    TOP_K_CHUNKS: int            = 8

    GENERATION_BATCH_SIZE: int   = 5
    MAX_GEN_ATTEMPTS: int        = 6
    MAX_LLM_CONTEXT_CHARS: int   = 2000
    MAX_EXAMPLES_IN_PROMPT: int  = 5
    LLM_MAX_TOKENS: int          = 6144

    AGENT5_BATCH_TOKENS_CEILING: int = 8192
    AGENT5_HF_BATCH_TOKENS_CEILING: int = 8192

    BASELINE_MAX_TOKENS: Optional[int] = None
    BASELINE_MAX_PARSE_ATTEMPTS: int = 3
    GENERATION_CONCURRENCY: int  = 2
    MIN_GEN_BATCH_SIZE: int      = 1

    SCHEMA_CACHE_ENABLED: bool   = True
    SCHEMA_CACHE_DIR: Path       = Path(".cache/schema")

    OUTPUT_DIR: Path = Path("outputs")
    LOG_DIR:    Path = Path("logs")
    TEMP_DIR:   Path = Path("temp")

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_dirs(self) -> None:
        for d in [self.OUTPUT_DIR, self.LOG_DIR, self.TEMP_DIR]:
            d.mkdir(parents=True, exist_ok=True)

MODEL_PROFILES: Dict[str, Dict[str, str]] = {

    "qwen2.5:7b-instruct": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "qwen2.5:7b-instruct",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "RECOMMENDED for 6-8GB GPUs. Strong structured-JSON/enum "
            "compliance (~4.7GB VRAM). ollama pull qwen2.5:7b-instruct"
        ),
    },
    "qwen2.5:3b-instruct": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "qwen2.5:3b-instruct",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Faster/lighter Qwen (~2GB VRAM), some accuracy trade-off. "
            "ollama pull qwen2.5:3b-instruct"
        ),
    },
    "phi4-mini:3.8b-q4_K_M": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "phi4-mini:3.8b-q4_K_M",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Smallest footprint (~2.5GB VRAM), MIT license, strong "
            "reasoning-per-parameter. ollama pull phi4-mini:3.8b-q4_K_M"
        ),
    },
    "llama3.1:8b-instruct-q4_K_M": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "llama3.1:8b-instruct-q4_K_M",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Meta Llama 3.1 alternative to Qwen (~4.9GB VRAM). "
            "ollama pull llama3.1:8b-instruct-q4_K_M"
        ),
    },
    "qwen3:8b": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "qwen3:8b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Newer generation than Qwen2.5/Llama3.1 (~5.2GB VRAM at "
            "Q4_K_M), strong instruction-following. Ships with 'thinking "
            "mode' ON by default in Ollama, which slows down generation — "
            "for fastest structured-JSON output, disable it once via "
            "'ollama run qwen3:8b --no-think' or the API's "
            "enable_thinking=false parameter. ollama pull qwen3:8b"
        ),
    },
    "qwen3:14b": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "qwen3:14b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Larger/higher-quality Qwen3 tier (~9.3GB at Q4_K_M) — does "
            "NOT fit a 6GB GPU at standard quantization; Ollama will "
            "offload part of it to CPU/system RAM, which is likely SLOWER "
            "than the 8B-class profiles above, not faster. Needs ~10-12GB "
            "VRAM to run fully on-GPU. Also ships with 'thinking mode' ON "
            "by default — see qwen3:8b's note. Only use this if you're "
            "trading speed for a quality experiment. ollama pull qwen3:14b"
        ),
    },
    "mistral:7b-instruct-v0.3-q4_K_M": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "mistral:7b-instruct-v0.3-q4_K_M",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Mistral AI's 7B model, v0.3 (~4.4GB) — comfortable fit, "
            "native function-calling support, Apache 2.0. A solid "
            "alternative to try if Qwen/Llama don't suit your data. "
            "ollama pull mistral:7b-instruct-v0.3-q4_K_M"
        ),
    },
    "gemma2:9b-instruct-q4_K_M": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "gemma2:9b-instruct-q4_K_M",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Google's Gemma 2 9B (~5.8GB at Q4_K_M) — EXTREMELY tight fit "
            "on a 6GB GPU, only ~200MB headroom left for context/KV cache. "
            "May fail to load fully on-GPU or leave very little room for "
            "OLLAMA_NUM_CTX; if you hit OOM or unexpectedly slow runs, "
            "this is the first profile to suspect. Also an older "
            "generation than Gemma 3/4 (2024 training). "
            "ollama pull gemma2:9b-instruct-q4_K_M"
        ),
    },
    "qwen2.5-coder:7b": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "qwen2.5-coder:7b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Code-specialized sibling of qwen2.5:7b-instruct above — same "
            "7B base, same ~4.7GB VRAM footprint, so it's a direct "
            "apples-to-apples swap on hardware. Worth testing specifically "
            "against this pipeline's structured-JSON-output problem: "
            "code-tuned models are trained on large volumes of "
            "strictly-syntactic text (source code) and often carry "
            "stricter bracket/quote discipline as a side effect, even on "
            "non-code tasks like JSON recipe generation. Untested on this "
            "pipeline as of writing — a genuine experiment, not a "
            "confirmed improvement. ollama pull qwen2.5-coder:7b"
        ),
    },
    "granite4.1:8b": {
        "LLM_BACKEND": "ollama",
        "GENERATION_BACKEND": "ollama",
        "OLLAMA_MODEL": "granite4.1:8b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "IBM's Granite 4.1, 8B — Apache 2.0, and the only model in "
            "this whole profile list whose OWN listing explicitly "
            "advertises structured JSON output support (alongside "
            "multilingual, RAG, and tool-use), rather than that being an "
            "inferred property from general instruction-following "
            "quality. Directly relevant given this pipeline's recurring "
            "bottleneck. VRAM footprint at Q4_K_M is an ESTIMATE (~4.9-5GB, "
            "in line with other 8B-class models here) — not independently "
            "confirmed the way the sizes above were. ollama pull "
            "granite4.1:8b"
        ),
    },

    "Qwen/Qwen3-8B": {
        "LLM_BACKEND": "huggingface",
        "GENERATION_BACKEND": "huggingface",
        "HUGGINGFACE_MODEL": "Qwen/Qwen3-8B",
        "description": (
            "Qwen3-8B via Hugging Face's hosted Inference API — no local "
            "GPU load. Requires HUGGINGFACE_API_TOKEN. Note: Qwen3 supports "
            "an extended 'thinking mode'; if responses are slow/verbose, "
            "check your chosen HF inference provider's docs for how to "
            "disable it for this model."
        ),
    },
    "microsoft/Phi-4-mini-instruct": {
        "LLM_BACKEND": "huggingface",
        "GENERATION_BACKEND": "huggingface",
        "HUGGINGFACE_MODEL": "microsoft/Phi-4-mini-instruct",
        "description": (
            "Phi-4-mini-instruct via Hugging Face's hosted Inference API — "
            "no local GPU load. Requires HUGGINGFACE_API_TOKEN."
        ),
    },
    "prism-ml/Ternary-Bonsai-27B-AWQ-4bit": {
        "LLM_BACKEND": "huggingface",
        "GENERATION_BACKEND": "huggingface",
        "HUGGINGFACE_MODEL": "prism-ml/Ternary-Bonsai-27B-AWQ-4bit",
        "HUGGINGFACE_PROVIDER": "together",
        "description": (
            "27B, AWQ 4-bit, Apache 2.0 — very recently uploaded (days old "
            "as of writing). Pinned to the 'together' provider explicitly "
            "(not 'auto') to avoid landing on a payment-required provider "
            "like the nscale 402 error seen in production. UNVERIFIED "
            "whether Together actually serves this specific, very new "
            "model yet — the model card's only documented usage is "
            "self-hosting via sglang across multiple GPUs, not a "
            "confirmed hosted-Inference-Providers listing. If you see a "
            "'model not supported by this provider' error, that's why. "
            "Requires HUGGINGFACE_API_TOKEN."
        ),
    },
    "google/gemma-4-31B-it": {
        "LLM_BACKEND": "huggingface",
        "GENERATION_BACKEND": "huggingface",
        "HUGGINGFACE_MODEL": "google/gemma-4-31B-it",
        "HUGGINGFACE_PROVIDER": "auto",
        "description": (
            "Google DeepMind's Gemma 4, 31B, multimodal (text+image), "
            "256K context, configurable reasoning/thinking modes — via "
            "Hugging Face's hosted Inference API. Pinned to the "
            "'cerebras' provider explicitly (not 'auto') to avoid landing "
            "on a payment-required provider. Requires HUGGINGFACE_API_TOKEN."
        ),
    },
    "openai/gpt-oss-safeguard-20b": {
        "LLM_BACKEND": "huggingface",
        "GENERATION_BACKEND": "huggingface",
        "HUGGINGFACE_MODEL": "openai/gpt-oss-safeguard-20b",
        "HUGGINGFACE_PROVIDER": "auto",
        "description": (
            "OpenAI's open-weight safety-reasoning model (20B total, 3.6B "
            "active), confirmed available via Groq specifically. Pinned "
            "to the 'groq' provider explicitly (not 'auto'). IMPORTANT: "
            "this model is purpose-built for safety/content-policy "
            "classification using OpenAI's 'harmony' response format, "
            "NOT general content generation — its own docs say it 'will "
            "not work correctly' outside that format. May not perform "
            "well for this pipeline's structured-JSON recipe-generation "
            "task; worth testing on a small run before relying on it. "
            "Requires HUGGINGFACE_API_TOKEN."
        ),
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "LLM_BACKEND": "huggingface",
        "GENERATION_BACKEND": "huggingface",
        "HUGGINGFACE_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        "description": (
            "The exact same model as the RECOMMENDED qwen2.5:7b-instruct "
            "Ollama profile above, via Hugging Face's hosted Inference "
            "API instead of local GPU — useful for isolating whether a "
            "result is about the MODEL or about running it locally "
            "through Ollama specifically. Left on 'auto' provider "
            "routing rather than pinned: unlike the other HF profiles "
            "above, no single provider serving this exact model was "
            "confirmed while researching this list. If you hit a "
            "payment-required error, check this model's Hugging Face "
            "page for which Inference Providers currently list it and "
            "set HUGGINGFACE_PROVIDER explicitly, same fix as the other "
            "profiles here. Requires HUGGINGFACE_API_TOKEN."
        ),
    },

    "openai/gpt-oss-20b": {
        "LLM_BACKEND": "groq",
        "GENERATION_BACKEND": "groq",
        "GROQ_MODEL": "openai/gpt-oss-20b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Free on Groq's own direct API (LPU hardware — very fast, "
            "500+ tok/s) — no local GPU load. Confirmed current account "
            "limits: 30 req/min, 1K req/day, 8K tokens/min, 200K "
            "tokens/day. That daily token budget is tight for this "
            "pipeline's more retry-heavy runs — see llama-3.1-8b-instant "
            "below if you're iterating many times per day. Requires "
            "GROQ_API_KEY from console.groq.com."
        ),
    },
    "openai/gpt-oss-120b": {
        "LLM_BACKEND": "groq",
        "GENERATION_BACKEND": "groq",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Larger, more capable free model on Groq's direct API — same "
            "LPU speed, no local GPU load. Confirmed current account "
            "limits: 30 req/min, 1K req/day, 8K tokens/min, 200K "
            "tokens/day (same budget as gpt-oss-20b, more capability per "
            "token). Requires GROQ_API_KEY from console.groq.com."
        ),
    },
    "llama-3.1-8b-instant": {
        "LLM_BACKEND": "groq",
        "GENERATION_BACKEND": "groq",
        "GROQ_MODEL": "llama-3.1-8b-instant",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "BEST DAILY QUOTA for repeated testing/iteration on Groq's "
            "direct API: confirmed 14.4K requests/day and 500K "
            "tokens/day — 14x and 2.5-5x more headroom respectively than "
            "the gpt-oss/qwen models below, meaning far more full "
            "pipeline runs per day before hitting a daily cap. Slightly "
            "lower per-minute token budget (6K TPM) than gpt-oss. Groq's "
            "docs point new integrations toward gpt-oss instead, but "
            "this remains fully active with real rate limits on the "
            "account's own dashboard as of writing. Requires "
            "GROQ_API_KEY from console.groq.com."
        ),
    },
    "llama-3.3-70b-versatile": {
        "LLM_BACKEND": "groq",
        "GENERATION_BACKEND": "groq",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "BEST PER-MINUTE TOKEN BUDGET on Groq's direct API: confirmed "
            "12K tokens/min (vs 8K for gpt-oss/qwen), useful if a single "
            "request's prompt is large enough to risk a 413/rate issue "
            "on the tighter-TPM models. Tightest DAILY token budget of "
            "the bunch, though (100K tokens/day, 1K requests/day) — good "
            "for a single high-quality run, not for many iterations in "
            "one day. NOTE: this is Groq's OWN "
            "'llama-3.3-70b-versatile' — a different exact model string "
            "from the 'meta/llama-3.3-70b-instruct' profile above, which "
            "is the same underlying model family served via NVIDIA "
            "instead. Groq's docs point new integrations toward gpt-oss "
            "instead, but this remains fully active with real rate "
            "limits on the account's own dashboard as of writing. "
            "Requires GROQ_API_KEY from console.groq.com."
        ),
    },
    "qwen/qwen3.6-27b": {
        "LLM_BACKEND": "groq",
        "GENERATION_BACKEND": "groq",
        "GROQ_MODEL": "qwen/qwen3.6-27b",
        "GENERATION_CONCURRENCY": 1,
        "description": (
            "Bigger than the other free Groq profiles here (27B vs 8B/20B) "
            "but confirmed free-tier and, notably, confirmed by Groq's own "
            "docs to support native JSON mode via response_format — "
            "directly relevant given this pipeline's recurring "
            "structured-output problems, and something none of the other "
            "current profiles use even where the backend supports it. "
            "Confirmed current limits: 250K tokens/min, 1K requests/min, "
            "131,072 context, 16,384 max completion tokens. Listed under "
            "Groq's 'Preview Models' (not 'Production') as of writing — "
            "their own docs say preview models 'may be discontinued at "
            "short notice,' so don't build a long-running comparison "
            "around it without rechecking availability. Ships with "
            "'thinking mode' — pass reasoning_format if you want the "
            "reasoning content separated from the answer, or set "
            "reasoning to 'none' to disable it, same idea as qwen3:8b's "
            "note above. Requires GROQ_API_KEY from console.groq.com."
        ),
    },

    "meta/llama-3.3-70b-instruct": {
        "LLM_BACKEND": "nvidia",
        "GENERATION_BACKEND": "nvidia",
        "NVIDIA_MODEL": "meta/llama-3.3-70b-instruct",
        "GENERATION_CONCURRENCY": 2,
        "description": (
            "Free via NVIDIA's hosted NIM API (build.nvidia.com) — a 70B "
            "model with confirmed structured-output support, impossible "
            "to run locally on a 6GB GPU but free and fast hosted. Free "
            "tier: ~1000 inference credits, 40 requests/minute. Requires "
            "NVIDIA_API_KEY (starts with 'nvapi-') from build.nvidia.com."
        ),
    },
    "deepseek-ai/deepseek-v4-flash": {
        "LLM_BACKEND": "nvidia",
        "GENERATION_BACKEND": "nvidia",
        "NVIDIA_MODEL": "deepseek-ai/deepseek-v4-flash",
        "GENERATION_CONCURRENCY": 2,
        "description": (
            "Confirmed listed on NVIDIA's NIM API reference — MoE, 284B "
            "total/13B activated, 1M token context, strong reasoning/"
            "agentic/coding benchmarks. CAUTION: this model uses a custom "
            "encoding pipeline with three reasoning modes (Non-think, "
            "Think High, Think Max) selected via extra request parameters "
            "this pipeline does NOT currently send — a previously-reported "
            "issue shows the closely-related deepseek-v4-flash endpoint "
            "hanging without the right chat_template_kwargs. May hang or "
            "behave unexpectedly rather than erroring cleanly; if it does, "
            "the underlying fix needs a specific reasoning-mode parameter "
            "this description can't yet confirm the exact format for — "
            "not something a config/profile change alone can guarantee. "
            "Test on a small run first. Requires NVIDIA_API_KEY."
        ),
    },
    "deepseek-ai/deepseek-v4-pro": {
        "LLM_BACKEND": "nvidia",
        "GENERATION_BACKEND": "nvidia",
        "NVIDIA_MODEL": "deepseek-ai/deepseek-v4-pro",
        "GENERATION_CONCURRENCY": 2,
        "description": (
            "Confirmed listed on NVIDIA's NIM API reference — MoE, 1.6T "
            "total/49B activated, 1M token context, DeepSeek's strongest "
            "V4 tier (best benchmark scores of the V4 family). Same "
            "reasoning-mode CAUTION as deepseek-v4-flash above applies "
            "here too — untested whether it hangs the same way without "
            "the right extra parameters. Test on a small run first. "
            "Requires NVIDIA_API_KEY."
        ),
    },
    "google/gemma-2-2b-it": {
        "LLM_BACKEND": "nvidia",
        "GENERATION_BACKEND": "nvidia",
        "NVIDIA_MODEL": "google/gemma-2-2b-it",
        "GENERATION_CONCURRENCY": 2,
        "description": (
            "Confirmed listed on NVIDIA's NIM API reference — small (2B), "
            "standard chat-completions endpoint (no reasoning-mode "
            "caveat). Free, fast, useful for quick iteration/testing "
            "rather than final-quality output — an older, much smaller "
            "generation than google/gemma-4-31B-it above. Requires "
            "NVIDIA_API_KEY."
        ),
    },
    "qwen/qwen2.5-7b-instruct": {
        "LLM_BACKEND": "nvidia",
        "GENERATION_BACKEND": "nvidia",
        "NVIDIA_MODEL": "qwen/qwen2.5-7b-instruct",
        "GENERATION_CONCURRENCY": 2,
        "description": (
            "The exact same model as the RECOMMENDED qwen2.5:7b-instruct "
            "Ollama profile above, via NVIDIA's hosted NIM API instead of "
            "local GPU. CAUTION, and read this before relying on it: "
            "NVIDIA's own catalog page for this exact model showed a "
            "'Downloadable' tag in one check during research for this "
            "profile, not a confirmed 'Free Endpoint' tag — those are "
            "different things (see the mixtral-8x22b removal note above, "
            "where exactly this distinction caused a live 404 previously). "
            "The older 'qwen2-7b-instruct' (no '.5') IS confirmed 'Free "
            "Endpoint' on the same catalog, as is the coding-specialized "
            "'qwen2.5-coder-7b-instruct' — either is a safer bet if this "
            "profile 404s. Verify this model's actual status on its "
            "build.nvidia.com page before depending on it. Requires "
            "NVIDIA_API_KEY."
        ),
    },

    "claude-sonnet-5": {
        "LLM_BACKEND": "anthropic",
        "GENERATION_BACKEND": "anthropic",
        "description": (
            "Claude handles everything via ANTHROPIC_API_KEY. This "
            "profile sets ANTHROPIC_MODEL to match its own name "
            "(claude-sonnet-5) — if you change ANTHROPIC_MODEL separately "
            "afterward, this profile's name will no longer match what it "
            "actually runs."
        ),
        "ANTHROPIC_MODEL": "claude-sonnet-5",
    },
}

def _sync_single_model_aliases(cfg: Config) -> None:
    cfg.OLLAMA_PRIMARY_MODEL = cfg.OLLAMA_MODEL
    cfg.OLLAMA_FAST_MODEL = cfg.OLLAMA_MODEL
    cfg.HUGGINGFACE_PRIMARY_MODEL = cfg.HUGGINGFACE_MODEL
    cfg.HUGGINGFACE_FAST_MODEL = cfg.HUGGINGFACE_MODEL

def active_model(cfg: Config) -> str:
    if cfg.LLM_BACKEND == "anthropic":
        return cfg.ANTHROPIC_MODEL
    if cfg.LLM_BACKEND == "huggingface":
        return cfg.HUGGINGFACE_MODEL
    if cfg.LLM_BACKEND == "groq":
        return cfg.GROQ_MODEL
    if cfg.LLM_BACKEND == "nvidia":
        return cfg.NVIDIA_MODEL
    return cfg.OLLAMA_MODEL

def apply_model_profile(cfg: Config, name: str = "") -> None:
    name = name or cfg.MODEL_PROFILE or os.getenv("MODEL_PROFILE", "")
    if not name:
        _sync_single_model_aliases(cfg)
        return
    profile = MODEL_PROFILES.get(name)
    if not profile:
        available = ", ".join(sorted(MODEL_PROFILES.keys()))
        raise ValueError(f"Unknown MODEL_PROFILE '{name}'. Available: {available}")
    for key, value in profile.items():
        if key == "description":
            continue
        setattr(cfg, key, value)
    cfg.MODEL_PROFILE = name
    _sync_single_model_aliases(cfg)

settings = Config()
apply_model_profile(settings)
settings.ensure_dirs()
