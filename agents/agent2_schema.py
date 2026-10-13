import hashlib
import json
from pathlib import Path

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

_SCHEMA_FALLBACK = {
    "top_level_type": "array",
    "record_count_approx": 0,
    "primary_entity": "record",
    "fields": {},
    "nested_structures": [],
    "categorical_fields": {},
    "numeric_ranges": {},
    "key_patterns": [],
    "domain": "unknown",
    "parse_error": True,
}

_SCHEMA_CACHE_VERSION = "v2"

class Agent2SchemaInspector(BaseAgent):

    def __init__(self, bus: MessageBus, force_recompute: bool = False):
        super().__init__("agent2", "Agent 2 · Schema Inspector", bus)
        self.force_recompute = force_recompute

    def _cache_path(self, key: str) -> Path:
        from config import settings
        cache_dir = Path(getattr(settings, "SCHEMA_CACHE_DIR", ".cache/schema"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}.json"

    async def execute(self) -> bool:
        from config import settings, active_model

        fmt = self.bus.get("data_format", "unknown")
        full_text = self.bus.get("data_full_text", "")
        inspection_text = full_text[:7000]

        cache_enabled = getattr(settings, "SCHEMA_CACHE_ENABLED", True)
        model_fingerprint = f"{settings.LLM_BACKEND}:{active_model(settings)}"
        cache_key = hashlib.sha256(
            (_SCHEMA_CACHE_VERSION + "::" + model_fingerprint + "::" + fmt + "::" + full_text)
            .encode("utf-8")
        ).hexdigest()[:16]
        cache_file = self._cache_path(cache_key) if cache_enabled else None

        if cache_file and cache_file.exists() and not self.force_recompute:
            try:
                schema = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.info(
                    f"[A2] Cache HIT — entity='{schema.get('primary_entity','?')}' "
                    f"domain='{schema.get('domain','?')}' (skipped 2 LLM calls)"
                )
                self.bus.set("inferred_schema", schema)
                self.bus.set("schema_analysis", schema.get("raw_analysis", ""))
                await self.bus.publish(Message(
                    agent_id=self.agent_id, topic="schema_inspected", payload=schema
                ))
                return True
            except Exception as exc:
                logger.warning(f"[A2] Cache read failed ({exc}) — recomputing.")
        elif cache_file and cache_file.exists() and self.force_recompute:
            logger.info(
                "[A2] force_recompute=True (deep retry) — bypassing cache HIT, "
                "re-inspecting schema from scratch."
            )

        analysis = self.llm.chat([
            self._sys(
                "You are an expert data scientist and schema analyst. "
                "Analyse the provided raw dataset exhaustively. "
                "Identify every field, its type, role, and constraints. "
                "Note nested structures, arrays, enumerations, and ID patterns."
            ),
            self._usr(
                f"Dataset format: {fmt}\n\n"
                f"Dataset content (sample):\n{inspection_text}\n\n"
                "Provide a thorough analysis covering:\n"
                "1. Top-level structure (array / single object / other)\n"
                "2. All fields with types, descriptions, required vs optional\n"
                "3. Nested structures and their schemas\n"
                "4. Enumerated / categorical fields and their known values\n"
                "5. Numeric fields and their typical ranges\n"
                "6. Unique identifiers and key relationships\n"
                "7. Domain and subject matter of the dataset"
            ),
        ])

        schema_raw = self.llm.chat([
            self._sys(
                "You are a precise data modeller. "
                "Output ONLY a single, valid JSON OBJECT that starts with { and ends with }. "
                "Do NOT wrap it in an array. Do NOT add markdown fences. "
                "No explanation before or after the JSON object."
            ),
            self._usr(
                f"Schema analysis:\n{analysis}\n\n"
                "Return a single JSON object with EXACTLY these keys "
                "(no extra keys, no wrapping array):\n"
                "{\n"
                "  \"top_level_type\": \"array or object or string\",\n"
                "  \"record_count_approx\": <integer>,\n"
                "  \"primary_entity\": \"<entity name>\",\n"
                "  \"fields\": {\"<fieldName>\": {\"type\": \"<type>\", "
                "\"description\": \"<desc>\", \"required\": true}},\n"
                "  \"nested_structures\": [\"<description>\"],\n"
                "  \"categorical_fields\": {\"<field>\": [\"<value1>\"]},\n"
                "  \"numeric_ranges\": {\"<field>\": {\"min\": 0, \"max\": 0}},\n"
                "  \"key_patterns\": [\"<description>\"],\n"
                "  \"domain\": \"<domain name>\"\n"
                "}\n\n"
                "IMPORTANT: output ONLY the JSON object above. "
                "Start your response with { and end with }."
            ),
        ], temperature=0.1)

        schema = self.llm.extract_json_dict(schema_raw, fallback=dict(_SCHEMA_FALLBACK))

        schema.setdefault("raw_analysis", analysis[:2000])

        logger.info(
            f"[A2] entity='{schema.get('primary_entity', '?')}' "
            f"type='{schema.get('top_level_type', '?')}' "
            f"domain='{schema.get('domain', '?')}'"
        )

        if cache_file:
            try:
                cache_file.write_text(
                    json.dumps(schema, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning(f"[A2] Cache write failed: {exc}")

        self.bus.set("inferred_schema", schema)
        self.bus.set("schema_analysis", analysis)

        await self.bus.publish(Message(
            agent_id=self.agent_id, topic="schema_inspected", payload=schema
        ))
        return True
