import json

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

class BaselineSingleAgentGenerator(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("baseline_gen", "Baseline · Single-Agent Generator", bus)

    def _extract_list_from_response(self, raw: str) -> list | None:
        try:
            parsed = self.llm.extract_json(raw)
        except Exception:
            return None

        if isinstance(parsed, list):
            dict_items = [x for x in parsed if isinstance(x, dict)]
            non_dict = len(parsed) - len(dict_items)
            if non_dict:
                logger.warning(
                    f"[Baseline] Dropped {non_dict} non-dict item(s) from "
                    f"the parsed list (model returned strings/numbers instead "
                    f"of JSON objects). Check that the model is following the "
                    f"schema, not producing numbered text."
                )
            return dict_items if dict_items else None

        if isinstance(parsed, dict):
            list_values = [v for v in parsed.values() if isinstance(v, list) and v]
            if list_values:
                best = max(list_values, key=len)
                dict_items = [x for x in best if isinstance(x, dict)]
                non_dict = len(best) - len(dict_items)
                if non_dict:
                    logger.warning(
                        f"[Baseline] Unwrapped dict wrapper and dropped "
                        f"{non_dict} non-dict item(s) from the inner list."
                    )
                if dict_items:
                    logger.info(
                        f"[Baseline] Unwrapped list from dict wrapper "
                        f"(total={len(best)}, dict items={len(dict_items)})."
                    )
                    return dict_items

        return None

    @staticmethod
    def _deduplicate(items: list) -> list:
        seen_titles, seen_ids, out = set(), set(), []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("label") or item.get("name")
            item_id = item.get("_id") or item.get("id")
            dup = (title and title in seen_titles) or (
                item_id is not None and item_id in seen_ids
            )
            if not dup:
                out.append(item)
                if title:
                    seen_titles.add(title)
                if item_id is not None:
                    seen_ids.add(item_id)
            else:
                logger.info(f"[Baseline] Dropped exact duplicate: title={title!r}")
        return out

    async def execute(self) -> bool:
        from config import settings

        fmt = self.bus.get("data_format", "unknown")
        full_text = self.bus.get("data_full_text", "")
        user_ctx = self.bus.get("user_context", "")

        sample_text = full_text[:12000]

        configured_max = getattr(settings, "BASELINE_MAX_TOKENS", None)
        llm_max = getattr(settings, "LLM_MAX_TOKENS", 6144)
        max_tokens = configured_max if configured_max else llm_max
        max_attempts = getattr(settings, "BASELINE_MAX_PARSE_ATTEMPTS", 3)

        logger.info(
            f"[Baseline] max_tokens={max_tokens} "
            f"max_parse_attempts={max_attempts}"
        )

        system_msg = self._sys(
            "You are a data-generation assistant. You will be given a "
            "sample of a raw dataset and a plain-language instruction. "
            "Infer the dataset's field names, nesting, and value conventions "
            "from the sample, then generate exactly as many items as requested.\n\n"
            "CRITICAL OUTPUT RULES — follow these exactly:\n"
            "1. Output ONLY a raw JSON array starting with [ and ending with ].\n"
            "2. Every element in the array must be a JSON object { } following "
            "the field names and structure you see in the sample. Never output "
            "plain text, numbered lists, or string values as array elements.\n"
            "3. Do NOT wrap the array in any object like {\"recipes\": [...]} "
            "or {\"data\": [...]}. The array must be the outermost element.\n"
            "4. Every item must be unique — different title and different "
            "ingredients. Never repeat the same item.\n"
            "5. No markdown fences (```), no explanation, no text before or "
            "after the array."
        )

        base_user_msg = (
            f"Dataset format: {fmt}\n\n"
            f"Dataset sample (infer the exact schema from this):\n{sample_text}\n\n"
            f"Instructions:\n{user_ctx}\n\n"
            "Generate the requested items now. Each item must be a JSON object "
            "{{ }} with the same fields visible in the sample above. "
            "Output ONLY the JSON array [ {{...}}, {{...}}, ... ] — nothing else."
        )

        output = None
        attempts_used = 0
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            user_msg = base_user_msg
            if attempt > 1:
                user_msg += (
                    f"\n\nPrevious attempt failed: {last_error}\n"
                    f"IMPORTANT: output ONLY a JSON array of objects. "
                    f"Each element must be a JSON object {{...}}, NOT a "
                    f"string or number. Start with [ and end with ]."
                )

            raw = self.llm.chat(
                [system_msg, self._usr(user_msg)],
                temperature=0.4,
                max_tokens=max_tokens,
            )

            result = self._extract_list_from_response(raw)
            if result:
                output = self._deduplicate(result)
                logger.info(
                    f"[Baseline] Attempt {attempt}/{max_attempts}: "
                    f"parsed OK → {len(result)} raw, {len(output)} after dedup."
                )
                break

            last_error = (
                "response did not contain a parseable JSON array of objects "
                f"(first 300 chars: {raw[:300]!r})"
            )
            logger.warning(
                f"[Baseline] Attempt {attempt}/{max_attempts}: {last_error}"
            )

        if not output:
            logger.error(
                f"[Baseline] All {max_attempts} attempt(s) failed to produce "
                f"a parseable JSON array of objects — writing an empty result."
            )
            output = []

        logger.info(
            f"[Baseline] Generated {len(output)} item(s) in "
            f"{attempts_used} attempt(s)."
        )

        self.bus.set("generated_output", output)
        self.bus.set("generated_format", "json")
        self.bus.set("baseline_parse_attempts", attempts_used)
        self.bus.set("run_kind", "baseline")

        await self.bus.publish(Message(
            agent_id=self.agent_id,
            topic="baseline_generated",
            payload={"items": len(output), "attempts": attempts_used},
        ))
        return True
