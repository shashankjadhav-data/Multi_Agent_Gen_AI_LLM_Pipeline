
import json

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

_TASK_FALLBACK = {
    "task_type": "generate",
    "action": "generate output based on user instructions",
    "output_format": "json",
    "output_schema": "same_as_input",
    "quantity": None,
    "constraints": [],
    "quality_criteria": ["valid output", "accurate content"],
    "requires_rag": True,
    "summary": "Generate output as instructed by the user.",
    "parse_error": True,
}

class Agent3TaskRecognizer(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("agent3", "Agent 3 · Task Recognizer", bus)

    async def execute(self) -> bool:
        user_ctx = self.bus.get("user_context", "")
        schema = self.bus.get("inferred_schema", {})
        fmt = self.bus.get("data_format", "")

        intent = self.llm.chat([
            self._sys(
                "You are an expert task analyst specialising in NLP and data workflows. "
                "Identify exactly what the user wants to achieve with their dataset."
            ),
            self._usr(
                f"User Instructions:\n{user_ctx}\n\n"
                f"Dataset Format: {fmt}\n"
                f"Dataset Schema:\n{json.dumps(schema, indent=2)[:2000]}\n\n"
                "Identify with precision:\n"
                "1. Main task category (generate/transform/analyse/summarise/extract/visualise)\n"
                "2. Specific action requested\n"
                "3. Desired output format (json/csv/text/image/markdown)\n"
                "4. Quantity or scope (number of items, etc.)\n"
                "5. Hard constraints that MUST be satisfied\n"
                "6. Quality or style requirements\n"
                "7. Whether RAG over the dataset is needed"
            ),
        ])

        spec_raw = self.llm.chat([
            self._sys(
                "You are a precise task specification generator. "
                "Output ONLY a single valid JSON OBJECT that starts with { and ends with }. "
                "Do NOT wrap it in an array. Do NOT add markdown fences. "
                "No explanation before or after the JSON object."
            ),
            self._usr(
                f"Task analysis:\n{intent}\n\n"
                "Return a single JSON object with EXACTLY these keys "
                "(no extra keys, no wrapping array):\n"
                "{\n"
                "  \"task_type\": \"generate|transform|analyse|summarise|extract|visualise\",\n"
                "  \"action\": \"<specific action description>\",\n"
                "  \"output_format\": \"json|csv|text|image|markdown\",\n"
                "  \"output_schema\": \"same_as_input|custom|described\",\n"
                "  \"quantity\": <integer or null>,\n"
                "  \"constraints\": [\"<constraint1>\", \"<constraint2>\"],\n"
                "  \"quality_criteria\": [\"<criterion1>\"],\n"
                "  \"requires_rag\": true,\n"
                "  \"summary\": \"<one-sentence task description>\"\n"
                "}\n\n"
                "IMPORTANT: output ONLY the JSON object above. "
                "Start your response with { and end with }."
            ),
        ], temperature=0.1)

        task = self.llm.extract_json_dict(spec_raw, fallback=dict(_TASK_FALLBACK))

        task.setdefault("raw_intent", intent[:500])

        logger.info(
            f"[A3] type={task.get('task_type')} "
            f"format={task.get('output_format')} "
            f"qty={task.get('quantity')} "
            f"summary={str(task.get('summary', ''))[:70]}"
        )

        self.bus.set("task_spec", task)
        self.bus.set("output_format", task.get("output_format", "json"))

        await self.bus.publish(Message(
            agent_id=self.agent_id, topic="task_recognised", payload=task
        ))
        return True
