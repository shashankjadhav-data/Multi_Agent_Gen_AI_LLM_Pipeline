from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

class Agent7NaiveRetry(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("agent7", "Agent 7 · Naive Retry Ablation", bus)

    async def execute(self) -> bool:
        from config import settings

        retry = self.bus.get("retry_count", 0) + 1
        self.bus.set("retry_count", retry)

        verdict = self.bus.get("validation_verdict", {})
        issues  = verdict.get("critical_issues", [])

        if self.bus.get("llm_backend_exhausted"):
            decision = "force_write"
            reason   = self.bus.get("llm_backend_exhausted_reason", "")
            logger.warning(
                f"[A7-Naive] Non-retryable backend error -> FORCE WRITE. "
                f"retry={retry}. Reason: {reason}"
            )

        elif retry >= settings.MAX_RETRIES:
            decision = "force_write"
            logger.warning(
                f"[A7-Naive] retry={retry} >= MAX={settings.MAX_RETRIES} "
                f"-> FORCE WRITE"
            )

        else:
            decision = "retry_agent5"

            self.bus.set("flagged_item_indices", [])

            prior_avoid = self.bus.get("avoid_note", "")
            generic_note = (
                "The previous generation attempt did not pass validation. "
                "Please regenerate all items carefully, following the schema "
                "and task constraints exactly."
            )
            if issues:
                issue_summary = "; ".join(str(i) for i in issues[:3])
                generic_note += f" Issues reported: {issue_summary}."
            if prior_avoid:
                generic_note += f" Previously noted: {prior_avoid}"
            self.bus.set("avoid_note", generic_note)

            logger.info(
                f"[A7-Naive] retry={retry}: naive full restart of Agent 5 "
                f"(no diagnostic split, no stall detection, no escalation). "
                f"Issues: {issues}"
            )

        self.bus.set("retry_decision", decision)

        await self.bus.publish(Message(
            agent_id=self.agent_id,
            topic="retry_decided",
            payload={"decision": decision, "retry_count": retry,
                     "critical_issues": issues},
        ))
        return True
