from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

class Agent7IncrementRetry(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("agent7", "Agent 7 · Increment Retry", bus)

    async def execute(self) -> bool:
        from config import settings
        retry = self.bus.get("retry_count", 0) + 1
        self.bus.set("retry_count", retry)

        action = self.bus.get("recommended_action", "retry_agent5")
        verdict = self.bus.get("validation_verdict", {})
        issues = verdict.get("critical_issues", [])

        quantity_only = bool(issues) and all(
            (
                "quantity mismatch" in str(i).lower()
                or "missing" in str(i).lower()
                or "output is empty" in str(i).lower()
                or "output is none" in str(i).lower()
            )
            for i in issues
        )

        if self.bus.get("llm_backend_exhausted"):
            decision = "force_write"
            reason = self.bus.get("llm_backend_exhausted_reason", "")
            logger.warning(
                f"[A7] Non-retryable backend error detected (billing/quota) "
                f"-> FORCE WRITE immediately, regardless of retry={retry}. "
                f"Reason: {reason}"
            )
        elif retry >= settings.MAX_RETRIES:
            decision = "force_write"
            logger.warning(
                f"[A7] retry={retry} >= MAX={settings.MAX_RETRIES} -> FORCE WRITE"
            )
        elif quantity_only:
            current_count = len(self.bus.get("carried_items", []) or [])
            last_count = self.bus.get("_last_quantity_shortfall_count")
            already_deep_retried = bool(self.bus.get("_stall_deep_retry_used"))
            if last_count is not None and current_count == last_count:
                if already_deep_retried:
                    decision = "retry_agent5"
                    logger.warning(
                        f"[A7] retry={retry}: still zero progress "
                        f"({current_count} items) even after a deep retry — "
                        f"NOT deep-retrying again, since re-extracting features "
                        f"from an unchanged input file is deterministic and "
                        f"already produced identical constraints. Falling back "
                        f"to a simple retry so the time goes to generation "
                        f"instead. Issues: {issues}"
                    )
                else:
                    decision = "retry_agent2"
                    self.bus.set("_stall_deep_retry_used", True)
                    logger.warning(
                        f"[A7] retry={retry}: quantity shortfall made ZERO progress "
                        f"since the last attempt ({current_count} items both times) "
                        f"-> escalating to deep retry (A2+A4) instead of repeating "
                        f"the same simple retry. Issues: {issues}"
                    )
            else:
                decision = "retry_agent5"
                self.bus.set("_stall_deep_retry_used", False)
                logger.info(
                    f"[A7] retry={retry}: quantity-only shortfall -> "
                    f"simple retry (A5, resumed). Issues: {issues}"
                )
            self.bus.set("_last_quantity_shortfall_count", current_count)
        elif action == "retry_agent2" and retry > 1:
            decision = "retry_agent2"
            logger.info(f"[A7] retry={retry}: deep retry (A2+A4). Issues: {issues}")
        else:
            decision = "retry_agent5"
            logger.info(f"[A7] retry={retry}: simple retry (A5). Issues: {issues}")

        self.bus.set("retry_decision", decision)

        await self.bus.publish(Message(
            agent_id=self.agent_id, topic="retry_decided",
            payload={"decision": decision, "retry_count": retry,
                     "critical_issues": issues},
        ))
        return True
