import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List

from utils.logger import logger

@dataclass
class Message:
    agent_id: str
    topic: str
    payload: Any
    status: str = "success"
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    retry_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

class MessageBus:

    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._history: List[Message] = []
        self._subscribers: Dict[str, List[Callable]] = {}

    def set(self, key: str, value: Any) -> None:
        self._context[key] = value
        logger.debug(f"[BUS] SET '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._context)

    def reset_for_retry(self) -> None:
        for k in [
            "generated_output", "generated_format",
            "validation_verdict", "validator_feedback",
            "validation_passed", "recommended_action",
        ]:
            self._context.pop(k, None)
        logger.debug("[BUS] Retry state cleared.")

    async def publish(self, msg: Message) -> None:
        logger.debug(
            f"[BUS] PUB topic='{msg.topic}' from='{msg.agent_id}' status='{msg.status}'"
        )
        self._history.append(msg)
        self._context[f"_last_{msg.topic}"] = msg.payload
        for cb in self._subscribers.get(msg.topic, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(msg)
                else:
                    cb(msg)
            except Exception as exc:
                logger.error(f"[BUS] Subscriber error on '{msg.topic}': {exc}")

    def subscribe(self, topic: str, callback: Callable) -> None:
        self._subscribers.setdefault(topic, []).append(callback)

    def get_history(self) -> List[dict]:
        return [m.to_dict() for m in self._history]

    def get_messages_by_agent(self, agent_id: str) -> List[Message]:
        return [m for m in self._history if m.agent_id == agent_id]
