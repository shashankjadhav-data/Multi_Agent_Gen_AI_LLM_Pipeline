from abc import ABC, abstractmethod

from core.message_bus import MessageBus, Message
from core.llm_client import LLMClient, llm
from utils.logger import log_agent_start, log_agent_done, log_agent_error

class BaseAgent(ABC):

    def __init__(self, agent_id: str, name: str, bus: MessageBus):
        self.agent_id = agent_id
        self.name = name
        self.bus = bus
        self.llm: LLMClient = llm

    async def run(self) -> bool:
        log_agent_start(self.name)
        try:
            result = await self.execute()
            log_agent_done(self.name)
            return result
        except Exception as exc:
            log_agent_error(self.name, str(exc))
            await self.bus.publish(Message(
                agent_id=self.agent_id,
                topic=f"{self.agent_id}_error",
                payload={"error": str(exc)},
                status="error",
            ))
            raise

    @abstractmethod
    async def execute(self) -> bool: ...

    def _sys(self, c: str) -> dict: return {"role": "system", "content": c}
    def _usr(self, c: str) -> dict: return {"role": "user",   "content": c}
    def _asst(self, c: str) -> dict: return {"role": "assistant", "content": c}
