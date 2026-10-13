
from pathlib import Path

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.file_loader import (
    load_file, serialize_for_llm, chunk_data,
    smart_sample_for_rag, select_diverse_examples,
)
from utils.logger import logger

class Agent1DataIngestion(BaseAgent):

    def __init__(self, bus: MessageBus, input_path: str, user_context: str):
        super().__init__("agent1", "Agent 1 · Data Ingestion", bus)
        self.input_path  = input_path
        self.user_context = user_context

    async def execute(self) -> bool:
        from config import settings

        data, fmt, sample = load_file(self.input_path)
        full_text = serialize_for_llm(data, fmt, max_chars=80_000)

        rag_chunks = smart_sample_for_rag(
            data, fmt,
            max_chunks=settings.MAX_FAISS_CHUNKS,
            chunk_size=settings.CHUNK_SIZE,
        )

        examples = select_diverse_examples(data, fmt, n=settings.MAX_EXAMPLES_IN_PROMPT)

        n_records = len(data) if isinstance(data, list) else "n/a"
        logger.info(
            f"[A1] Format={fmt.upper()}  Records={n_records}  "
            f"RAG_chunks={len(rag_chunks)}  TotalChars={len(full_text)}"
        )

        self.bus.set("raw_data",       data)
        self.bus.set("data_format",    fmt)
        self.bus.set("data_sample",    sample)
        self.bus.set("data_rag_chunks", rag_chunks)
        self.bus.set("data_examples",   examples)
        self.bus.set("data_full_text",  full_text)
        self.bus.set("input_path",     self.input_path)
        self.bus.set("user_context",   self.user_context)
        self.bus.set("retry_count",    0)

        await self.bus.publish(Message(
            agent_id=self.agent_id,
            topic="data_ingested",
            payload={
                "format": fmt,
                "records": n_records,
                "rag_chunks": len(rag_chunks),
                "total_chars": len(full_text),
                "input_file": Path(self.input_path).name,
            },
        ))
        return True
