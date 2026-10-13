import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from core.message_bus import MessageBus
from agents.agent1_ingestion  import Agent1DataIngestion
from agents.agent2_schema     import Agent2SchemaInspector
from agents.agent3_tasks      import Agent3TaskRecognizer
from agents.agent4_features   import Agent4FeatureExtractor
from agents.agent5_generator  import Agent5OutputGenerator
from agents.agent6_validator  import Agent6QualityValidator
from agents.agent7_retry      import Agent7IncrementRetry
from agents.agent8_writer     import Agent8OutputWriter
from config import settings
from utils.logger import logger

console = Console()

@dataclass
class PipelineResult:
    success:      bool
    output_path:  str
    audit_path:   str
    output:       Any
    verdict:      dict
    retry_count:  int
    run_id:       str   = ""
    elapsed_sec:  float = 0.0
    timing:       dict  = field(default_factory=dict)

class MultiAgentPipeline:

    def __init__(self):
        self._feat_agent: Optional[Agent4FeatureExtractor] = None

    async def run(self, input_path: str, user_context: str) -> PipelineResult:
        console.print(Rule("[bold blue]Multi-Agent Generative AI Pipeline — Start[/bold blue]"))
        bus    = MessageBus()
        t0     = time.perf_counter()
        timing = {}

        async def _run_agent(agent, label: str):
            ts = time.perf_counter()
            await agent.run()
            elapsed = round(time.perf_counter() - ts, 1)
            timing[label] = f"{elapsed}s"
            logger.info(f"[PIPELINE] {label} completed in {elapsed}s")

        await _run_agent(Agent1DataIngestion(bus, input_path, user_context), "A1_Ingest")

        await _run_agent(Agent2SchemaInspector(bus), "A2_Schema")

        await _run_agent(Agent3TaskRecognizer(bus), "A3_Tasks")

        if self._feat_agent is None:
            self._feat_agent = Agent4FeatureExtractor(bus)
        else:
            self._feat_agent.bus = bus
        await _run_agent(self._feat_agent, "A4_Features")

        for loop in range(1, settings.MAX_RETRIES + 2):
            logger.info(f"[PIPELINE] Gen/Val loop {loop}/{settings.MAX_RETRIES + 1}")

            await _run_agent(Agent5OutputGenerator(bus), f"A5_Gen_L{loop}")
            await _run_agent(Agent6QualityValidator(bus), f"A6_Val_L{loop}")

            passed = bus.get("validation_passed", False)
            if passed:
                logger.info(f"[PIPELINE] Validation PASSED at loop {loop}.")
                bus.set("carried_items", [])
                break

            logger.warning(f"[PIPELINE] Validation FAILED at loop {loop}.")
            await _run_agent(Agent7IncrementRetry(bus), f"A7_Retry_L{loop}")
            decision = bus.get("retry_decision", "force_write")

            if decision == "force_write":
                logger.warning("[PIPELINE] Max retries reached — force write.")
                break

            previous_items = bus.get("generated_output", []) or []
            previous_feedback = bus.get("validator_feedback", "") or ""
            previous_flagged = bus.get("flagged_item_indices", []) or []
            bus.reset_for_retry()
            bus.set("carried_items", previous_items)
            if previous_feedback:
                bus.set("validator_feedback", previous_feedback)
            if previous_flagged:
                bus.set("flagged_item_indices", previous_flagged)

            if decision == "retry_agent2":
                logger.info(
                    "[PIPELINE] Deep retry: re-running A2 + A4 "
                    "(A2's schema cache bypassed so this retry actually "
                    "re-inspects instead of replaying the prior result)."
                )
                await _run_agent(
                    Agent2SchemaInspector(bus, force_recompute=True), f"A2_Schema_R{loop}"
                )
                self._feat_agent._index  = None
                self._feat_agent._indexed = []
                self._feat_agent.bus     = bus
                await _run_agent(self._feat_agent, f"A4_Features_R{loop}")
            else:
                logger.info(
                    f"[PIPELINE] Simple retry: re-running A5 "
                    f"(resuming {len(previous_items)} accepted item(s))."
                )

        await _run_agent(Agent8OutputWriter(bus), "A8_Write")

        total_elapsed = round(time.perf_counter() - t0, 1)
        timing["TOTAL"] = f"{total_elapsed}s"

        tbl = Table(title="Agent Timing", show_header=True, header_style="bold cyan")
        tbl.add_column("Agent", style="cyan")
        tbl.add_column("Time",  style="white")
        for k, v in timing.items():
            tbl.add_row(k, v)
        console.print(tbl)

        console.print(Rule("[bold green]Pipeline Complete[/bold green]"))

        return PipelineResult(
            success      = bus.get("validation_passed", False),
            output_path  = bus.get("output_path", ""),
            audit_path   = bus.get("audit_path", ""),
            output       = bus.get("generated_output"),
            verdict      = bus.get("validation_verdict", {}),
            retry_count  = bus.get("retry_count", 0),
            run_id       = bus.get("run_id", ""),
            elapsed_sec  = total_elapsed,
            timing       = timing,
        )
