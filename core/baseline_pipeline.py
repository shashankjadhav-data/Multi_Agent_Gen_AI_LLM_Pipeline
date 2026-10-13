import time

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from core.message_bus import MessageBus
from core.pipeline import PipelineResult
from agents.agent1_ingestion import Agent1DataIngestion
from agents.agent8_writer import Agent8OutputWriter
from baseline.single_agent_baseline import SingleAgentBaseline
from utils.logger import logger

console = Console()

class SingleAgentBaselinePipeline:

    async def run(self, input_path: str, user_context: str) -> PipelineResult:
        console.print(Rule("[bold yellow]Single-Agent Baseline — Start[/bold yellow]"))
        bus = MessageBus()
        t0  = time.perf_counter()
        timing = {}

        async def _run_agent(agent, label: str):
            ts = time.perf_counter()
            await agent.run()
            elapsed = round(time.perf_counter() - ts, 1)
            timing[label] = f"{elapsed}s"
            logger.info(f"[BASELINE] {label} completed in {elapsed}s")

        await _run_agent(Agent1DataIngestion(bus, input_path, user_context), "A1_Ingest")
        await _run_agent(SingleAgentBaseline(bus), "Baseline_SingleAgent")
        bus.set("run_kind", "baseline")
        await _run_agent(Agent8OutputWriter(bus), "A8_Write")

        total_elapsed = round(time.perf_counter() - t0, 1)
        timing["TOTAL"] = f"{total_elapsed}s"

        tbl = Table(title="Baseline Agent Timing", show_header=True, header_style="bold yellow")
        tbl.add_column("Step", style="yellow")
        tbl.add_column("Time",  style="white")
        for k, v in timing.items():
            tbl.add_row(k, v)
        console.print(tbl)

        console.print(Rule("[bold green]Single-Agent Baseline Complete[/bold green]"))

        return PipelineResult(
            success     = bus.get("validation_passed", False),
            output_path = bus.get("output_path", ""),
            audit_path  = bus.get("audit_path", ""),
            output      = bus.get("generated_output"),
            verdict     = bus.get("validation_verdict", {}),
            retry_count = 0,
            run_id      = bus.get("run_id", ""),
            elapsed_sec = total_elapsed,
            timing      = timing,
        )
