import asyncio
import logging
import time

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from agents.agent1_ingestion   import Agent1DataIngestion
from agents.agent2_schema      import Agent2SchemaInspector
from agents.agent3_tasks       import Agent3TaskRecognizer
from agents.agent4_features    import Agent4FeatureExtractor
from agents.agent5_generator   import Agent5OutputGenerator
from agents.agent6_validator   import Agent6QualityValidator
from agents.agent7_naive_retry import Agent7NaiveRetry
from agents.agent8_writer      import Agent8OutputWriter
from core.message_bus          import MessageBus

logger  = logging.getLogger(__name__)
console = Console()

class NaiveRetryAblationPipeline:

    async def run(self, data_path: str, context: str) -> dict:
        from config import settings

        bus = MessageBus()

        bus.set("run_kind", "naive_retry_ablation")

        console.print(
            Rule("[bold yellow]Naïve-Retry Ablation Pipeline — RQ3 Evaluation[/bold yellow]")
        )
        console.print(
            "[yellow]Agent 7: diagnostic split DISABLED — naïve full restart "
            "on every failure, no escalation, no stall detection.[/yellow]\n"
        )

        timing: dict[str, str] = {}
        t_total = time.perf_counter()

        console.print("[dim]Agent 1 · Data Ingestion[/dim]")
        agent1 = Agent1DataIngestion(bus, data_path, context)
        t0 = time.perf_counter()
        await agent1.run()
        timing["Agent1_Ingestion"] = f"{time.perf_counter() - t0:.1f}s"

        console.print("[dim]Agent 2 · Schema Inspector[/dim]")
        agent2 = Agent2SchemaInspector(bus)
        t0 = time.perf_counter()
        await agent2.run()
        timing["Agent2_Schema"] = f"{time.perf_counter() - t0:.1f}s"

        console.print("[dim]Agent 3 · Task Recognizer[/dim]")
        agent3 = Agent3TaskRecognizer(bus)
        t0 = time.perf_counter()
        await agent3.run()
        timing["Agent3_Task"] = f"{time.perf_counter() - t0:.1f}s"

        console.print("[dim]Agent 4 · Feature Extractor[/dim]")
        agent4 = Agent4FeatureExtractor(bus)
        t0 = time.perf_counter()
        await agent4.run()
        timing["Agent4_Features"] = f"{time.perf_counter() - t0:.1f}s"

        console.print(
            "\n[dim]Retry loop: Agent 5 (Generate) → Agent 6 (Validate) "
            "→ Agent 7 Naive Retry[/dim]\n"
        )

        agent5 = Agent5OutputGenerator(bus)
        agent6 = Agent6QualityValidator(bus)
        agent7 = Agent7NaiveRetry(bus)
        agent8 = Agent8OutputWriter(bus)

        loop_num = 0
        while True:
            loop_num += 1
            console.print(f"[bold cyan]── Loop {loop_num} ──[/bold cyan]")

            t0 = time.perf_counter()
            await agent5.run()
            timing[f"Loop{loop_num}_Generator"] = f"{time.perf_counter() - t0:.1f}s"

            t0 = time.perf_counter()
            await agent6.run()
            timing[f"Loop{loop_num}_Validator"] = f"{time.perf_counter() - t0:.1f}s"

            t0 = time.perf_counter()
            await agent7.run()
            timing[f"Loop{loop_num}_RetryDecision"] = f"{time.perf_counter() - t0:.2f}s"

            decision   = bus.get("retry_decision", "")
            carried    = len(bus.get("carried_items", []) or [])
            verdict    = bus.get("validation_verdict", {})
            confidence = verdict.get("confidence", "N/A")
            n_issues   = len(verdict.get("critical_issues", []))

            console.print(
                f"  decision={decision}  items_accumulated={carried}  "
                f"confidence={confidence}  issues={n_issues}"
            )

            if decision == "force_write":
                logger.warning("[Ablation] Force-write triggered — exiting retry loop.")
                break
            if decision == "done":
                logger.info("[Ablation] Validation passed — exiting retry loop.")
                break

            if decision not in ("retry_agent5", "retry_agent2",
                                "force_write", "done"):
                logger.error(
                    "[Ablation] Unexpected retry_decision=%r — forcing write.", decision
                )
                bus.set("force_write", True)
                break

        console.print("[dim]Agent 8 · Output Writer[/dim]")
        t0 = time.perf_counter()
        await agent8.run()
        timing["Agent8_Writer"] = f"{time.perf_counter() - t0:.1f}s"

        elapsed_total = time.perf_counter() - t_total
        verdict     = bus.get("validation_verdict", {})
        retry_count = bus.get("retry_count", 0)
        force_write = bus.get("retry_decision") == "force_write"

        tbl = Table(
            title="[bold]Naïve-Retry Ablation — Timing[/bold]",
            show_header=True,
        )
        tbl.add_column("Step",  style="yellow", no_wrap=True)
        tbl.add_column("Time",  style="white")
        for k, v in timing.items():
            tbl.add_row(k, str(v))
        console.print(tbl)

        colour = "red" if force_write else "green"
        label  = "FORCED WRITE" if force_write else "SUCCESS"
        console.print(
            f"\n[bold {colour}]{label}[/bold {colour}]  "
            f"Retries: {retry_count}  |  "
            f"Confidence: {verdict.get('confidence', 'N/A')}  |  "
            f"Completeness: {verdict.get('completeness', 'N/A')}  |  "
            f"Elapsed: {elapsed_total:.1f}s"
        )
        console.print(Rule("[bold yellow]Ablation Run Complete[/bold yellow]"))

        return {
            "success":      bus.get("validation_passed", False),
            "output_path":  bus.get("output_path", ""),
            "audit_path":   bus.get("audit_path", ""),
            "verdict":      verdict,
            "retry_count":  retry_count,
            "force_write":  force_write,
            "elapsed_sec":  elapsed_total,
            "timing":       timing,
        }
