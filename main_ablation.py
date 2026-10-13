
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║  RQ3 NAÏVE-RETRY ABLATION — Multi-Agent LLM Pipeline            ║
║  Agent 7: diagnostic split DISABLED (naïve full restart only)   ║
║  All other agents: unmodified production code                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RQ3 Naïve-Retry Ablation — 8-agent pipeline with Agent 7 "
                    "replaced by naïve full-regeneration retry."
    )
    p.add_argument("--input",          required=True,
                   help="Path to the input dataset JSON file.")
    p.add_argument("--context",        required=True,
                   help="Plain-language task description.")
    p.add_argument("--model-profile",  required=True, dest="model_profile",
                   help="Model profile name (e.g. qwen2.5:7b-instruct).")
    p.add_argument("--log-level",      default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO).")
    return p

async def _run(data_path: str, context: str, model_profile: str) -> dict:
    from config import settings, apply_model_profile

    apply_model_profile(settings, model_profile)

    console.print(f"[dim]Model profile: {model_profile}[/dim]")
    console.print(f"[dim]Input:         {data_path}[/dim]")
    console.print(
        f"[dim]Context:       "
        f"{context[:80]}{'...' if len(context) > 80 else ''}[/dim]\n"
    )

    from core.ablation_pipeline import NaiveRetryAblationPipeline
    return await NaiveRetryAblationPipeline().run(data_path, context)

def main() -> None:
    p   = _build_parser()
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    console.print(BANNER, style="bold yellow")

    if not Path(args.input).exists():
        console.print(f"[red]ERROR: input file not found: {args.input}[/red]")
        sys.exit(1)

    result = asyncio.run(_run(args.input, args.context, args.model_profile))

    console.print(Rule("[bold]Ablation Run Summary[/bold]"))
    success = result.get("success", False)
    status  = "[green]SUCCESS[/green]" if success else "[red]FORCED WRITE[/red]"
    verdict = result.get("verdict", {})

    console.print(
        f"  Status:       {status}\n"
        f"  Retries:      {result.get('retry_count', 'N/A')}\n"
        f"  Confidence:   {verdict.get('confidence', 'N/A')}\n"
        f"  Completeness: {verdict.get('completeness', 'N/A')}\n"
        f"  Accuracy:     {verdict.get('accuracy', 'N/A')}\n"
        f"  Elapsed:      {result.get('elapsed_sec', 0):.1f}s\n"
        f"  Output:       {result.get('output_path', 'N/A')}\n"
        f"  Audit log:    {result.get('audit_path', 'N/A')}"
    )

    issues = verdict.get("critical_issues", [])
    if issues:
        console.print("\n  [yellow]Unresolved issues:[/yellow]")
        for iss in issues:
            console.print(f"    • {iss}")

    console.print(
        "\n[dim]Share the audit_log.json with the researcher for RQ3 analysis.[/dim]"
    )

if __name__ == "__main__":
    main()
