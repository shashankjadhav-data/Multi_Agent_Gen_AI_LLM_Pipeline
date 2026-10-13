
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

console = Console()

class _TeeStream:
    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):

        return self._original.isatty()

def _setup_run_logging():
    from datetime import datetime
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file = open(log_path, "a", encoding="utf-8")

    banner = (
        f"\n{'=' * 100}\n"
        f"RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Command:   {' '.join(sys.argv)}\n"
        f"{'=' * 100}\n"
    )
    log_file.write(banner)
    log_file.flush()

    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)
    return log_file, log_path

def _finalize_run_logging(log_file, log_path):
    from datetime import datetime
    try:
        log_file.write(f"\nRUN END:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 100}\n")
        log_file.flush()
    except (ValueError, OSError):
        pass
    finally:
        try:
            log_file.close()
        except (ValueError, OSError):
            pass

        if isinstance(sys.stdout, _TeeStream):
            sys.stdout = sys.stdout._original
        if isinstance(sys.stderr, _TeeStream):
            sys.stderr = sys.stderr._original

BANNER = """
 Multi-Agent Generative AI Pipeline
 Schema-Agnostic | RAG | LLM-as-Judge | Ollama | FAISS
"""

def print_result(r: PipelineResult) -> None:
    tbl = Table(title="Pipeline Result", header_style="bold cyan", show_header=True)
    tbl.add_column("Metric", style="cyan", min_width=20)
    tbl.add_column("Value", style="white")
    tbl.add_row("Status",
                "[bold green]SUCCESS[/bold green]" if r.success
                else "[bold yellow]FORCED WRITE[/bold yellow]")
    tbl.add_row("Run ID", r.run_id or "n/a")
    tbl.add_row("Retries", str(r.retry_count))
    tbl.add_row("Output", r.output_path or "n/a")
    tbl.add_row("Audit Log", r.audit_path or "n/a")
    v = r.verdict or {}
    if v:
        tbl.add_row("Confidence",   f"{float(v.get('confidence',0)):.2f}")
        tbl.add_row("Completeness", f"{float(v.get('completeness',0)):.2f}")
        tbl.add_row("Accuracy",     f"{float(v.get('accuracy',0)):.2f}")
        tbl.add_row("Schema OK",    str(v.get('schema_compliance','?')))
        iss = v.get('critical_issues', [])
        tbl.add_row("Issues",
                    "[green]None[/green]" if not iss
                    else f"[red]{len(iss)}[/red]: {str(iss[0])[:55]}")
    console.print(tbl)

def print_model_profiles() -> None:
    from config import MODEL_PROFILES
    tbl = Table(title="Available Model Profiles", header_style="bold cyan", show_header=True)
    tbl.add_column("Profile", style="cyan", min_width=16)
    tbl.add_column("Backend", style="magenta")
    tbl.add_column("Model", style="white")
    tbl.add_column("Notes", style="dim")
    for name, prof in MODEL_PROFILES.items():
        backend = prof.get("LLM_BACKEND", "?")
        model = (
            prof.get("OLLAMA_MODEL")
            or prof.get("HUGGINGFACE_MODEL")
            or prof.get("GROQ_MODEL")
            or prof.get("NVIDIA_MODEL")
            or "(uses ANTHROPIC_MODEL)"
        )
        tbl.add_row(name, backend, model, prof.get("description", ""))
    console.print(tbl)
    console.print(
        "\nUsage: [bold]python main.py --input <file> --context \"...\" "
        "--model-profile <name>[/bold]"
        "\nOr set in .env:  [bold]MODEL_PROFILE=<name>[/bold]"
        "\n\nEach profile uses exactly ONE model — no dual-model VRAM contention.\n"
    )

async def _run(path: str, ctx: str, baseline: bool = False) -> PipelineResult:
    console.print(BANNER, style="bold blue")
    if baseline:
        from core.baseline_pipeline import SingleAgentBaselinePipeline
        return await SingleAgentBaselinePipeline().run(path, ctx)
    from core.pipeline import MultiAgentPipeline
    return await MultiAgentPipeline().run(path, ctx)

def _main_impl() -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Agent Generative AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",      "-i", required=False,
                   help="Input data file (JSON/CSV/XLSX/TXT/PDF/DOCX)")
    p.add_argument("--context",    "-c", required=False,
                   help="Plain-language task instructions")
    p.add_argument("--model",      "-m", default=None,
                   help="Override the active model directly (e.g. qwen2.5:7b-instruct "
                        "for Ollama, or a Hugging Face repo id like Qwen/Qwen3-8B if "
                        "--backend/--model-profile/LLM_BACKEND selects huggingface). "
                        "Takes precedence over --model-profile. NOTE: there is "
                        "intentionally no separate 'fast model' flag — this pipeline "
                        "always uses exactly one model per backend to avoid loading "
                        "two models into VRAM at once.")
    p.add_argument("--backend",    choices=["ollama", "anthropic", "huggingface", "groq", "nvidia"], default=None,
                   help="Which backend --model applies to. Without this, --model "
                        "silently targets whatever backend is currently active "
                        "(ollama by default) — e.g. `--model Qwen/Qwen3-8B` alone "
                        "does NOT switch to Hugging Face, it just sets an invalid "
                        "Ollama model name. Use --backend huggingface --model "
                        "Qwen/Qwen3-8B, or the equivalent --model-profile, instead.")
    p.add_argument("--model-profile",    default=None,
                   help="Named model profile — sets backend + model together. "
                        "Run --list-model-profiles to see all options.")
    p.add_argument("--list-model-profiles", action="store_true",
                   help="Print available model profiles and exit.")
    p.add_argument("--baseline", action="store_true",
                   help="Run the single-agent baseline instead of the 8-agent "
                        "pipeline: one LLM call does schema-inference + "
                        "task-understanding + generation together, no RAG "
                        "grounding, no deterministic repair, no quality-based "
                        "retry. Scored by the SAME Agent 6 validator against "
                        "the SAME Agent 4 ground truth as a normal run, so "
                        "the resulting audit_log.json is directly comparable "
                        "to a full-pipeline run on the same input/context.")
    args = p.parse_args()

    if args.list_model_profiles:
        print_model_profiles()
        sys.exit(0)

    if not args.input or not args.context:
        p.error("--input and --context are required (unless using --list-model-profiles)")

    if not Path(args.input).exists():
        console.print(f"[bold red]File not found:[/bold red] {args.input}")
        sys.exit(1)

    from config import settings, apply_model_profile, _sync_single_model_aliases

    if args.model_profile:
        try:
            apply_model_profile(settings, args.model_profile)
            console.print(f"[cyan]Applied model profile:[/cyan] {args.model_profile}")
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            sys.exit(1)

    if args.backend:
        settings.LLM_BACKEND = args.backend
        settings.GENERATION_BACKEND = args.backend
        console.print(f"[cyan]Using backend:[/cyan] {args.backend}")

    if args.model:
        from config import MODEL_PROFILES
        if args.model in MODEL_PROFILES and not args.model_profile:
            apply_model_profile(settings, args.model)
            console.print(
                f"[cyan]'{args.model}' matches an existing model profile — "
                f"applying it in full (backend, model, and any pinned "
                f"provider).[/cyan]"
            )
        else:

            looks_like_hf_repo = "/" in args.model and ":" not in args.model
            backend_was_explicit = bool(args.backend or args.model_profile)
            if looks_like_hf_repo and settings.LLM_BACKEND != "huggingface" and not backend_was_explicit:
                console.print(
                    f"[cyan]'{args.model}' looks like a Hugging Face repo id — "
                    f"switching backend to huggingface automatically.[/cyan]"
                )
                settings.LLM_BACKEND = "huggingface"
                settings.GENERATION_BACKEND = "huggingface"
            elif looks_like_hf_repo and settings.LLM_BACKEND != "huggingface":

                console.print(
                    f"[bold yellow]Warning:[/bold yellow] '{args.model}' looks like a "
                    f"Hugging Face repo id, but you explicitly set backend "
                    f"'{settings.LLM_BACKEND}'. Proceeding with your explicit choice — "
                    f"this will likely fail as an Ollama model."
                )
            if settings.LLM_BACKEND == "huggingface":
                settings.HUGGINGFACE_MODEL = args.model
            else:
                settings.OLLAMA_MODEL = args.model
        _sync_single_model_aliases(settings)
        console.print(f"[cyan]Using model:[/cyan] {args.model}")

    from config import active_model
    console.print(
        f"[cyan]Model configuration:[/cyan] "
        f"profile={settings.MODEL_PROFILE or '(none)'} "
        f"backend={settings.LLM_BACKEND} "
        f"model={active_model(settings)}"
    )

    if args.baseline:
        console.print("[yellow]Running the SINGLE-AGENT BASELINE (not the 8-agent pipeline).[/yellow]")
    result = asyncio.run(_run(args.input, args.context, baseline=args.baseline))
    print_result(result)
    sys.exit(0 if result.success else 1)

def main() -> None:
    log_file, log_path = _setup_run_logging()
    try:
        _main_impl()
    finally:
        _finalize_run_logging(log_file, log_path)

if __name__ == "__main__":
    main()
