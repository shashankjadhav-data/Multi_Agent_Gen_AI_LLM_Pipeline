import sys
from loguru import logger as _logger
from rich.console import Console
from rich.rule import Rule

def _init_logger():
    from config import settings
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    _logger.remove()
    _logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}",
        colorize=True,
    )
    _logger.add(
        settings.LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log",
        rotation="50 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )
    return _logger

logger = _init_logger()
console = Console()

def log_agent_start(name: str):
    console.print(Rule(f"[bold cyan]{name}[/bold cyan]"))
    logger.info(f"START >> {name}")

def log_agent_done(name: str, msg: str = ""):
    console.print(f"  [bold green]DONE[/bold green]  {name}  {msg}")
    logger.info(f"DONE  >> {name} | {msg}")

def log_agent_error(name: str, err: str):
    console.print(f"  [bold red]ERROR[/bold red] {name}  {err}")
    logger.error(f"ERROR >> {name} | {err}")
