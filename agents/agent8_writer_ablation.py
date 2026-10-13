
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus

logger = logging.getLogger(__name__)

class Agent8WriterAblation(BaseAgent):

    def __init__(self) -> None:
        super().__init__("Agent8_Writer_Ablation")

    async def execute(self) -> None:
        from config import settings

        bus: MessageBus = self.bus

        run_id   = bus.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        run_kind = bus.get("run_kind", "run")

        if run_kind == "baseline":
            out_dir = settings.OUTPUT_DIR / f"baseline_run_{run_id}"
        elif run_kind == "naive_retry_ablation":
            out_dir = settings.OUTPUT_DIR / f"naive_retry_run_{run_id}"
        else:
            out_dir = settings.OUTPUT_DIR / f"run_{run_id}"

        out_dir.mkdir(parents=True, exist_ok=True)
        bus.set("run_id", run_id)

        generated_output = bus.get("generated_output") or bus.get("carried_items", [])
        verdict          = bus.get("validation_verdict", {})
        retry_count      = bus.get("retry_count", 0)
        force_write      = bus.get("force_write", False)
        elapsed          = bus.get("elapsed_sec", 0.0)
        timing           = bus.get("pipeline_timing", {})

        output_path = out_dir / "output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_output, f, ensure_ascii=False, indent=2)
        logger.info("[Writer] output.json → %s (%d items)", output_path, len(generated_output) if isinstance(generated_output, list) else 1)

        audit = {
            "run_id":           run_id,
            "run_kind":         run_kind,                                   
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "data_path":        bus.get("data_path", ""),
            "user_context":     bus.get("user_context", ""),
            "model_profile":    bus.get("model_profile", ""),
            "retry_count":      retry_count,
            "force_write":      force_write,
            "validation_passed": bus.get("validation_passed", False),
            "validation_verdict": verdict,
            "elapsed_sec":      elapsed,
            "timing":           timing,
            "items_generated":  len(generated_output) if isinstance(generated_output, list) else (1 if generated_output else 0),
            "ablation_note":    (
                "RQ3 ablation run: Agent 7's diagnostic split replaced with "
                "naïve full-regeneration retry. No retry_agent2 path, no stall "
                "detection, no per-item targeting. All other agents unmodified."
            ),
        }
        audit_path = out_dir / "audit_log.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
        logger.info("[Writer] audit_log.json → %s", audit_path)

        bus.set("output_path", str(output_path))
        bus.set("audit_path",  str(audit_path))

        logger.info(
            "[Writer] Run complete — kind=%s  retries=%d  force_write=%s  "
            "items=%d  confidence=%s",
            run_kind,
            retry_count,
            force_write,
            len(generated_output) if isinstance(generated_output, list) else 0,
            verdict.get("confidence", "N/A"),
        )
