import json
import uuid
from datetime import datetime
from pathlib import Path

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

class Agent8OutputWriter(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("agent8", "Agent 8 · Output Writer", bus)

    async def execute(self) -> bool:
        from config import settings, active_model
        output  = self.bus.get("generated_output")
        fmt     = self.bus.get("generated_format", "json")
        verdict = self.bus.get("validation_verdict", {})
        forced  = self.bus.get("retry_decision") == "force_write"
        user_ctx = self.bus.get("user_context", "")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        run_kind = self.bus.get("run_kind", "full_pipeline")
        prefix = "baseline_run" if run_kind == "baseline" else "run"
        out_dir: Path = settings.OUTPUT_DIR / f"{prefix}_{ts}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.bus.set("run_id", run_id)

        out_path = self._save_output(output, fmt, out_dir)

        audit = {
            "run_id": run_id,
            "run_kind": run_kind,
            "timestamp": datetime.utcnow().isoformat(),
            "user_context": user_ctx,
            "forced_write": forced,
            "validation_passed": self.bus.get("validation_passed", False),
            "retry_count": self.bus.get("retry_count", 0),
            "validation_verdict": verdict,
            "llm_config": {
                "model_profile": settings.MODEL_PROFILE or None,
                "llm_backend": settings.LLM_BACKEND,
                "generation_backend": settings.GENERATION_BACKEND,
                "active_model": active_model(settings),
                "ollama_model": settings.OLLAMA_MODEL,
                "anthropic_model": settings.ANTHROPIC_MODEL,
                "huggingface_model": settings.HUGGINGFACE_MODEL,
            },
            "task_spec": self.bus.get("task_spec", {}),
            "inferred_schema": self.bus.get("inferred_schema", {}),
            "extracted_features": self.bus.get("extracted_features", {}),
            "canonical_fields": self.bus.get("canonical_fields", []),
            "categorical_enums": self.bus.get("categorical_enums", {}),
            "input_file": self.bus.get("input_path", ""),
            "output_file": str(out_path),
            "pipeline_messages": self.bus.get_history(),
        }
        audit_path = out_dir / "audit_log.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(f"[A8] Output  -> {out_path}")
        logger.info(f"[A8] Audit   -> {audit_path}")
        logger.info(
            f"[A8] run_id={run_id} valid={self.bus.get('validation_passed', False)} "
            f"forced={forced} model={active_model(settings)}"
        )

        self.bus.set("output_path", str(out_path))
        self.bus.set("audit_path", str(audit_path))

        await self.bus.publish(Message(
            agent_id=self.agent_id, topic="output_written",
            payload={"output_path": str(out_path),
                     "audit_path": str(audit_path), "run_id": run_id},
        ))
        return True

    def _save_output(self, output, fmt: str, out_dir: Path) -> Path:
        if fmt == "json":
            path = out_dir / "output.json"
            if isinstance(output, (list, dict)):
                path.write_text(
                    json.dumps(output, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8")
            else:
                from core.llm_client import llm
                try:
                    parsed = llm.extract_json(str(output))
                    path.write_text(
                        json.dumps(parsed, indent=2, default=str, ensure_ascii=False),
                        encoding="utf-8")
                except ValueError:
                    path = out_dir / "output_raw.txt"
                    path.write_text(str(output), encoding="utf-8")
            return path
        elif fmt == "csv":
            path = out_dir / "output.csv"
            path.write_text(str(output), encoding="utf-8")
            return path
        elif fmt in ("markdown", "md"):
            path = out_dir / "output.md"
            path.write_text(str(output), encoding="utf-8")
            return path
        elif fmt == "image":
            path = out_dir / "output_image.png"
            code = str(output).replace("output_image.png", str(path))
            try:
                exec(compile(code, "<img_code>", "exec"), {"__name__": "__main__"})
            except Exception as exc:
                logger.error(f"[A8] Image exec failed: {exc}")
                path = out_dir / "image_code.py"
                path.write_text(str(output), encoding="utf-8")
            return path
        else:
            path = out_dir / f"output.{fmt}"
            path.write_text(str(output), encoding="utf-8")
            return path
