
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import settings, apply_model_profile, active_model, MODEL_PROFILES
from core.pipeline import MultiAgentPipeline
from utils.logger import logger

app = FastAPI(
    title="Multi-Agent Generative AI Pipeline API",
    description="Schema-agnostic 8-agent pipeline. Upload any dataset, provide instructions, get AI-generated output.",
    version="1.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

JOBS: dict = {}

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "llm_backend": settings.LLM_BACKEND,
        "generation_backend": settings.GENERATION_BACKEND,
        "model_profile": settings.MODEL_PROFILE or None,
        "active_model": active_model(settings),
    }

@app.get("/model-profiles", tags=["System"])
async def model_profiles():
    return {
        "current": settings.MODEL_PROFILE or None,
        "current_backend": settings.LLM_BACKEND,
        "current_model": active_model(settings),
        "profiles": {
            name: {
                "backend": prof.get("LLM_BACKEND"),
                "model": (
                    prof.get("OLLAMA_MODEL")
                    or prof.get("HUGGINGFACE_MODEL")
                    or prof.get("GROQ_MODEL")
                    or prof.get("NVIDIA_MODEL")
                    or "(uses ANTHROPIC_MODEL)"
                ),
                "description": prof.get("description", ""),
            }
            for name, prof in MODEL_PROFILES.items()
        },
    }

def _apply_profile_or_400(model_profile: Optional[str]) -> None:
    if not model_profile:
        return
    try:
        apply_model_profile(settings, model_profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

def _apply_backend_or_400(backend: Optional[str]) -> None:
    if not backend:
        return
    valid = {"ollama", "anthropic", "huggingface", "groq", "nvidia"}
    if backend not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown backend '{backend}'. Valid: {sorted(valid)}")
    settings.LLM_BACKEND = backend
    settings.GENERATION_BACKEND = backend

@app.post("/run", tags=["Pipeline"])
async def run_pipeline(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input dataset"),
    user_context: str = Form(..., description="Task instructions"),
    async_mode: bool = Form(False, description="Background execution"),
    model_profile: Optional[str] = Form(
        None,
        description="Optional model profile to switch to before running "
                     "(see GET /model-profiles). Affects the whole server "
                     "process, not just this request — see /model-profiles "
                     "docs for details.",
    ),
    backend: Optional[str] = Form(
        None,
        description="Optional explicit backend override for this call — "
                     "'ollama' | 'anthropic' | 'huggingface' | 'groq' | "
                     "'nvidia'. Mirrors the CLI's --backend flag. Applied "
                     "after model_profile, so it can override a profile's "
                     "backend if both are given. Same process-wide caveat "
                     "as model_profile.",
    ),
):
    _apply_profile_or_400(model_profile)
    _apply_backend_or_400(backend)

    job_id = str(uuid.uuid4())
    tmp = settings.TEMP_DIR / f"{job_id}_{file.filename}"
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(tmp, "wb") as fh:
        await fh.write(await file.read())

    if async_mode:
        JOBS[job_id] = {"status": "running", "started": datetime.utcnow().isoformat()}
        background_tasks.add_task(_bg_run, job_id, str(tmp), user_context)
        return {"job_id": job_id, "status": "running", "poll": f"/jobs/{job_id}"}

    try:
        r = await MultiAgentPipeline().run(str(tmp), user_context)
        return _fmt(r)
    except Exception as exc:
        logger.error(f"[API] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp.unlink(missing_ok=True)

async def _bg_run(job_id: str, path: str, ctx: str):
    try:
        r = await MultiAgentPipeline().run(path, ctx)
        JOBS[job_id] = {"status": "done", "completed": datetime.utcnow().isoformat(),
                        "result": _fmt(r)}
    except Exception as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}
    finally:
        Path(path).unlink(missing_ok=True)

def _fmt(r) -> dict:
    v = r.verdict or {}
    return {"success": r.success, "run_id": r.run_id, "retry_count": r.retry_count,
            "output_path": r.output_path, "audit_path": r.audit_path,
            "confidence": float(v.get("confidence", 0)),
            "schema_compliance": v.get("schema_compliance"),
            "critical_issues": v.get("critical_issues", [])}

@app.get("/jobs/{job_id}", tags=["Pipeline"])
async def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]

@app.get("/download/{job_id}", tags=["Pipeline"])
async def download(job_id: str):
    j = JOBS.get(job_id, {})
    if j.get("status") != "done":
        raise HTTPException(404, "Job not ready")
    p = j.get("result", {}).get("output_path", "")
    if not p or not Path(p).exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(p, filename=Path(p).name)

@app.get("/jobs", tags=["Pipeline"])
async def list_jobs():
    return {jid: {"status": d.get("status")} for jid, d in JOBS.items()}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host=settings.API_HOST,
                port=settings.API_PORT, reload=False, workers=1)
