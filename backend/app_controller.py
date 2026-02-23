"""
FastAPI controller for the Career Architect multi-agent system.
Handles job analysis, resume ingestion, and HITL resume for cold email generation.
"""
import base64
import os
import tempfile
from uuid import uuid4
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from backend import utility, agents


def _serialize_state(state: dict) -> dict:
    """Convert state to JSON-serializable form."""
    out = {}
    for k, v in state.items():
        if k == "message":
            out[k] = [
                {"type": getattr(m, "type", "message"), "content": getattr(m, "content", str(m))}
                for m in (v or [])
            ]
        elif hasattr(v, "model_dump"):
            out[k] = v.model_dump()
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = str(v)
    return out


app = FastAPI(title="Career Architect API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobAnalysisRequest(BaseModel):
    resume_base64: str
    location: str
    role: str


class ResumeDecisionRequest(BaseModel):
    thread_id: str
    decision: str


@app.post("/job-analysis")
def job_analysis(request: JobAnalysisRequest):
    """Ingest resume into ChromaDB, run research+analysis, pause at HITL."""
    try:
        pdf_bytes = base64.b64decode(request.resume_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 resume data")
    thread_id = str(uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        utility.ingest_pdfs_to_chroma(tmp_path, thread_id)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Resume ingestion failed: {str(e)}")
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "message": [HumanMessage(content=f"Find {request.role} jobs in {request.location}")],
        "location": request.location,
        "role": request.role,
        "job_list": None,
        "generated_email": None,
        "decision": None,
    }
    try:
        result = agents.graph_app.invoke(initial_state, config=config)
        interrupted = "__interrupt__" in result
        state = {k: v for k, v in result.items() if k != "__interrupt__"}
        interrupt_val = result.get("__interrupt__")
        return {
            "thread_id": thread_id,
            "interrupted": interrupted,
            "interrupt_value": [{"value": getattr(i, "value", i)} for i in (interrupt_val or [])],
            "state": _serialize_state(state),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/resume")
def resume_after_decision(request: ResumeDecisionRequest):
    """Resume graph after HITL with yes/no decision. Uses Command(resume=...)."""
    config = {"configurable": {"thread_id": request.thread_id}}
    decision = request.decision.strip().lower()
    if decision not in ("yes", "no"):
        raise HTTPException(status_code=400, detail="decision must be 'yes' or 'no'")
    try:
        result = agents.graph_app.invoke(
            Command(resume=decision),
            config=config,
        )
        interrupted = "__interrupt__" in result
        state = {k: v for k, v in result.items() if k != "__interrupt__"}
        return {
            "interrupted": interrupted,
            "state": _serialize_state(state),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
