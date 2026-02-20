from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from utility import ingest_pdfs_to_chroma
import agents 
from uuid import uuid4
from langgraph.types import Command

app = FastAPI()

class job_analysis_request(BaseModel):
    resume_file: str
    location: str
    role: str

class email_generation_request(BaseModel):
    thread_id: uuid4
    job_analysis: str
    decision: str


@app.post("/job-analysis")
def job_analysis(request: job_analysis_request):
    ingest_pdfs_to_chroma(request.resume_file)
    thread_id = str(uuid4())
    result = agents.app.invoke(
        {
            "message":[],
            "location":request.location,
            "role":request.role
        },
        config={"configurable":{"thread_id":thread_id}}
    )

    return {"thread_id": thread_id , "result":result}

@app.post("/generate-cold-email")
def generate_cold_email(email_request:email_generation_request):
    result = agents.app.invoke(
        Command(update={"decision":email_request.decision}),
        config={"configurable": {"thread_id":email_request.thread_id}}
    )
    return result
