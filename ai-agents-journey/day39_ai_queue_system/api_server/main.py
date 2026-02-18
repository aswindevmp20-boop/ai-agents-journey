from fastapi import FastAPI
from pydantic import BaseModel
import redis
import json
import uuid

app=FastAPI()

r = redis.Redis(host="redis", port=6379, decode_responses=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/submit/")
def submit_job(request: QueryRequest):
    job_id = str(uuid.uuid4())

    payload = {
        "job_id": job_id,
        "query": request.query
    }

    r.rpush("task_queue", json.dumps(payload))

    return{
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = r.get(f"result: {job_id}")

    if not result:
        return {"status": "processing"}

    return {
        "status":"completed",
        "result": result
    }