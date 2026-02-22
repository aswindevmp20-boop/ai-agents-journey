from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
import redis
import json
import uuid
from shared.db import get_connection
from api_server.auth import verify_api_key
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

app=FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code = 429,
        content={"error": "Rate limit exceeded"}
    )

r = redis.Redis(host="redis", port=6379, decode_responses=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/submit/")
@limiter.limit("5/minute")
def submit_job(
    request: Request,                     
    body: QueryRequest,                   
    api_key: str = Depends(verify_api_key)
):

    job_id = str(uuid.uuid4())

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO jobs (id,status) VALUES (%s,%s)",
        (job_id, "queued")
    )

    conn.commit()
    cur.close()
    conn.close()

    payload = {
        "job_id": job_id,
        "query": body.query,             
        "retries": 0
    }

    r.rpush("task_queue", json.dumps(payload))

    return {
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT status, result, error FROM jobs WHERE id = %s",
        (job_id,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return {"error": "Job not found"}
    
    status, result, error = row

    return {
        "status": status,
        "result": result,
        "error": error
    }
