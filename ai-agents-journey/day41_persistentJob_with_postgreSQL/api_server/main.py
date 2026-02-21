from fastapi import FastAPI
from pydantic import BaseModel
import redis
import json
import uuid
from shared.db import get_connection

app=FastAPI()

r = redis.Redis(host="redis", port=6379, decode_responses=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/submit/")
def submit_job(request: QueryRequest):
    
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
        "query": request.query,
        "retries": 0
    }

    r.rpush("task_queue", json.dumps(payload))

    return{
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