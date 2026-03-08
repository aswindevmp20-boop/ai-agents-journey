from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import redis
import json
import uuid
from shared.db import get_connection
from shared.cache import check_cache, hash_query
from api_server.auth import verify_api_key
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from shared.llm_service import generate_response
from shared.logger import get_logger

app=FastAPI()

MAX_QUEUE_LENGTH = 100
MAX_CONTEXT_MESSAGES = 10

logger = get_logger("api")

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
    priority: str = "normal"   # "high" or "normal"
    session_id : str
    
@app.post("/submit/")
@limiter.limit("5/minute")
def submit_job(
    request: Request,
    body: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    
    request_id = str(uuid.uuid4())
    job_id = hash_query(body.query)

    # 🔹 Check cache first
    cached_result = check_cache(body.query)

    if cached_result:
        return {
            "job_id": job_id,
            "request_id": request_id,
            "status": "completed",
            "result": cached_result,
            "cached": True
        }

    # 🔹 Load conversation history
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT role, content FROM conversations WHERE session_id=%s ORDER BY created_at",
        (body.session_id,)
    )

    history = cur.fetchall()
    
    # Keep only recent messages
    if len(history) > MAX_CONTEXT_MESSAGES:
        history = history[-MAX_CONTEXT_MESSAGES:]

    messages = [
        {"role": role, "content": content}
        for role, content in history
    ]

    # Append current user message
    messages.append({"role": "user", "content": body.query})

    # 🔹 Store user message in DB
    cur.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
        (body.session_id, "user", body.query)
    )

    # 🔹 Insert job record
    cur.execute(
        "INSERT INTO jobs (id, status) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (job_id, "queued")
    )

    conn.commit()
    cur.close()
    conn.close()

    # 🔹 Load shedding
    current_queue_length = (
        r.llen("normal_queue") +
        r.llen("high_priority_queue")
    )

    if current_queue_length > MAX_QUEUE_LENGTH:
        return JSONResponse(
            status_code=503,
            content={"error": "System overloaded. Please try again later."}
        )

    # 🔹 Push to Redis
    payload = {
        "job_id": job_id,
        "messages": messages,
        "session_id": body.session_id,
        "retries": 0,
        "request_id": request_id
    }

    queue_name = (
        "high_priority_queue"
        if body.priority == "high"
        else "normal_queue"
    )

    r.rpush(queue_name, json.dumps(payload))

    logger.info(
        "Job submitted",
        extra={"request_id": request_id}
    )

    return {
        "job_id": job_id,
        "request_id": request_id,
        "status": "queued",
        "cached": False
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

@app.post("/stream/")
def stream_response(body: QueryRequest):

    def token_generator():
        messages = [
            {"role": "user", "content": body.query}
        ]

        for token in generate_response(messages):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")