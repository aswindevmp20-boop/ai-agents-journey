import redis
import json
from shared.llm_service import generate_response
import time
from shared.db import get_connection
from shared.logger import get_logger
from shared.planner_agent import create_plan, execute_plan
from shared.research_agent import run_research_agent

r = redis.Redis(host="redis", port=6379, decode_responses=True)

logger = get_logger("worker"
                    )
print("Worker started...")

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, 
                status TEXT,
                result TEXT,
                error TEXT
            )
         """)
    
    conn.commit()
    cur.close()
    conn.close()

init_db()

while True:
    #updated
    queue, task = r.blpop(["high_priority_queue", "normal_queue"])
    data = json.loads(task)

    request_id = data.get("request_id")
    job_id = data["job_id"]
    query = data["query"]

    logger.info(
        "processing job",
        extra = {"request_id": request_id}
    )

    print(f"Processing job {job_id}")

    # Update status to processing
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE jobs SET status=%s WHERE id=%s",
        ("processing", job_id)
    )
    conn.commit()
    cur.close()
    conn.close()

    try:
        query = messages[-1]["content"]
        plan = create_plan(query)
        result = run_research_agent(query)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status=%s, result=%s WHERE id=%s",
            ("completed", result, job_id)
        )
        conn.commit()
        cur.close()
        conn.close()

        print(f"Job {job_id} completed.")

        logger.info(
            "job completed",
            extra = {"request_id": request_id}
        )

    except Exception as e:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status=%s, error=%s WHERE id=%s",
            ("failed", str(e), job_id)
        )
        conn.commit()
        cur.close()
        conn.close()

        print(f"Job {job_id} failed.")

        logger.error(
            "job failed",
            extra = {"request_id": request_id}
        )