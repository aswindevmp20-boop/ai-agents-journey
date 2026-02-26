import redis
import json
from shared.llm_service import generate_response
import time
from shared.db import get_connection

r = redis.Redis(host="redis", port=6379, decode_responses=True)


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

    job_id = data["job_id"]
    query = data["query"]

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
        result = generate_response([
            {"role": "user", "content": query}
        ])

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