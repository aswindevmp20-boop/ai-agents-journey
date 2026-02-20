import redis
import json
from shared.llm_service import generate_response
import time

r = redis.Redis(host="redis", port=6379, decode_responses=True)

MAX_RETRIES = 3

print("Worker started...")

while True:
    _, task = r.blpop("task_queue")
    data = json.loads(task)

    job_id = data["job_id"]
    query = data["query"]
    retries = data.get("retries", 0)

    print(f"Processing job {job_id}, attempt {retries + 1}")

    try:
        result = generate_response([
            {"role": "user", "content": query}
        ])

        r.set(f"result:{job_id}", json.dumps({
            "status": "completed",
            "result": result
        }))

        print(f"job {job_id} completed.")

    except Exception as e:
        print(f"Job {job_id} failed: {str(e)}")

        if retries < MAX_RETRIES:
            print(f"Retrying job {job_id}...")
            data["retries"] = retries + 1
            time.sleep(2)
            r.rpush("task_queue", json.dumps(data))
        else:
            r.set(f"result:{job_id}", json.dumps({
                "status": "failed",
                "error": str(e)
            }))

        print(f"Job {job_id} permanently failed.")

