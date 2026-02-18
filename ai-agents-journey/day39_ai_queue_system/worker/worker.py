import redis
import json
from shared.llm_service import generate_response

r = redis.Redis(host="redis", port=6379, decode_responses=True)

print("Worker started...")

while True:
    _, task = r.blpop("task_queue")
    data = json.loads(task)

    job_id = data["job_id"]
    query = data["query"]

    print(f"Processing job {job_id}")

    try:
        result = generate_response([
            {"role": "user", "content": query}
        ])
        r.set(f"result:{job_id}", result)
    except Exception as e:
        print("Worker error:", str(e))

    print("Saving result to Redis...")
    r.set(f"result:{job_id}", result)
    print("Saved.")

