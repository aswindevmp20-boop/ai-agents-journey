from fastapi import FastAPI
from app.api.routes import router
from app.core.logging_config import setup_logging

logger = setup_logging()

app = FastAPI(title="AI Agent Service")

logger.info("AI Service Starting... ")

app.include_router(router)


# Expected Output:- 

# docker build -t ai-service (Build docker)
# docker start ai-container (to start the docker)
# docker stop ai-container (to stop the docker)
# look of "http://localhost:8000/docs"


# 2026-02-13 06:28:59,623 | INFO | ai_service | AI Service Starting... 
# INFO:     Started server process [1]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit) 
# INFO:     172.17.0.1:36624 - "GET /docs HTTP/1.1" 200 OK
# INFO:     172.17.0.1:36624 - "GET /openapi.json HTTP/1.1" 200 OK
# 2026-02-13 06:30:16,315 | INFO | httpx | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
# 2026-02-13 06:30:16,344 | INFO | root | LLM call successful.
# INFO:     172.17.0.1:36636 - "POST /agent/ HTTP/1.1" 200 OK


