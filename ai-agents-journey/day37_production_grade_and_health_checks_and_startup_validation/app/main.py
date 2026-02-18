from fastapi import FastAPI
from app.config import settings
from app.api.routes import router
from app.core.logging_config import setup_logging

logger = setup_logging()

app = FastAPI(title="AI Agent Service")

logger.info("AI Service Starting... ")

@app.on_event("startup")
def startup_event():
    logger.info("Validating configuration...")
    if not settings.GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY")
    logger.info("Configuration validated successfully.")

app.include_router(router)


# Expected Output:- 

# docker build -t ai-service (Build docker)
# docker run -p 8000:8000 -e GROQ_API_KEY=REMOVED --name ai-container-37 ai-service

# docker start ai-container-37 (to start the docker)
# docker stop ai-container-37 (to stop the docker)
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


