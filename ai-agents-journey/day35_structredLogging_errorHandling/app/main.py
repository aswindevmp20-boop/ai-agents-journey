from fastapi import FastAPI
from app.api.routes import router
from app.core.logging_config import setup_logging

logger = setup_logging()

app = FastAPI(title="AI Agent Service")

logger.info("AI Service Starting... ")

app.include_router(router)


# Expected Output:- 

# uvicorn app.main:app --reload

# uvicorn app.main:app --reload
# INFO:     Will watch for changes in these directories: ['C:\\Users\\AswinDev\\OneDrive - DIGITAL BIZ SOLUTIONS PTE. LTD\\Documents\\ai-agents-journey\\ai-agents-journey\\day35_structredLogging_errorHandling']     
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# INFO:     Started reloader process [18336] using StatReload
######## 2026-02-12 19:20:07,552 | INFO | ai_service | AI Service Starting... 
# INFO:     Started server process [17964]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.



# 2026-02-12 19:22:13,468 | INFO | httpx | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
# 2026-02-12 19:22:13,495 | INFO | root | LLM call successful.
# INFO:     127.0.0.1:57991 - "POST /agent/ HTTP/1.1" 200 OK