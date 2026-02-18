from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.agent import run_agent
import logging

logger = logging.getLogger("ai_service")

router  = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/agent/")
def agent_endpoint(request: QueryRequest):

    if not request.query:
        raise HTTPException(status_code=200, detail="Query is required.")

    try:
        result = run_agent(request.query)
        return{
            "query": request.query,
            "response": result
        }
    
    except Exception as e:
        logger.error(f"Endpoint failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal AI error")